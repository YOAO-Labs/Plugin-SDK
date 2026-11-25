from __future__ import annotations

import ipaddress
import json
import math
import re
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

EXTENSION_MANIFEST_SCHEMA = "yoao-extension-manifest/v1"
EXTENSION_POINTS = (
    "inference",
    "notification_channel",
    "format_adapter",
    "trainer",
    "theme",
)

INFERENCE_PROTOCOL_VERSION = "yoao-inference/v2"
INFERENCE_WIRE_PROTOCOL_VERSION = "2.0"
INFERENCE_EXECUTION_MODES = ("container", "direct")
INFERENCE_TASK_VALUES = (
    "detect",
    "segment",
    "classify",
    "pose",
    "obb",
    "ocr",
    "doc_layout",
    "tracker",
    "interactive_seg",
)
INFERENCE_CAPABILITIES = (
    "inference",
    "detection",
    "segmentation",
    "interactive",
    "video_tracking",
    "ocr",
    "classification",
    "obb",
    "pose",
    "doc_layout",
    "managed_models",
    "managed_evaluation",
)

PLUGIN_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
OCI_DIGEST_PATTERN = re.compile(
    r"^[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[0-9]+)?"
    r"(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*"
    r"(?::[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})?@sha256:[0-9a-f]{64}$"
)
SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ENTRYPOINT_HEAD_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
_NUMERIC_HOST_PATTERN = re.compile(
    r"^(?:[0-9]+|0[xX][0-9a-fA-F]+)(?:\.(?:[0-9]+|0[xX][0-9a-fA-F]+)){0,3}$"
)
_SHELL_EXECUTABLES = frozenset({"sh", "bash", "dash", "ash", "zsh", "ksh", "fish"})
_MAX_MANIFEST_BYTES = 65536

_PROTOCOL_BY_POINT: dict[str, str] = {
    "inference": INFERENCE_PROTOCOL_VERSION,
    "notification_channel": "yoao-notification-channel/v1",
    "format_adapter": "yoao-format-adapter/v1",
    "trainer": "yoao-trainer/v1",
    "theme": "yoao-theme/v1",
}

_CAPABILITIES_BY_POINT: dict[str, frozenset[str]] = {
    "inference": frozenset(INFERENCE_CAPABILITIES),
    "notification_channel": frozenset({"deliver", "verify_address"}),
    "format_adapter": frozenset({"import", "export"}),
    "trainer": frozenset({"train", "evaluate", "preview", "export_weights"}),
    "theme": frozenset({"light", "dark"}),
}


def _has_control_chars(value: str) -> bool:
    return any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value)


def _validate_printable_text(value: str, *, max_length: int, label: str) -> str:
    cleaned = value.strip()
    if not cleaned or len(cleaned) > max_length or _has_control_chars(value):
        raise ValueError(f"{label} must be 1..{max_length} printable characters")
    return cleaned


def _is_forbidden_public_host(hostname: str) -> bool:
    lowered = hostname.lower().rstrip(".")
    if lowered == "localhost" or lowered.endswith(".localhost"):
        return True
    try:
        addr = ipaddress.ip_address(lowered)
        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
            addr = addr.ipv4_mapped
    except ValueError:
        return bool(_NUMERIC_HOST_PATTERN.fullmatch(lowered))
    return (
        addr.is_loopback
        or addr.is_link_local
        or addr.is_private
        or addr.is_multicast
        or addr.is_unspecified
        or addr.is_reserved
        or not addr.is_global
    )


def validate_homepage(value: str | None) -> str | None:
    if value is None:
        return None
    if (
        not value
        or value != value.strip()
        or len(value) > 255
        or any(ord(ch) < 0x21 or ord(ch) > 0x7E for ch in value)
    ):
        raise ValueError("homepage must be a valid https:// URL up to 255 ASCII chars")
    parsed = urlsplit(value)
    if parsed.netloc.endswith(":"):
        raise ValueError("homepage must not contain an empty port")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("homepage contains an invalid port") from exc
    if port is not None and not (1 <= port <= 65535):
        raise ValueError("homepage port out of range")
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or _is_forbidden_public_host(parsed.hostname)
    ):
        raise ValueError("homepage must be a public https:// URL")
    return value


def validate_entrypoint(entrypoint: list[str]) -> list[str]:
    if not entrypoint or len(entrypoint) > 64:
        raise ValueError("entrypoint must contain 1..64 arguments")
    if any(
        not arg or not arg.strip() or any(c in arg for c in "\x00\r\n")
        for arg in entrypoint
    ):
        raise ValueError("entrypoint arguments must be non-empty single-line strings")
    head = entrypoint[0]
    if not ENTRYPOINT_HEAD_PATTERN.fullmatch(head) or head.lower() in _SHELL_EXECUTABLES:
        raise ValueError("entrypoint must not invoke a shell")
    if head.lower().startswith("python") and any(
        arg == "-c" or (arg.startswith("-") and not arg.startswith("--") and "c" in arg[1:])
        for arg in entrypoint[1:]
    ):
        raise ValueError("entrypoint must not invoke python -c")
    return entrypoint


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite_json_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON numbers must be finite")
    return number


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {value}")


def _loads_strict_json(raw: str) -> Any:
    encoded = raw.encode("utf-8")
    if len(encoded) > _MAX_MANIFEST_BYTES:
        raise ValueError(f"manifest exceeds {_MAX_MANIFEST_BYTES} bytes")
    if not raw.strip():
        raise ValueError("manifest must be a nonempty JSON string")
    return json.loads(
        raw,
        object_pairs_hook=_no_duplicate_keys,
        parse_float=_finite_json_float,
        parse_constant=_reject_json_constant,
    )


class ExtensionResourceV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_mb: StrictInt = Field(ge=16, le=65536)


class InferenceComputeV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: Literal["cpu", "cuda"]
    gpu_memory_mb: StrictInt = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_compute(self) -> InferenceComputeV1:
        if self.device == "cpu" and self.gpu_memory_mb != 0:
            raise ValueError("CPU compute must have gpu_memory_mb == 0")
        return self


class InferenceModelCapabilityV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: StrictStr = Field(min_length=1, max_length=128)
    task: Literal[
        "detect",
        "segment",
        "classify",
        "pose",
        "obb",
        "ocr",
        "doc_layout",
        "tracker",
        "interactive_seg",
    ]
    is_interactive: StrictBool
    supported_prompts: list[StrictStr] = Field(min_length=1)
    supported_inputs: list[StrictStr] = Field(min_length=1)
    supported_geometric_outputs: list[StrictStr]

    @model_validator(mode="after")
    def _validate_model_capabilities(self) -> InferenceModelCapabilityV1:
        if self.id.strip() != self.id:
            raise ValueError("model id must be canonical")
        for values in (
            self.supported_prompts,
            self.supported_inputs,
            self.supported_geometric_outputs,
        ):
            if any(not value or value.strip() != value for value in values):
                raise ValueError("capabilities must be nonempty canonical strings")
            if len(values) != len(set(values)):
                raise ValueError("capabilities must be unique")
        return self


class InferencePoolTopologyV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cap: StrictInt = Field(gt=0)


class InferenceManagedModelCompatibilityV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    framework: StrictStr = Field(min_length=1, max_length=64)
    format: StrictStr | None = Field(min_length=1, max_length=64)
    runtime_device: Literal["cpu", "cuda"]

    @model_validator(mode="after")
    def _validate_canonical(self) -> InferenceManagedModelCompatibilityV1:
        if self.framework.strip() != self.framework:
            raise ValueError("framework must be canonical")
        if self.format is not None and self.format.strip() != self.format:
            raise ValueError("format must be canonical")
        return self


class InferenceExtensionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_mode: Literal["container", "direct"] = "container"
    models: list[InferenceModelCapabilityV1] = Field(min_length=1, max_length=128)
    pools: dict[str, InferencePoolTopologyV1] = Field(min_length=1, max_length=128)
    compute: InferenceComputeV1
    sdk_compatibility: StrictStr | None = Field(default=None, min_length=1, max_length=255)
    asset_manifest_sha256: StrictStr | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    allowed_asset_sources: list[StrictStr] = Field(default_factory=list)
    managed_model_compatibility: list[InferenceManagedModelCompatibilityV1] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def _validate_inference_extension(self) -> InferenceExtensionV1:
        if len({model.id for model in self.models}) != len(self.models):
            raise ValueError("model ids must be unique")
        if any(
            not pool or len(pool) > 128 or pool.strip() != pool for pool in self.pools
        ):
            raise ValueError("pool ids must be nonempty and canonical")
        if self.sdk_compatibility is not None and not self.sdk_compatibility.strip():
            raise ValueError("sdk_compatibility must not be blank")
        if any(not item or not item.strip() for item in self.allowed_asset_sources):
            raise ValueError("allowed_asset_sources must not contain blank entries")
        if any(
            item.runtime_device != self.compute.device
            for item in self.managed_model_compatibility
        ):
            raise ValueError("managed_model_compatibility runtime_device must match compute.device")
        return self


class ExtensionManifestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_schema: Literal["yoao-extension-manifest/v1"]
    extension_point: Literal[
        "inference", "notification_channel", "format_adapter", "trainer", "theme"
    ]
    plugin_id: StrictStr = Field(min_length=1, max_length=128)
    version: StrictStr = Field(min_length=1, max_length=128)
    display_name: StrictStr | None = None
    description: StrictStr | None = None
    homepage: StrictStr | None = None
    publisher: StrictStr = Field(min_length=1, max_length=255)
    license: StrictStr = Field(min_length=1, max_length=255)
    protocol_version: StrictStr = Field(min_length=1, max_length=64)
    entrypoint: list[StrictStr] | None = None
    capabilities: list[StrictStr] = Field(min_length=1, max_length=16)
    resource: ExtensionResourceV1 | None = None
    extension: dict[str, Any] = Field(default_factory=dict)
    artifact: StrictStr | None = None

    @field_validator("plugin_id")
    @classmethod
    def _validate_plugin_id(cls, value: str) -> str:
        if not PLUGIN_ID_PATTERN.fullmatch(value):
            raise ValueError("plugin_id must match ^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
        return value

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: str) -> str:
        return _validate_printable_text(value, max_length=128, label="version")

    @field_validator("publisher")
    @classmethod
    def _validate_publisher(cls, value: str) -> str:
        return _validate_printable_text(value, max_length=255, label="publisher")

    @field_validator("license")
    @classmethod
    def _validate_license(cls, value: str) -> str:
        return _validate_printable_text(value, max_length=255, label="license")

    @field_validator("display_name")
    @classmethod
    def _validate_display_name(cls, value: str | None) -> str | None:
        return None if value is None else _validate_printable_text(value, max_length=64, label="display_name")

    @field_validator("description")
    @classmethod
    def _validate_description(cls, value: str | None) -> str | None:
        return None if value is None else _validate_printable_text(value, max_length=280, label="description")

    @field_validator("homepage")
    @classmethod
    def _validate_homepage(cls, value: str | None) -> str | None:
        return validate_homepage(value)

    @field_validator("entrypoint")
    @classmethod
    def _validate_entrypoint(cls, value: list[str] | None) -> list[str] | None:
        return None if value is None else validate_entrypoint(value)

    @field_validator("artifact")
    @classmethod
    def _validate_artifact(cls, value: str | None) -> str | None:
        if value is not None and not OCI_DIGEST_PATTERN.fullmatch(value):
            raise ValueError("artifact must be a digest-pinned OCI reference (@sha256:...)")
        return value

    @model_validator(mode="after")
    def _validate_extension_point_rules(self) -> ExtensionManifestV1:
        if "source_tier" in self.extension:
            raise ValueError("manifest must not self-declare source_tier")
        expected_protocol = _PROTOCOL_BY_POINT[self.extension_point]
        if self.protocol_version != expected_protocol:
            raise ValueError(
                f"extension_point {self.extension_point!r} requires protocol_version {expected_protocol!r}"
            )
        allowed_caps = _CAPABILITIES_BY_POINT[self.extension_point]
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("capabilities must not contain duplicates")
        invalid = set(self.capabilities) - allowed_caps
        if invalid:
            raise ValueError(f"unsupported capabilities for {self.extension_point}: {sorted(invalid)}")
        if self.extension_point == "inference":
            if "inference" not in self.capabilities:
                raise ValueError("inference manifest must include 'inference' capability")
            parsed_ext = InferenceExtensionV1.model_validate(self.extension)
            if ("managed_models" in self.capabilities) != bool(
                parsed_ext.managed_model_compatibility
            ):
                raise ValueError(
                    "managed_models capability and managed_model_compatibility declarations must agree"
                )
            if (
                "managed_evaluation" in self.capabilities
                and "managed_models" not in self.capabilities
            ):
                raise ValueError(
                    "managed_evaluation capability requires managed_models capability"
                )
            if parsed_ext.execution_mode == "container":
                if not self.entrypoint:
                    raise ValueError("container inference manifest requires entrypoint")
                if self.resource is None:
                    raise ValueError("container inference manifest requires resource.memory_mb")
                if parsed_ext.asset_manifest_sha256 is None:
                    raise ValueError(
                        "container inference manifest requires extension.asset_manifest_sha256"
                    )
                if (
                    parsed_ext.compute.device == "cuda"
                    and parsed_ext.compute.gpu_memory_mb <= 0
                ):
                    raise ValueError(
                        "container CUDA inference manifest requires extension.compute.gpu_memory_mb > 0"
                    )
            elif self.artifact is not None:
                raise ValueError("direct inference manifest must not carry an OCI artifact")
        elif self.extension_point != "theme":
            if not self.entrypoint:
                raise ValueError(f"{self.extension_point} manifest requires entrypoint")
            if self.resource is None:
                raise ValueError(f"{self.extension_point} manifest requires resource.memory_mb")
        return self

    def bind_artifact(self, digest_ref: str) -> ExtensionManifestV1:
        if self.artifact is not None:
            raise ValueError("image manifest must not embed artifact before installation")
        if (
            self.extension_point == "inference"
            and self.extension.get("execution_mode") == "direct"
        ):
            raise ValueError("direct inference manifest cannot bind an OCI artifact")
        if not OCI_DIGEST_PATTERN.fullmatch(digest_ref):
            raise ValueError("artifact must be a digest-pinned OCI reference (@sha256:...)")
        return self.model_copy(update={"artifact": digest_ref})


def validate_manifest_dict(payload: dict[str, Any]) -> ExtensionManifestV1:
    return ExtensionManifestV1.model_validate(payload)


def load_manifest_file(path: str | Path) -> ExtensionManifestV1:
    raw = _loads_strict_json(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("manifest must be a JSON object")
    return validate_manifest_dict(raw)
