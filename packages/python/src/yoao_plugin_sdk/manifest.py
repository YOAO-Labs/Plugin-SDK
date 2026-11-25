from __future__ import annotations

import ipaddress
import json
import re
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator, model_validator

EXTENSION_MANIFEST_SCHEMA = "yoao-extension-manifest/v1"
EXTENSION_POINTS = (
    "inference",
    "notification_channel",
    "format_adapter",
    "trainer",
    "theme",
)

PLUGIN_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
OCI_DIGEST_PATTERN = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")
ENTRYPOINT_HEAD_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
_SHELL_EXECUTABLES = frozenset({"sh", "bash", "dash", "ash", "zsh", "ksh", "fish"})

_PROTOCOL_BY_POINT: dict[str, str] = {
    "inference": "yoao-inference/v2",
    "notification_channel": "yoao-notification-channel/v1",
    "format_adapter": "yoao-format-adapter/v1",
    "trainer": "yoao-trainer/v1",
    "theme": "yoao-theme/v1",
}

_CAPABILITIES_BY_POINT: dict[str, frozenset[str]] = {
    "inference": frozenset({"predict", "train", "track", "health"}),
    "notification_channel": frozenset({"deliver", "verify_address"}),
    "format_adapter": frozenset({"import", "export"}),
    "trainer": frozenset({"train", "evaluate", "preview", "export_weights"}),
    "theme": frozenset({"light", "dark"}),
}


def _has_control_chars(value: str) -> bool:
    return any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value)


def _is_forbidden_public_host(hostname: str) -> bool:
    lowered = hostname.lower().rstrip(".")
    if lowered == "localhost" or lowered.endswith(".localhost"):
        return True
    try:
        addr = ipaddress.ip_address(lowered)
        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
            addr = addr.ipv4_mapped
    except ValueError:
        return False
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
    if any(not arg or any(c in arg for c in "\x00\r\n") for arg in entrypoint):
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


class ExtensionResourceV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_mb: StrictInt = Field(ge=16, le=65536)


class ExtensionManifestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_schema: Literal["yoao-extension-manifest/v1"] = EXTENSION_MANIFEST_SCHEMA
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

    @field_validator("display_name")
    @classmethod
    def _validate_display_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned or len(cleaned) > 64 or _has_control_chars(value):
            raise ValueError("display_name must be 1..64 printable characters")
        return cleaned

    @field_validator("description")
    @classmethod
    def _validate_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned or len(cleaned) > 280 or _has_control_chars(value):
            raise ValueError("description must be 1..280 printable characters")
        return cleaned

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
        if self.extension_point != "theme":
            if not self.entrypoint:
                raise ValueError(f"{self.extension_point} manifest requires entrypoint")
            if self.resource is None:
                raise ValueError(f"{self.extension_point} manifest requires resource.memory_mb")
        return self

    def bind_artifact(self, digest_ref: str) -> ExtensionManifestV1:
        if self.artifact is not None:
            raise ValueError("image manifest must not embed artifact before installation")
        if not OCI_DIGEST_PATTERN.fullmatch(digest_ref):
            raise ValueError("artifact must be a digest-pinned OCI reference (@sha256:...)")
        return self.model_copy(update={"artifact": digest_ref})


def validate_manifest_dict(payload: dict[str, Any]) -> ExtensionManifestV1:
    return ExtensionManifestV1.model_validate(payload)


def load_manifest_file(path: str | Path) -> ExtensionManifestV1:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_manifest_dict(raw)
