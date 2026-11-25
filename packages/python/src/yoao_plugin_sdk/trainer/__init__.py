from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

TRAINER_PROTOCOL_VERSION = "yoao-trainer/v1"


class ServingCompatibilityV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    framework: StrictStr = Field(min_length=1, max_length=64)
    format: StrictStr | None = Field(min_length=1, max_length=64)
    training_runtime: dict[str, Any] = Field(default_factory=dict)
    compatible_serving_runtimes: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("framework")
    @classmethod
    def _validate_framework(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("framework must be a non-empty trimmed string")
        return value

    @field_validator("format")
    @classmethod
    def _validate_format(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip() or value != value.strip():
            raise ValueError("format must be null or a non-empty trimmed string")
        return value


class TrainerArchitectureV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: StrictStr = Field(min_length=1, max_length=128)
    family: StrictStr = Field(min_length=1, max_length=64)
    tasks: list[StrictStr] = Field(min_length=1)
    end2end: StrictBool = False
    base_asset_filename: StrictStr | None = None
    base_asset_sha256: StrictStr | None = None


class TrainerCapabilitiesV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal["yoao-trainer/v1"] = TRAINER_PROTOCOL_VERSION
    framework_id: StrictStr = Field(min_length=1, max_length=64)
    framework_version: StrictStr = Field(min_length=1, max_length=64)
    runtime_versions: dict[str, StrictStr] = Field(default_factory=dict)
    supported_tasks: list[StrictStr] = Field(min_length=1)
    supported_dataset_contracts: list[StrictStr] = Field(min_length=1)
    architectures: list[TrainerArchitectureV1] = Field(min_length=1)
    parameter_schema: dict[str, Any] = Field(default_factory=dict)
    evaluator_descriptors: dict[str, dict[str, Any]] = Field(default_factory=dict)
    policy_digests: dict[str, StrictStr] = Field(default_factory=dict)
    serving_compatibility: list[ServingCompatibilityV1] = Field(default_factory=list)


class TrainSpecV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal["yoao-trainer/v1"] = TRAINER_PROTOCOL_VERSION
    run_id: StrictStr = Field(min_length=1, max_length=128)
    attempt_id: StrictStr = Field(min_length=1, max_length=128)
    framework_id: StrictStr = Field(min_length=1, max_length=64)
    task: StrictStr = Field(min_length=1, max_length=64)
    architecture_id: StrictStr = Field(min_length=1, max_length=128)
    dataset_contract: StrictStr = Field(min_length=1, max_length=128)
    dataset_dir: StrictStr = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    base_weights_path: StrictStr | None = None
    resume_checkpoint_path: StrictStr | None = None
    init_weights_path: StrictStr | None = None
    native_budget: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_initialization_exclusivity(self) -> TrainSpecV1:
        if self.resume_checkpoint_path is not None and self.init_weights_path is not None:
            raise ValueError(
                "resume_checkpoint_path and init_weights_path are mutually exclusive"
            )
        return self


class TrainEventV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seq: StrictInt = Field(ge=1)
    kind: Literal["phase", "epoch", "progress", "checkpoint", "warning"]
    phase: StrictStr | None = None
    epoch: StrictInt | None = Field(default=None, ge=0)
    total_epochs: StrictInt | None = Field(default=None, ge=1)
    progress: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    metrics: dict[str, Any] = Field(default_factory=dict)
    detail: dict[str, Any] = Field(default_factory=dict)


class TrainArtifactV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["weights", "checkpoint", "metrics", "config", "input_membership", "extra"]
    filename: StrictStr = Field(min_length=1, max_length=255)
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: StrictInt = Field(ge=0)


class TrainReceiptV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal["yoao-trainer/v1"] = TRAINER_PROTOCOL_VERSION
    status: Literal["succeeded", "failed", "cancelled"]
    artifacts: list[TrainArtifactV1] = Field(default_factory=list)
    serving_compatibility: ServingCompatibilityV1 | None = None
    failure_code: StrictStr | None = None
    failure_message: StrictStr | None = None


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        dir_fd = os.open(str(directory), flags)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def atomic_write_json(path: str | Path, payload: BaseModel | dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    encoded = (
        json.dumps(
            data,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=str(target.parent))
    fd_open = True
    try:
        with os.fdopen(fd, "wb") as stream:
            fd_open = False
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, target)
        _fsync_directory(target.parent)
    except BaseException:
        if fd_open:
            os.close(fd)
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def write_event_jsonl(path: str | Path, event: TrainEventV1) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = (
        json.dumps(
            event.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    with target.open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


__all__ = [
    "TRAINER_PROTOCOL_VERSION",
    "ServingCompatibilityV1",
    "TrainArtifactV1",
    "TrainEventV1",
    "TrainReceiptV1",
    "TrainSpecV1",
    "TrainerArchitectureV1",
    "TrainerCapabilitiesV1",
    "atomic_write_json",
    "write_event_jsonl",
]
