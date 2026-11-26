"""Trainer runtime helpers: event streaming, checkpoint slots, artifacts, and signals."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import signal
import sys
from typing import Any, BinaryIO, Callable, Literal

from yoao_plugin_sdk.trainer.errors import (
    InputMismatchError,
    NativeConfigurationMismatch,
    classify_failure,
    sanitize_failure_message,
)
from yoao_plugin_sdk.trainer.files import (
    atomic_write_json,
    digest_file,
    immutable_copy,
)
from yoao_plugin_sdk.trainer.protocol import (
    TRAINER_PROTOCOL_VERSION,
    TrainArtifactV1,
    TrainEventV1,
    TrainReceiptV1,
    TrainSpecV1,
    write_event_jsonl,
)

SUPERVISOR_OWNED_ARTIFACTS = frozenset(
    {"manifest.json", "train.log", "observability.json", "config.json", "result.json"}
)

_CHECKPOINT_EXT_RE = re.compile(r"^\.[a-zA-Z0-9_-]{1,32}$")


class _CancelledTraining(BaseException):
    """Internal signal cancellation carrier not caught by except Exception."""


class EventEmitter:
    """Emits strictly monotonic TrainEventV1 JSONL lines."""

    def __init__(self, output_dir_or_path: str | Path) -> None:
        target = Path(output_dir_or_path)
        if target.suffix == ".jsonl" or (target.exists() and target.is_file()):
            self.path = target
        else:
            self.path = target / "events.jsonl"
        self._seq = 0

    @property
    def seq(self) -> int:
        return self._seq

    def emit(
        self,
        kind: Literal["phase", "epoch", "progress", "checkpoint", "warning"],
        *,
        phase: str | None = None,
        epoch: int | None = None,
        total_epochs: int | None = None,
        progress: float | None = None,
        metrics: dict[str, Any] | None = None,
        detail: dict[str, Any] | None = None,
    ) -> TrainEventV1:
        next_seq = self._seq + 1
        event = TrainEventV1(
            seq=next_seq,
            kind=kind,
            phase=phase,
            epoch=epoch,
            total_epochs=total_epochs,
            progress=progress,
            metrics=metrics or {},
            detail=detail or {},
        )
        write_event_jsonl(self.path, event)
        self._seq = next_seq
        return event

    def phase(
        self,
        phase: str,
        *,
        total_epochs: int | None = None,
        metrics: dict[str, Any] | None = None,
        detail: dict[str, Any] | None = None,
    ) -> TrainEventV1:
        return self.emit(
            "phase",
            phase=phase,
            total_epochs=total_epochs,
            metrics=metrics,
            detail=detail,
        )

    def epoch(
        self,
        *,
        epoch: int,
        total_epochs: int,
        metrics: dict[str, Any] | None = None,
        progress: float | None = None,
        phase: str = "training",
        detail: dict[str, Any] | None = None,
    ) -> TrainEventV1:
        computed_progress = (
            min(1.0, max(0.0, float(epoch) / float(total_epochs)))
            if progress is None
            else progress
        )
        return self.emit(
            "epoch",
            phase=phase,
            epoch=epoch,
            total_epochs=total_epochs,
            progress=computed_progress,
            metrics=metrics,
            detail=detail,
        )

    def progress(
        self,
        *,
        progress: float,
        phase: str | None = "training",
        epoch: int | None = None,
        total_epochs: int | None = None,
        metrics: dict[str, Any] | None = None,
        detail: dict[str, Any] | None = None,
    ) -> TrainEventV1:
        return self.emit(
            "progress",
            phase=phase,
            epoch=epoch,
            total_epochs=total_epochs,
            progress=progress,
            metrics=metrics,
            detail=detail,
        )

    def checkpoint(
        self,
        *,
        epoch: int,
        filename: str,
        sha256: str,
        total_epochs: int | None = None,
        phase: str = "training",
        detail: dict[str, Any] | None = None,
    ) -> TrainEventV1:
        merged = dict(detail) if detail else {}
        merged["filename"] = filename
        merged["sha256"] = sha256
        return self.emit(
            "checkpoint",
            phase=phase,
            epoch=epoch,
            total_epochs=total_epochs,
            detail=merged,
        )

    def warning(
        self,
        message: str,
        *,
        phase: str | None = None,
        epoch: int | None = None,
        total_epochs: int | None = None,
        detail: dict[str, Any] | None = None,
    ) -> TrainEventV1:
        merged = dict(detail) if detail else {}
        merged["message"] = message
        return self.emit(
            "warning",
            phase=phase,
            epoch=epoch,
            total_epochs=total_epochs,
            detail=merged,
        )


def build_artifact_entry(
    role: Literal["weights", "checkpoint", "metrics", "config", "input_membership", "extra"],
    path: str | Path,
) -> TrainArtifactV1:
    target = Path(path)
    if target.name in SUPERVISOR_OWNED_ARTIFACTS:
        raise NativeConfigurationMismatch(
            f"Refusing to declare supervisor-owned artifact {target.name!r} in TrainReceiptV1"
        )
    return TrainArtifactV1(
        role=role,
        filename=target.name,
        sha256=digest_file(target),
        size_bytes=target.stat().st_size,
    )


class CheckpointSlotStore:
    """Crash-safe rotating checkpoint store with atomic checkpoint.json updates."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        extension: str = ".pt",
        metadata_filename: str = "checkpoint.json",
    ) -> None:
        if not isinstance(extension, str) or not _CHECKPOINT_EXT_RE.fullmatch(extension):
            raise ValueError(
                "Checkpoint extension must match ^\\.[a-zA-Z0-9_-]{1,32}$"
            )
        if (
            not isinstance(metadata_filename, str)
            or not metadata_filename.strip()
            or "/" in metadata_filename
            or "\\" in metadata_filename
            or metadata_filename in {".", ".."}
            or metadata_filename in SUPERVISOR_OWNED_ARTIFACTS
        ):
            raise ValueError(f"Invalid checkpoint metadata filename: {metadata_filename!r}")
        self.output_dir = Path(output_dir)
        self.extension = extension
        self.metadata_filename = metadata_filename
        self.current: dict[str, Any] | None = None

    @staticmethod
    def _validate_epoch(epoch: int) -> int:
        if type(epoch) is not int or not 1 <= epoch <= 9999:
            raise ValueError("Checkpoint epoch must be an integer in 1..9999")
        return epoch

    def slot_filename(self, epoch: int) -> str:
        validated = self._validate_epoch(epoch)
        return f"checkpoint-{validated:04d}{self.extension}"

    def commit_file(self, source: str | Path, epoch: int) -> dict[str, Any]:
        validated = self._validate_epoch(epoch)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        previous = self.current
        info = immutable_copy(
            Path(source),
            self.output_dir / self.slot_filename(validated),
        )
        record: dict[str, Any] = {
            **info,
            "role": "checkpoint",
            "epoch": validated,
        }
        atomic_write_json(self.output_dir / self.metadata_filename, record)
        self.current = record
        if previous is not None and previous["name"] != record["name"]:
            (self.output_dir / previous["name"]).unlink(missing_ok=True)
        return dict(record)

    def commit_via_writer(
        self,
        epoch: int,
        write_fn: Callable[[BinaryIO], None],
    ) -> dict[str, Any]:
        validated = self._validate_epoch(epoch)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        staging = self.output_dir / f"checkpoint-staging{self.extension}"
        try:
            with staging.open("wb") as stream:
                write_fn(stream)
                stream.flush()
                os.fsync(stream.fileno())
            return self.commit_file(staging, validated)
        finally:
            staging.unlink(missing_ok=True)


def load_dataset_identity(spec_path: str | Path | None) -> dict[str, Any]:
    if spec_path is None:
        return {}
    sidecar = Path(spec_path).parent / "dataset-identity.json"
    if not sidecar.is_file():
        return {}
    try:
        loaded = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InputMismatchError(f"Invalid dataset-identity.json: {exc}") from exc
    if not isinstance(loaded, dict):
        raise InputMismatchError("dataset-identity.json must be a JSON object")
    return loaded


def _write_terminal_receipt(output_dir: Path, receipt: TrainReceiptV1) -> None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output_dir / "result.json", receipt)
    except Exception:
        pass


def run_with_signal_receipt(
    spec_path: str | Path,
    output_dir: str | Path,
    run_fn: Callable[[TrainSpecV1, Path, Path], TrainReceiptV1],
    *,
    logger_prefix: str = "yoao-trainer",
) -> int:
    sp = Path(spec_path)
    out_dir = Path(output_dir)
    previous_handlers: dict[int, Any] = {}

    def _handle_cancel(signum: int, _frame: Any) -> None:
        raise _CancelledTraining(f"Received signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _handle_cancel)
        except (ValueError, OSError):
            pass

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        raw = json.loads(sp.read_text(encoding="utf-8"))
        spec = TrainSpecV1.model_validate(raw)
        receipt = run_fn(spec, out_dir, sp)
        if isinstance(receipt, TrainReceiptV1):
            result_path = out_dir / "result.json"
            if not result_path.is_file():
                atomic_write_json(result_path, receipt)
            if receipt.status == "cancelled":
                return 130
            if receipt.status == "failed":
                return 1
        return 0
    except (_CancelledTraining, KeyboardInterrupt) as exc:
        receipt = TrainReceiptV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            status="cancelled",
            artifacts=[],
            failure_message=sanitize_failure_message(exc, "Training cancelled"),
        )
        _write_terminal_receipt(out_dir, receipt)
        return 130
    except Exception as exc:
        code = classify_failure(exc)
        message = sanitize_failure_message(exc, type(exc).__name__)
        receipt = TrainReceiptV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            status="failed",
            artifacts=[],
            failure_code=code,
            failure_message=message,
        )
        _write_terminal_receipt(out_dir, receipt)
        print(f"{logger_prefix} failed ({code}): {message}", file=sys.stderr)
        return 1
    finally:
        for sig, handler in previous_handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass


__all__ = [
    "SUPERVISOR_OWNED_ARTIFACTS",
    "CheckpointSlotStore",
    "EventEmitter",
    "build_artifact_entry",
    "load_dataset_identity",
    "run_with_signal_receipt",
]
