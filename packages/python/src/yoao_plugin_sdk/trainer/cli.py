"""Reusable yoao-trainer/v1 CLI driver and TrainerPlugin protocol."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from yoao_plugin_sdk.trainer.errors import classify_failure, sanitize_failure_message
from yoao_plugin_sdk.trainer.files import atomic_write_json
from yoao_plugin_sdk.trainer.protocol import (
    TRAINER_PROTOCOL_VERSION,
    PreviewReportV1,
    PreviewSpecV1,
    TrainerCapabilitiesV1,
    TrainReceiptV1,
    TrainSpecV1,
    ValidateConfigResultV1,
    canonical_json_bytes,
)
from yoao_plugin_sdk.trainer.runtime import run_with_signal_receipt

if TYPE_CHECKING:
    import argparse


@runtime_checkable
class TrainerPlugin(Protocol):
    """Protocol implemented by yoao-trainer/v1 plugins driven by run_trainer_cli."""

    def describe(self) -> TrainerCapabilitiesV1: ...

    def validate_config(self, payload: Any) -> ValidateConfigResultV1: ...

    def train(
        self,
        spec: TrainSpecV1,
        output_dir: Path,
        *,
        spec_path: Path | None = None,
    ) -> TrainReceiptV1: ...


def build_trainer_parser(
    *,
    prog: str,
    description: str,
    supports_preview: bool = False,
) -> argparse.ArgumentParser:
    import argparse

    parser = argparse.ArgumentParser(prog=prog, description=description)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "describe",
        help="Emit TrainerCapabilitiesV1 canonical JSON without GPU or network access.",
    )

    validate_p = sub.add_parser(
        "validate-config",
        help="Validate a training configuration spec and emit ValidateConfigResultV1 JSON.",
    )
    validate_p.add_argument("--spec", type=Path, required=True)

    train_p = sub.add_parser(
        "train",
        help="Execute a frozen yoao-trainer/v1 training attempt.",
    )
    train_p.add_argument("--spec", type=Path, required=True)
    train_p.add_argument("--output", type=Path, required=True)

    if supports_preview:
        preview_p = sub.add_parser(
            "preview",
            help="Run CPU-only augmentation preview and write report.json.",
        )
        preview_p.add_argument("--spec", type=Path, required=True)
        preview_p.add_argument("--output", type=Path, required=True)

    return parser


def _print_canonical(payload: Any) -> None:
    sys.stdout.write(canonical_json_bytes(payload).decode("utf-8") + "\n")
    sys.stdout.flush()


def _detect_cuda_initialized() -> bool:
    torch_mod = sys.modules.get("torch")
    cuda_mod = getattr(torch_mod, "cuda", None)
    is_initialized = getattr(cuda_mod, "is_initialized", None)
    if callable(is_initialized):
        try:
            return bool(is_initialized())
        except Exception:
            return False
    return False


def _run_preview_command(
    plugin: TrainerPlugin,
    spec_path: Path,
    output_dir: Path,
    *,
    logger_prefix: str,
) -> int:
    preview_fn = getattr(plugin, "preview", None)
    if not callable(preview_fn):
        print(f"{logger_prefix}: plugin does not implement preview()", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)
    input_digest = "0" * 64
    policy_digest = "0" * 64
    try:
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
        spec = PreviewSpecV1.model_validate(raw)
        input_digest = spec.input_digest
        policy_digest = spec.policy_digest
        result = preview_fn(spec, output_dir, spec_path=spec_path)
        if isinstance(result, PreviewReportV1):
            report_path = output_dir / "report.json"
            if not report_path.is_file():
                atomic_write_json(report_path, result)
            return 0 if result.status == "succeeded" else 1
        if isinstance(result, int):
            return int(result)
        return 0
    except Exception as exc:
        code = classify_failure(exc)
        if code == "native_exception":
            code = "preview_failed"
        message = sanitize_failure_message(exc, type(exc).__name__)
        failed_report = PreviewReportV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            status="failed",
            input_digest=input_digest,
            policy_digest=policy_digest,
            cuda_initialized=_detect_cuda_initialized(),
            outputs=[],
            failure_code=code,
            failure_message=message,
        )
        try:
            atomic_write_json(output_dir / "report.json", failed_report)
        except Exception:
            pass
        print(f"{logger_prefix} preview failed ({code}): {message}", file=sys.stderr)
        return 1


def run_trainer_cli(
    plugin: TrainerPlugin,
    *,
    prog: str,
    description: str,
    supports_preview: bool = False,
    argv: Sequence[str] | None = None,
) -> int:
    parser = build_trainer_parser(
        prog=prog,
        description=description,
        supports_preview=supports_preview,
    )
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1

    if args.command == "describe":
        caps = plugin.describe()
        _print_canonical(caps)
        return 0

    if args.command == "validate-config":
        try:
            payload = json.loads(args.spec.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"Failed to read validation spec: {exc}", file=sys.stderr)
            return 2
        result = plugin.validate_config(payload)
        _print_canonical(result)
        return 0

    if args.command == "train":
        return run_with_signal_receipt(
            args.spec,
            args.output,
            lambda spec, output_dir, sp: plugin.train(spec, output_dir, spec_path=sp),
            logger_prefix=prog,
        )

    if args.command == "preview":
        return _run_preview_command(plugin, args.spec, args.output, logger_prefix=prog)

    return 2


__all__ = [
    "TrainerPlugin",
    "build_trainer_parser",
    "run_trainer_cli",
]
