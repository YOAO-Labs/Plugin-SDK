from __future__ import annotations

import json
import os
from pathlib import Path
import signal
from typing import Any

import pytest

from yoao_plugin_sdk.trainer import (
    TRAINER_PROTOCOL_VERSION,
    PreviewOutputV1,
    PreviewReportV1,
    PreviewSourceV1,
    PreviewSpecV1,
    TrainerArchitectureV1,
    TrainerCapabilitiesV1,
    TrainerPlugin,
    TrainReceiptV1,
    TrainSpecV1,
    ValidateConfigResultV1,
    ValidationIssueV1,
    build_artifact_entry,
    canonical_json_bytes,
    digest_file,
    run_trainer_cli,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
DIGEST_D = "d" * 64
DIGEST_E = "e" * 64


class _StubPlugin:
    def __init__(self, *, mode: str = "ok") -> None:
        self.mode = mode

    def describe(self) -> TrainerCapabilitiesV1:
        return TrainerCapabilitiesV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            framework_id="stub",
            framework_version="0.1.0",
            runtime_versions={"python": "3.12"},
            supported_tasks=["detect"],
            supported_dataset_contracts=["ultralytics-bbox-1"],
            architectures=[
                TrainerArchitectureV1(id="stub-n", family="stub", tasks=["detect"])
            ],
        )

    def validate_config(self, payload: Any) -> ValidateConfigResultV1:
        if not isinstance(payload, dict) or payload.get("task") != "detect":
            return ValidateConfigResultV1(
                protocol_version=TRAINER_PROTOCOL_VERSION,
                valid=False,
                issues=[ValidationIssueV1(code="unsupported_task", field="task", message="only detect")],
            )
        return ValidateConfigResultV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            valid=True,
            schema_id="stub-v1",
            schema_digest=DIGEST_A,
            resolved_config={"task": "detect"},
            effective_values={"epochs": 1},
            origins={"epochs": "default"},
        )

    def train(
        self,
        spec: TrainSpecV1,
        output_dir: Path,
        *,
        spec_path: Path | None = None,
    ) -> TrainReceiptV1:
        if self.mode == "cancel":
            os.kill(os.getpid(), signal.SIGTERM)
        if self.mode == "fail":
            raise RuntimeError("training exploded")
        weights = output_dir / "best.pt"
        weights.write_bytes(b"stub-weights")
        return TrainReceiptV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            status="succeeded",
            artifacts=[build_artifact_entry("weights", weights)],
        )

    def preview(
        self,
        spec: PreviewSpecV1,
        output_dir: Path,
        *,
        spec_path: Path | None = None,
    ) -> PreviewReportV1:
        if self.mode == "fail":
            raise RuntimeError("preview exploded")
        out_img = output_dir / "preview-0.png"
        out_img.write_bytes(b"png-bytes")
        return PreviewReportV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            status="succeeded",
            input_digest=spec.input_digest,
            policy_digest=spec.policy_digest,
            cuda_initialized=False,
            outputs=[
                PreviewOutputV1(
                    filename=out_img.name,
                    sha256=digest_file(out_img),
                    size_bytes=out_img.stat().st_size,
                    width=64,
                    height=64,
                    box_count=1,
                )
            ],
        )


def _train_spec(dataset_dir: Path) -> dict[str, Any]:
    return TrainSpecV1(
        protocol_version=TRAINER_PROTOCOL_VERSION,
        run_id="run-1",
        attempt_id="attempt-1",
        framework_id="stub",
        task="detect",
        architecture_id="stub-n",
        dataset_contract="ultralytics-bbox-1",
        dataset_dir=str(dataset_dir),
        dataset_snapshot_digest=DIGEST_A,
        config_digest=DIGEST_B,
        classes_digest=DIGEST_C,
    ).model_dump(mode="json")


def _preview_spec() -> dict[str, Any]:
    return PreviewSpecV1(
        protocol_version=TRAINER_PROTOCOL_VERSION,
        preview_id="prev-1",
        framework_id="stub",
        task="detect",
        architecture_id="stub-n",
        input_digest=DIGEST_E,
        config_digest=DIGEST_A,
        dataset_snapshot_digest=DIGEST_B,
        policy_digest=DIGEST_C,
        seed=42,
        epoch=1,
        count=1,
        sources=[
            PreviewSourceV1(
                task_id="task-1",
                filename="images/train/a.jpg",
                sha256=DIGEST_D,
                width=64,
                height=64,
                size_bytes=100,
                box_count=1,
            )
        ],
    ).model_dump(mode="json")


def test_trainer_cli_describe_and_validate_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plugin = _StubPlugin()
    assert isinstance(plugin, TrainerPlugin)

    rc = run_trainer_cli(
        plugin, prog="stub-trainer", description="Stub", argv=["describe"]
    )
    assert rc == 0
    raw = capsys.readouterr().out
    caps = TrainerCapabilitiesV1.model_validate_json(raw)
    assert raw == canonical_json_bytes(caps).decode("utf-8") + "\n"

    valid_spec = tmp_path / "valid.json"
    valid_spec.write_text(json.dumps({"task": "detect"}), encoding="utf-8")
    assert (
        run_trainer_cli(
            plugin,
            prog="stub-trainer",
            description="Stub",
            argv=["validate-config", "--spec", str(valid_spec)],
        )
        == 0
    )
    res = ValidateConfigResultV1.model_validate_json(capsys.readouterr().out)
    assert res.valid is True and res.schema_id == "stub-v1"

    bad_spec = tmp_path / "bad.json"
    bad_spec.write_text("{not-json", encoding="utf-8")
    assert (
        run_trainer_cli(
            plugin,
            prog="stub-trainer",
            description="Stub",
            argv=["validate-config", "--spec", str(bad_spec)],
        )
        == 2
    )
    assert (
        run_trainer_cli(
            plugin,
            prog="stub-trainer",
            description="Stub",
            argv=["validate-config", "--spec", str(tmp_path / "missing.json")],
        )
        == 2
    )


def test_trainer_cli_train_and_preview_exit_codes(tmp_path: Path) -> None:
    spec_path = tmp_path / "train-spec.json"
    spec_path.write_text(json.dumps(_train_spec(tmp_path / "ds")), encoding="utf-8")

    out_ok = tmp_path / "out-ok"
    assert (
        run_trainer_cli(
            _StubPlugin(mode="ok"),
            prog="stub-trainer",
            description="Stub",
            argv=["train", "--spec", str(spec_path), "--output", str(out_ok)],
        )
        == 0
    )
    assert TrainReceiptV1.model_validate_json(
        (out_ok / "result.json").read_text(encoding="utf-8")
    ).status == "succeeded"

    out_fail = tmp_path / "out-fail"
    assert (
        run_trainer_cli(
            _StubPlugin(mode="fail"),
            prog="stub-trainer",
            description="Stub",
            argv=["train", "--spec", str(spec_path), "--output", str(out_fail)],
        )
        == 1
    )
    assert TrainReceiptV1.model_validate_json(
        (out_fail / "result.json").read_text(encoding="utf-8")
    ).status == "failed"

    out_cancel = tmp_path / "out-cancel"
    assert (
        run_trainer_cli(
            _StubPlugin(mode="cancel"),
            prog="stub-trainer",
            description="Stub",
            argv=["train", "--spec", str(spec_path), "--output", str(out_cancel)],
        )
        == 130
    )
    assert TrainReceiptV1.model_validate_json(
        (out_cancel / "result.json").read_text(encoding="utf-8")
    ).status == "cancelled"

    prev_spec = tmp_path / "preview-spec.json"
    prev_spec.write_text(json.dumps(_preview_spec()), encoding="utf-8")

    # Rejects preview when supports_preview=False.
    assert (
        run_trainer_cli(
            _StubPlugin(mode="ok"),
            prog="stub-trainer",
            description="Stub",
            supports_preview=False,
            argv=["preview", "--spec", str(prev_spec), "--output", str(tmp_path / "no-prev")],
        )
        == 2
    )

    out_prev = tmp_path / "prev-ok"
    assert (
        run_trainer_cli(
            _StubPlugin(mode="ok"),
            prog="stub-trainer",
            description="Stub",
            supports_preview=True,
            argv=["preview", "--spec", str(prev_spec), "--output", str(out_prev)],
        )
        == 0
    )
    report = PreviewReportV1.model_validate_json(
        (out_prev / "report.json").read_text(encoding="utf-8")
    )
    assert report.status == "succeeded" and report.cuda_initialized is False

    out_prev_fail = tmp_path / "prev-fail"
    assert (
        run_trainer_cli(
            _StubPlugin(mode="fail"),
            prog="stub-trainer",
            description="Stub",
            supports_preview=True,
            argv=["preview", "--spec", str(prev_spec), "--output", str(out_prev_fail)],
        )
        == 1
    )
    failed_report = PreviewReportV1.model_validate_json(
        (out_prev_fail / "report.json").read_text(encoding="utf-8")
    )
    assert failed_report.status == "failed" and failed_report.failure_code == "preview_failed"

    # Dirty failure_code on preview exception is sanitized so PreviewReportV1 validates cleanly.
    from yoao_plugin_sdk.trainer import TrainerError

    class _DirtyPreviewPlugin(_StubPlugin):
        def preview(self, spec: PreviewSpecV1, output_dir: Path, *, spec_path: Path | None = None) -> PreviewReportV1:
            raise TrainerError("bad preview", failure_code="\x00dirty\ncode " + ("z" * 100))

    out_prev_dirty = tmp_path / "prev-dirty"
    assert (
        run_trainer_cli(
            _DirtyPreviewPlugin(),
            prog="stub-trainer",
            description="Stub",
            supports_preview=True,
            argv=["preview", "--spec", str(prev_spec), "--output", str(out_prev_dirty)],
        )
        == 1
    )
    dirty_report = PreviewReportV1.model_validate_json(
        (out_prev_dirty / "report.json").read_text(encoding="utf-8")
    )
    assert dirty_report.status == "failed"
    assert dirty_report.failure_code is not None
    assert len(dirty_report.failure_code) <= 64
    assert not any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in dirty_report.failure_code)
