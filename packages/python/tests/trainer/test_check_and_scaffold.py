from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from yoao_plugin_sdk.trainer._scaffold import init_trainer_example
from yoao_plugin_sdk.trainer.check import TrainerCheckReport, check_trainer_plugin


def _write_mutated_trainer(target: Path, defect: str) -> list[str]:
    script = target / f"buggy_{defect}.py"
    script.write_text(
        f'''from __future__ import annotations
import json
from pathlib import Path
import signal
import sys
import time

from yoao_plugin_sdk.trainer import (
    POLICY_DIGEST,
    TRAINER_PROTOCOL_VERSION,
    CheckpointSlotStore,
    EventEmitter,
    InputMembership,
    PreviewOutputV1,
    PreviewReportV1,
    PreviewSpecV1,
    ServingCompatibilityV1,
    TrainArtifactV1,
    TrainerArchitectureV1,
    TrainerCapabilitiesV1,
    TrainEventV1,
    TrainReceiptV1,
    TrainSpecV1,
    ValidateConfigResultV1,
    ValidationIssueV1,
    atomic_write_json,
    build_artifact_entry,
    canonical_json_bytes,
    canonical_json_digest,
    digest_file,
    load_dataset_identity,
    run_trainer_cli,
    validate_frozen_manifest_core,
    validate_sample_files,
    write_event_jsonl,
)

DEFECT = {defect!r}
SCHEMA_ID = "buggy-detect-1"
SCHEMA_DIGEST = canonical_json_digest({{"schema": SCHEMA_ID}})


class BuggyTrainer:
    def describe(self) -> TrainerCapabilitiesV1:
        contracts = (
            ["custom-seg-only-1", "ultralytics-bbox-1"]
            if DEFECT == "contract_fallback"
            else ["ultralytics-bbox-1"]
        )
        return TrainerCapabilitiesV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            framework_id="buggy",
            framework_version="0.1.0",
            runtime_versions={{"python": "3.12"}},
            supported_tasks=["detect"],
            supported_dataset_contracts=contracts,
            architectures=[
                TrainerArchitectureV1(id="buggy-n", family="buggy", tasks=["detect"])
            ],
            policy_digests={{"input_membership": POLICY_DIGEST}},
            serving_compatibility=[ServingCompatibilityV1(framework="buggy", format=None)],
        )

    def validate_config(self, payload):
        if (
            not isinstance(payload, dict)
            or payload.get("task") != "detect"
            or payload.get("architecture_id") != "buggy-n"
            or payload.get("dataset_contract") != "ultralytics-bbox-1"
        ):
            return ValidateConfigResultV1(
                protocol_version=TRAINER_PROTOCOL_VERSION,
                valid=False,
                schema_id=SCHEMA_ID,
                schema_digest=SCHEMA_DIGEST,
                issues=[ValidationIssueV1(code="invalid", field="dataset_contract", message="unsupported")],
            )
        return ValidateConfigResultV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            valid=True,
            schema_id=SCHEMA_ID,
            schema_digest=SCHEMA_DIGEST,
            resolved_config={{"epochs": 2}},
            effective_values={{"epochs": 2}},
            origins={{"epochs": "default"}},
        )

    def train(self, spec: TrainSpecV1, output_dir: Path, *, spec_path: Path | None = None) -> TrainReceiptV1:
        output_dir.mkdir(parents=True, exist_ok=True)
        if DEFECT == "ignore_sigterm":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        elif DEFECT == "omit_cancel_receipt":
            def _exit_no_receipt(_signum, _frame):
                raise SystemExit(130)
            signal.signal(signal.SIGTERM, _exit_no_receipt)

        emitter = EventEmitter(output_dir)
        emitter.phase("preparing")
        if DEFECT == "non_monotonic_seq":
            write_event_jsonl(output_dir / "events.jsonl", TrainEventV1(seq=1, kind="phase", phase="dup"))

        epochs = int(spec.parameters.get("epochs", 2))
        for ep in range(1, epochs + 1):
            time.sleep(0.05)
            emitter.epoch(epoch=ep, total_epochs=epochs, metrics={{"loss": 0.1}})

        dataset_dir = Path(spec.dataset_dir)
        manifest, _classes = validate_frozen_manifest_core(
            dataset_dir,
            expected_classes_digest=spec.classes_digest,
            expected_snapshot_digest=spec.dataset_snapshot_digest,
        )
        rows = validate_sample_files(dataset_dir, manifest)
        identity = load_dataset_identity(spec_path)
        membership = InputMembership(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            manifest=manifest,
            run_id=spec.run_id,
            attempt_id=spec.attempt_id,
            config_digest=spec.config_digest,
            dataset_snapshot_digest=spec.dataset_snapshot_digest,
            dataset_id=identity.get("dataset_id"),
            dataset_sha256=identity.get("dataset_sha256"),
        )
        membership.observe("train", [img for s, img, _ in rows if s["split"] == "train"])
        membership.observe("val", [img for s, img, _ in rows if s["split"] == "val"])
        mem_res = membership.complete()
        assert mem_res is not None
        if DEFECT == "wrong_policy_digest":
            bad_proof = dict(mem_res.proof)
            bad_proof["policy_digest"] = "0" * 64
            atomic_write_json(mem_res.path, bad_proof)

        weights = output_dir / "best.pt"
        weights.write_bytes(b"weights")
        artifacts = [
            build_artifact_entry("weights", weights),
            build_artifact_entry("input_membership", mem_res.path),
        ]
        if DEFECT == "supervisor_owned_artifact":
            owned = output_dir / "manifest.json"
            owned.write_text("{{}}", encoding="utf-8")
            artifacts.append(
                TrainArtifactV1(
                    role="config",
                    filename="manifest.json",
                    sha256=digest_file(owned),
                    size_bytes=owned.stat().st_size,
                )
            )
        receipt = TrainReceiptV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            status="succeeded",
            artifacts=artifacts,
        )
        atomic_write_json(output_dir / "result.json", receipt)
        return receipt

    def preview(self, spec: PreviewSpecV1, output_dir: Path, *, spec_path: Path | None = None) -> PreviewReportV1:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_img = output_dir / "preview-0.png"
        out_img.write_bytes(b"\\x89PNG\\r\\n\\x1a\\n")
        if DEFECT == "cuda_preview":
            raw = {{
                "protocol_version": TRAINER_PROTOCOL_VERSION,
                "status": "succeeded",
                "input_digest": spec.input_digest,
                "policy_digest": spec.policy_digest,
                "cuda_initialized": True,
                "outputs": [{{
                    "filename": out_img.name,
                    "sha256": digest_file(out_img),
                    "size_bytes": out_img.stat().st_size,
                    "width": 16,
                    "height": 16,
                    "box_count": 1,
                    "metadata": {{}},
                }}],
                "warnings": [],
                "runtime": {{}},
                "details": {{}},
                "failure_code": None,
                "failure_message": None,
            }}
            atomic_write_json(output_dir / "report.json", raw)
            return 0  # type: ignore[return-value]
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
                    width=16,
                    height=16,
                    box_count=1,
                )
            ],
        )


if __name__ == "__main__":
    raise SystemExit(
        run_trainer_cli(
            BuggyTrainer(),
            prog="buggy-trainer",
            description="Buggy test trainer",
            supports_preview=True,
        )
    )
''',
        encoding="utf-8",
    )
    return [sys.executable, str(script)]


def test_scaffold_example_passes_full_contract_check_and_own_tests(tmp_path: Path) -> None:
    project_dir = tmp_path / "demo-trainer"
    init_trainer_example(project_dir)
    assert (project_dir / "manifest.json").is_file()
    assert (project_dir / "pyproject.toml").is_file()
    assert (project_dir / "demo_trainer.py").is_file()
    assert (project_dir / "tests" / "test_demo_trainer.py").is_file()

    with pytest.raises(FileExistsError):
        init_trainer_example(project_dir)

    report = check_trainer_plugin(
        [sys.executable, str(project_dir / "demo_trainer.py")],
        manifest_path=project_dir / "manifest.json",
        check_preview=True,
        timeout_seconds=30.0,
    )
    assert isinstance(report, TrainerCheckReport)
    assert report.passed is True, report.model_dump()
    for name in (
        "manifest",
        "describe",
        "validate_config",
        "train",
        "input_membership",
        "cancel",
        "preview",
    ):
        assert report.checks[name].passed is True
        assert report.checks[name].skipped is False

    # --skip-train skips train/cancel/input_membership cleanly.
    quick = check_trainer_plugin(
        [sys.executable, str(project_dir / "demo_trainer.py")],
        manifest_path=project_dir / "manifest.json",
        skip_train=True,
        timeout_seconds=30.0,
    )
    assert quick.passed is True
    assert "train" not in quick.checks
    assert "cancel" not in quick.checks


def test_check_trainer_plugin_negative_controls(tmp_path: Path) -> None:
    # 1. Non-monotonic seq in events.jsonl
    r_seq = check_trainer_plugin(_write_mutated_trainer(tmp_path, "non_monotonic_seq"))
    assert r_seq.passed is False
    assert r_seq.checks["train"].passed is False
    assert "monotonic" in r_seq.checks["train"].detail

    # 2. Supervisor-owned artifact (manifest.json) declared in TrainReceiptV1
    r_owned = check_trainer_plugin(_write_mutated_trainer(tmp_path, "supervisor_owned_artifact"))
    assert r_owned.passed is False
    assert r_owned.checks["train"].passed is False
    assert "supervisor-owned" in r_owned.checks["train"].detail

    # 3. Wrong POLICY_DIGEST in input-membership.json
    r_policy = check_trainer_plugin(_write_mutated_trainer(tmp_path, "wrong_policy_digest"))
    assert r_policy.passed is False
    assert r_policy.checks["train"].passed is True
    assert r_policy.checks["input_membership"].passed is False
    assert "policy_digest" in r_policy.checks["input_membership"].detail

    # 4. Ignores SIGTERM
    r_ign = check_trainer_plugin(
        _write_mutated_trainer(tmp_path, "ignore_sigterm"), timeout_seconds=10.0
    )
    assert r_ign.passed is False
    assert r_ign.checks["cancel"].passed is False

    # 5. Exits 130 on SIGTERM without writing result.json
    r_omit = check_trainer_plugin(
        _write_mutated_trainer(tmp_path, "omit_cancel_receipt"), timeout_seconds=10.0
    )
    assert r_omit.passed is False
    assert r_omit.checks["cancel"].passed is False
    assert "result.json" in r_omit.checks["cancel"].detail

    # 6. Reports cuda_initialized=True in preview
    r_cuda = check_trainer_plugin(
        _write_mutated_trainer(tmp_path, "cuda_preview"),
        skip_train=True,
        check_preview=True,
    )
    assert r_cuda.passed is False
    assert r_cuda.checks["preview"].passed is False
    assert "cuda_initialized" in r_cuda.checks["preview"].detail

    # 7. Contract fallback succeeds when first listed contract is non-detection
    r_fb = check_trainer_plugin(
        _write_mutated_trainer(tmp_path, "contract_fallback"),
        skip_train=True,
    )
    assert r_fb.passed is True
    assert r_fb.checks["validate_config"].evidence["dataset_contract"] == "ultralytics-bbox-1"


def test_check_and_scaffold_stay_out_of_trainer_init_and_cli_version() -> None:
    probe = (
        "import json, sys, yoao_plugin_sdk.trainer; "
        "from yoao_plugin_sdk.cli import main; "
        "assert main(['version']) == 0; "
        "forbidden = {'yoao_plugin_sdk.trainer.check', 'yoao_plugin_sdk.trainer._scaffold', 'httpx', 'fastapi', 'torch', 'yaml'}; "
        "print(json.dumps(sorted(m for m in forbidden if m in sys.modules)))"
    )
    out = subprocess.check_output([sys.executable, "-c", probe], text=True)
    lines = out.strip().splitlines()
    assert json.loads(lines[-1]) == []
