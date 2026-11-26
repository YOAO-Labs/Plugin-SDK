from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from yoao_plugin_sdk.trainer._scaffold import init_trainer_example
from yoao_plugin_sdk.trainer.check import TrainerCheckReport, check_trainer_plugin


def _write_mutated_trainer(target: Path, defect: str, *, extra_env_file: Path | None = None) -> list[str]:
    script = target / f"buggy_{defect}.py"
    script.write_text(
        f'''from __future__ import annotations
import json
import os
from pathlib import Path
import signal
import subprocess
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
MARKER_FILE = {str(extra_env_file) if extra_env_file is not None else ""!r}
SCHEMA_ID = "buggy-detect-1"
SCHEMA_DIGEST = canonical_json_digest({{"schema": SCHEMA_ID}})


class BuggyTrainer:
    def describe(self) -> TrainerCapabilitiesV1:
        if DEFECT == "timeout_with_grandchild":
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            Path(MARKER_FILE).write_text(str(child.pid), encoding="utf-8")
            time.sleep(60)
        contracts = (
            ["custom-seg-only-1", "ultralytics-bbox-1"]
            if DEFECT == "contract_fallback"
            else ["ultralytics-bbox-1"]
        )
        policy_digests = (
            {{}}
            if DEFECT in {{"undeclared_im_file", "symlink_im_undeclared", "no_im_optional"}}
            else {{"input_membership": POLICY_DIGEST}}
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
            policy_digests=policy_digests,
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
        elif DEFECT == "cancel_timeout_grandchild":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            if int(spec.parameters.get("epochs", 2)) >= 10:
                child = subprocess.Popen(
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                Path(MARKER_FILE).write_text(str(child.pid), encoding="utf-8")
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
        elif DEFECT == "im_schema_bool":
            bad_proof = dict(mem_res.proof)
            bad_proof["schema"] = True
            atomic_write_json(mem_res.path, bad_proof)
        elif DEFECT == "im_extra_key":
            bad_proof = dict(mem_res.proof)
            bad_proof["extra"] = 1
            atomic_write_json(mem_res.path, bad_proof)
        elif DEFECT == "im_counts_bool":
            bad_proof = dict(mem_res.proof)
            bad_proof["counts"] = {{"train": 2, "val": True}}
            atomic_write_json(mem_res.path, bad_proof)
        elif DEFECT == "im_wrong_dataset_id":
            bad_proof = dict(mem_res.proof)
            bad_proof["dataset_id"] = "00000000-0000-0000-0000-000000000000"
            atomic_write_json(mem_res.path, bad_proof)

        weights = output_dir / "best.pt"
        weights.write_bytes(b"weights")
        artifacts = [build_artifact_entry("weights", weights)]
        if DEFECT == "omit_declared_im":
            mem_res.path.unlink()
        elif DEFECT == "undeclared_im_file":
            pass
        elif DEFECT == "symlink_im_undeclared":
            real_im = output_dir / "real-im.json"
            mem_res.path.rename(real_im)
            os.symlink(real_im, mem_res.path)
        elif DEFECT == "im_wrong_filename":
            renamed = output_dir / "custom-im.json"
            mem_res.path.rename(renamed)
            artifacts.append(build_artifact_entry("input_membership", renamed))
        elif DEFECT == "im_wrong_role":
            artifacts.append(build_artifact_entry("extra", mem_res.path))
        elif DEFECT == "no_im_optional":
            mem_res.path.unlink()
        else:
            artifacts.append(build_artifact_entry("input_membership", mem_res.path))

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
        elif DEFECT == "symlink_checkpoint_json":
            target_meta = output_dir / "real-ckpt.json"
            target_meta.write_text("{{}}", encoding="utf-8")
            os.symlink(target_meta, output_dir / "checkpoint.json")
        elif DEFECT == "ckpt_epoch_mismatch":
            ckpt_file = output_dir / "checkpoint-0002.pt"
            ckpt_file.write_bytes(b"ckpt")
            atomic_write_json(
                output_dir / "checkpoint.json",
                {{
                    "name": ckpt_file.name,
                    "role": "checkpoint",
                    "epoch": 1,
                    "sha256": digest_file(ckpt_file),
                    "size_bytes": ckpt_file.stat().st_size,
                }},
            )
        elif DEFECT == "ckpt_size_bool":
            ckpt_file = output_dir / "checkpoint-0001.pt"
            ckpt_file.write_bytes(b"x")
            atomic_write_json(
                output_dir / "checkpoint.json",
                {{
                    "name": ckpt_file.name,
                    "role": "checkpoint",
                    "epoch": 1,
                    "sha256": digest_file(ckpt_file),
                    "size_bytes": True,
                }},
            )
        elif DEFECT == "ckpt_size_zero":
            ckpt_file = output_dir / "checkpoint-0001.pt"
            ckpt_file.write_bytes(b"")
            atomic_write_json(
                output_dir / "checkpoint.json",
                {{
                    "name": ckpt_file.name,
                    "role": "checkpoint",
                    "epoch": 1,
                    "sha256": digest_file(ckpt_file),
                    "size_bytes": 0,
                }},
            )
        elif DEFECT == "readonly_output_subdir":
            ro_dir = output_dir / "ro-subdir"
            ro_dir.mkdir(parents=True, exist_ok=True)
            ro_file = ro_dir / "locked.txt"
            ro_file.write_text("locked", encoding="utf-8")
            os.chmod(ro_file, 0o444)
            os.chmod(ro_dir, 0o555)

        receipt = TrainReceiptV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            status="succeeded",
            artifacts=artifacts,
        )
        if DEFECT == "symlink_events_jsonl":
            real_events = output_dir / "real-events.jsonl"
            (output_dir / "events.jsonl").rename(real_events)
            os.symlink(real_events, output_dir / "events.jsonl")
        if DEFECT == "symlink_result_json":
            real_result = output_dir / "real-result.json"
            atomic_write_json(real_result, receipt)
            os.symlink(real_result, output_dir / "result.json")
            return 0  # type: ignore[return-value]
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
        report = PreviewReportV1(
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
        if DEFECT == "symlink_report_json":
            real_report = output_dir / "real-report.json"
            atomic_write_json(real_report, report)
            os.symlink(real_report, output_dir / "report.json")
            return 0  # type: ignore[return-value]
        return report


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

    demo_src = (project_dir / "demo_trainer.py").read_text(encoding="utf-8")
    assert "\nimport json\n" not in demo_src
    assert "atomic_write_bytes(weights_path," in demo_src
    assert "atomic_write_bytes(out_img," in demo_src

    with pytest.raises(FileExistsError):
        init_trainer_example(project_dir)

    # Actually run the generated plugin's own pytest suite.
    test_env = dict(os.environ)
    existing_pp = test_env.get("PYTHONPATH", "")
    test_env["PYTHONPATH"] = (
        f"{project_dir}{os.pathsep}{existing_pp}" if existing_pp else str(project_dir)
    )
    own_tests = subprocess.run(
        [sys.executable, "-m", "pytest", str(project_dir / "tests" / "test_demo_trainer.py"), "-q"],
        cwd=str(project_dir),
        env=test_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert own_tests.returncode == 0, own_tests.stdout + own_tests.stderr

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


def test_check_trainer_plugin_hardening_invariants(tmp_path: Path) -> None:
    # 1. Invalid timeout_seconds rejected up front.
    for bad_timeout in (float("nan"), float("inf"), float("-inf"), 0.0, -1.0, True):
        with pytest.raises(ValueError, match="timeout_seconds"):
            check_trainer_plugin(["echo"], timeout_seconds=bad_timeout)  # type: ignore[arg-type]

    # 2. Input membership strict declaration, pairing, schema, types, and identity checks.
    for defect, expected_substr in (
        ("omit_declared_im", "omitted input-membership.json"),
        ("undeclared_im_file", "exists on disk"),
        ("symlink_im_undeclared", "exists on disk"),
        ("im_wrong_filename", "input-membership.json"),
        ("im_wrong_role", "role='input_membership'"),
        ("im_schema_bool", "schema must be integer 1"),
        ("im_extra_key", "schema keys"),
        ("im_counts_bool", "counts"),
        ("im_wrong_dataset_id", "dataset_id/dataset_sha256"),
    ):
        rep = check_trainer_plugin(_write_mutated_trainer(tmp_path, defect))
        assert rep.passed is False, f"{defect} unexpectedly passed"
        assert rep.checks["train"].passed is True, f"{defect}: {rep.checks['train'].detail}"
        assert rep.checks["input_membership"].passed is False, defect
        assert expected_substr in rep.checks["input_membership"].detail, (
            defect,
            rep.checks["input_membership"].detail,
        )

    # Optional input_membership still skips cleanly when policy_digests and disk both omit it.
    rep_opt = check_trainer_plugin(_write_mutated_trainer(tmp_path, "no_im_optional"))
    assert rep_opt.passed is True, rep_opt.model_dump()
    assert rep_opt.checks["input_membership"].skipped is True

    # 3. Symlink and checkpoint hardening.
    for defect, expected_substr in (
        ("symlink_checkpoint_json", "symlink"),
        ("ckpt_epoch_mismatch", "does not match epoch"),
        ("ckpt_size_bool", "positive integer"),
        ("ckpt_size_zero", "positive integer"),
        ("symlink_events_jsonl", "symlink"),
        ("symlink_result_json", "symlink"),
    ):
        rep = check_trainer_plugin(_write_mutated_trainer(tmp_path, defect))
        assert rep.passed is False, f"{defect} unexpectedly passed"
        assert rep.checks["train"].passed is False, defect
        assert expected_substr in rep.checks["train"].detail, (
            defect,
            rep.checks["train"].detail,
        )

    rep_sym_preview = check_trainer_plugin(
        _write_mutated_trainer(tmp_path, "symlink_report_json"),
        skip_train=True,
        check_preview=True,
    )
    assert rep_sym_preview.passed is False
    assert rep_sym_preview.checks["preview"].passed is False
    assert "symlink" in rep_sym_preview.checks["preview"].detail

    # 4. Read-only subdirectory created by plugin inside output dir is cleaned up without error.
    rep_ro = check_trainer_plugin(_write_mutated_trainer(tmp_path, "readonly_output_subdir"))
    assert rep_ro.passed is True, rep_ro.model_dump()

    # 5. _verify_file_entry directly rejects bool expected_size even when file size is 1 byte.
    from yoao_plugin_sdk.trainer.check import _verify_file_entry
    from yoao_plugin_sdk.trainer.files import digest_file

    one_byte = tmp_path / "one.bin"
    one_byte.write_bytes(b"x")
    err = _verify_file_entry(tmp_path, one_byte.name, digest_file(one_byte), True, "artifact")
    assert err is not None and "integer" in err


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group kill verification")
def test_check_trainer_plugin_kills_grandchildren_on_timeout_and_cancel(tmp_path: Path) -> None:
    def _assert_pid_dead(pid: int) -> None:
        deadline = time.monotonic() + 2.0
        try:
            while time.monotonic() < deadline:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    return
                time.sleep(0.05)
            pytest.fail(f"grandchild PID {pid} survived process-group kill")
        finally:
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass

    # 1. Timeout in _run_subprocess kills grandchild.
    pid_file_timeout = tmp_path / "grandchild-timeout.pid"
    cmd_timeout = _write_mutated_trainer(
        tmp_path, "timeout_with_grandchild", extra_env_file=pid_file_timeout
    )
    rep_timeout = check_trainer_plugin(cmd_timeout, skip_train=True, timeout_seconds=0.8)
    assert rep_timeout.passed is False
    assert "timed out" in rep_timeout.checks["describe"].detail
    assert pid_file_timeout.is_file()
    _assert_pid_dead(int(pid_file_timeout.read_text(encoding="utf-8").strip()))

    # 2. Cancel grace timeout in _check_cancel kills grandchild.
    pid_file_cancel = tmp_path / "grandchild-cancel.pid"
    cmd_cancel = _write_mutated_trainer(
        tmp_path, "cancel_timeout_grandchild", extra_env_file=pid_file_cancel
    )
    rep_cancel = check_trainer_plugin(cmd_cancel, timeout_seconds=1.5)
    assert rep_cancel.passed is False
    assert rep_cancel.checks["cancel"].passed is False
    assert pid_file_cancel.is_file()
    _assert_pid_dead(int(pid_file_cancel.read_text(encoding="utf-8").strip()))
