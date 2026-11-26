from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from yoao_plugin_sdk import (
    EXTENSION_MANIFEST_SCHEMA,
    EXTENSION_POINTS,
    ExtensionManifestV1,
    load_manifest_file,
    read_secret,
    verify_bearer_token,
)
from yoao_plugin_sdk.cli import main as cli_main
from yoao_plugin_sdk.format_adapter import ImportParseResponse, ParsedAnnotation, ParsedItem
from yoao_plugin_sdk.inference import INFERENCE_PROTOCOL_VERSION, SUPPORTED_CAPABILITIES
from yoao_plugin_sdk.notification import DeliverRequest, DeliverResult, DeliveryEnvelope
from yoao_plugin_sdk.trainer import (
    PREVIEW_MAX_POOL_BYTES,
    PREVIEW_MAX_SOURCE_BYTES,
    TRAINER_PROTOCOL_VERSION,
    PreviewOutputV1,
    PreviewReportV1,
    PreviewSourceV1,
    PreviewSpecV1,
    ServingCompatibilityV1,
    TrainArtifactV1,
    TrainEventV1,
    TrainReceiptV1,
    TrainSpecV1,
    TrainerArchitectureV1,
    TrainerCapabilitiesV1,
    ValidateConfigResultV1,
    ValidationIssueV1,
    atomic_write_json,
    canonical_json_bytes,
    canonical_json_digest,
    write_event_jsonl,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
DIGEST_D = "d" * 64
DIGEST_E = "e" * 64


def _collect_schema_keys(node: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            found.add(key)
            found.update(_collect_schema_keys(value))
    elif isinstance(node, list):
        for item in node:
            found.update(_collect_schema_keys(item))
    return found


@pytest.mark.parametrize("extension_point", EXTENSION_POINTS)
def test_cli_init_and_validate_all_extension_points(
    tmp_path: Path, extension_point: str
) -> None:
    target = tmp_path / f"demo-{extension_point.replace('_', '-')}"
    assert cli_main(["init", "--type", extension_point, str(target)]) == 0
    manifest_file = target / "manifest.json"
    assert manifest_file.exists()
    assert cli_main(["validate", str(manifest_file)]) == 0
    manifest = load_manifest_file(manifest_file)
    assert manifest.manifest_schema == EXTENSION_MANIFEST_SCHEMA
    assert manifest.extension_point == extension_point


def test_cli_version_and_inference_example_scaffold(tmp_path: Path) -> None:
    assert cli_main(["version"]) == 0
    target = tmp_path / "demo-inference-example"
    assert cli_main(["init", "--type", "inference", "--example", str(target)]) == 0
    assert (target / "manifest.json").is_file()
    assert (target / "pyproject.toml").is_file()
    assert (target / "my_color_region_plugin.py").is_file()
    assert "yoao-plugin-sdk[inference]" in (target / "pyproject.toml").read_text(encoding="utf-8")
    manifest = load_manifest_file(target / "manifest.json")
    assert manifest.entrypoint == [
        "my-color-region-plugin",
        "serve",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    assert cli_main(["init", "--type", "inference", "--example", str(target)]) == 1


def test_cli_trainer_example_scaffold_and_trainer_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys

    target = tmp_path / "my-cpu-trainer"
    assert cli_main(["init", "--type", "trainer", "--example", str(target)]) == 0
    assert (target / "manifest.json").is_file()
    assert (target / "pyproject.toml").is_file()
    assert (target / "demo_trainer.py").is_file()
    assert (target / "tests" / "test_demo_trainer.py").is_file()
    assert cli_main(["validate", str(target / "manifest.json")]) == 0
    assert cli_main(["init", "--type", "trainer", "--example", str(target)]) == 1

    capsys.readouterr()
    report_path = tmp_path / "trainer-check-report.json"
    rc = cli_main(
        [
            "trainer",
            "check",
            "--manifest",
            str(target / "manifest.json"),
            "--entrypoint",
            sys.executable,
            "--arg",
            str(target / "demo_trainer.py"),
            "--preview",
            "--report",
            str(report_path),
        ]
    )
    assert rc == 0
    stdout_report = json.loads(capsys.readouterr().out)
    file_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert stdout_report == file_report
    assert file_report["passed"] is True
    for check_name in (
        "manifest",
        "describe",
        "validate_config",
        "train",
        "input_membership",
        "cancel",
        "preview",
    ):
        assert file_report["checks"][check_name]["passed"] is True


def test_manifest_rejects_shell_entrypoint_and_self_reported_source_tier() -> None:
    with pytest.raises(ValidationError):
        ExtensionManifestV1.model_validate(
            {
                "manifest_schema": EXTENSION_MANIFEST_SCHEMA,
                "extension_point": "format_adapter",
                "plugin_id": "bad.shell",
                "version": "0.1.0",
                "publisher": "Test",
                "license": "MIT",
                "protocol_version": "yoao-format-adapter/v1",
                "entrypoint": ["bash", "-c", "id"],
                "capabilities": ["import"],
                "resource": {"memory_mb": 128},
                "extension": {},
            }
        )
    with pytest.raises(ValidationError):
        ExtensionManifestV1.model_validate(
            {
                "manifest_schema": EXTENSION_MANIFEST_SCHEMA,
                "extension_point": "theme",
                "plugin_id": "spoof.tier",
                "version": "0.1.0",
                "publisher": "Test",
                "license": "MIT",
                "protocol_version": "yoao-theme/v1",
                "capabilities": ["dark"],
                "extension": {"source_tier": "official"},
            }
        )


def test_auth_and_protocol_models(tmp_path: Path) -> None:
    assert INFERENCE_PROTOCOL_VERSION == "yoao-inference/v2"
    assert "inference" in SUPPORTED_CAPABILITIES
    assert "detection" in SUPPORTED_CAPABILITIES
    assert "managed_evaluation" in SUPPORTED_CAPABILITIES
    assert "train" not in SUPPORTED_CAPABILITIES

    secret_file = tmp_path / "token.txt"
    secret_file.write_text("  super-secret  \n", encoding="utf-8")
    assert read_secret({"T_FILE": str(secret_file)}, "T", "T_FILE") == "super-secret"
    assert verify_bearer_token("Bearer super-secret", "super-secret") is True
    assert verify_bearer_token("Bearer wrong", "super-secret") is False

    req = DeliverRequest(
        channel_id="feishu_dm",
        envelope=DeliveryEnvelope(
            event="task.rejected",
            delivery_id="11111111-1111-4111-8111-111111111111",
            occurred_at="2026-11-25T00:00:00Z",
            data={"task_display_id": "T-1"},
        ),
        recipient_address={"open_id": "ou_123"},
    )
    assert req.channel_id == "feishu_dm"
    assert DeliverResult(status="delivered", provider_message_id="om_1").status == "delivered"

    parsed = ImportParseResponse(
        items=[
            ParsedItem(
                media_path="a.jpg",
                width=100,
                height=100,
                annotations=[
                    ParsedAnnotation(
                        class_name="car",
                        tool_unit_id="bbox",
                        geometry={"type": "bbox", "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
                    )
                ],
            )
        ]
    )
    assert len(json.loads(parsed.model_dump_json())["items"]) == 1


def test_inference_extension_manifest_container_and_direct_modes(tmp_path: Path) -> None:
    base_extension = {
        "models": [
            {
                "id": "seg",
                "task": "segment",
                "is_interactive": False,
                "supported_prompts": ["none"],
                "supported_inputs": ["full_image", "crop"],
                "supported_geometric_outputs": ["polygon", "mask_rle"],
            }
        ],
        "pools": {"seg": {"cap": 2}},
        "compute": {"device": "cuda", "gpu_memory_mb": 2048},
        "asset_manifest_sha256": DIGEST_A,
        "allowed_asset_sources": ["https://example.com/weights"],
        "managed_model_compatibility": [
            {"framework": "ultralytics", "format": None, "runtime_device": "cuda"}
        ],
    }
    container_doc = {
        "manifest_schema": EXTENSION_MANIFEST_SCHEMA,
        "extension_point": "inference",
        "plugin_id": "yoao.inference.yolo",
        "version": "1.0.0",
        "display_name": "YOLO Inference",
        "description": "Multi-task detection and segmentation plugin.",
        "homepage": "https://example.com/yolo",
        "publisher": "YOAO Official",
        "license": "AGPL-3.0",
        "protocol_version": INFERENCE_PROTOCOL_VERSION,
        "entrypoint": ["yoao-yolo", "serve", "--port", "8000"],
        "capabilities": [
            "inference",
            "detection",
            "segmentation",
            "managed_models",
            "managed_evaluation",
        ],
        "resource": {"memory_mb": 2048},
        "extension": base_extension,
    }
    parsed_container = ExtensionManifestV1.model_validate(container_doc)
    bound = parsed_container.bind_artifact(f"ghcr.io/yoao/yolo@sha256:{DIGEST_B}")
    assert bound.artifact == f"ghcr.io/yoao/yolo@sha256:{DIGEST_B}"

    direct_doc = {
        "manifest_schema": EXTENSION_MANIFEST_SCHEMA,
        "extension_point": "inference",
        "plugin_id": "custom.inference.ocr",
        "version": "0.2.0",
        "publisher": "Team",
        "license": "MIT",
        "protocol_version": INFERENCE_PROTOCOL_VERSION,
        "capabilities": ["inference", "ocr", "doc_layout"],
        "extension": {
            "execution_mode": "direct",
            "models": [
                {
                    "id": "ocr",
                    "task": "ocr",
                    "is_interactive": False,
                    "supported_prompts": ["none"],
                    "supported_inputs": ["full_image"],
                    "supported_geometric_outputs": ["polygon"],
                }
            ],
            "pools": {"ocr": {"cap": 1}},
            "compute": {"device": "cuda", "gpu_memory_mb": 0},
        },
    }
    parsed_direct = ExtensionManifestV1.model_validate(direct_doc)
    assert parsed_direct.entrypoint is None
    assert parsed_direct.resource is None
    with pytest.raises(ValueError, match="direct inference manifest"):
        parsed_direct.bind_artifact(f"ghcr.io/yoao/ocr@sha256:{DIGEST_B}")

    invalid_cases = [
        {**container_doc, "capabilities": ["inference", "train"]},
        {**container_doc, "capabilities": ["detection"]},
        {**container_doc, "capabilities": ["inference", "managed_evaluation"]},
        {
            **container_doc,
            "capabilities": ["inference", "detection"],
            "extension": {**base_extension, "managed_model_compatibility": [{"framework": "x", "format": None, "runtime_device": "cuda"}]},
        },
        {
            **container_doc,
            "extension": {**base_extension, "asset_manifest_sha256": None},
        },
        {
            **container_doc,
            "extension": {**base_extension, "compute": {"device": "cuda", "gpu_memory_mb": 0}},
        },
        {**direct_doc, "artifact": f"ghcr.io/yoao/ocr@sha256:{DIGEST_B}"},
        {**direct_doc, "extension": {**direct_doc["extension"], "source_tier": "official"}},
    ]
    for bad in invalid_cases:
        with pytest.raises(ValidationError):
            ExtensionManifestV1.model_validate(bad)

    for extra_bad in (
        {k: v for k, v in container_doc.items() if k != "manifest_schema"},
        {**container_doc, "homepage": "https://example.com:99999/x"},
        {**container_doc, "homepage": "https://example.com:/x"},
        {**container_doc, "homepage": "https://2130706433/x"},
        {**container_doc, "entrypoint": ["yoao-yolo", "   "]},
        {**container_doc, "version": "   "},
        {**container_doc, "license": "MIT\x00"},
    ):
        with pytest.raises(ValidationError):
            ExtensionManifestV1.model_validate(extra_bad)

    dup_file = tmp_path / "dup-manifest.json"
    dup_file.write_text(
        '{"manifest_schema":"yoao-extension-manifest/v1","manifest_schema":"x"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_manifest_file(dup_file)


def test_trainer_protocol_models_and_atomic_writers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compat = ServingCompatibilityV1(
        framework="example-trainer",
        format=None,
        training_runtime={"python": "3.12", "framework": "1.0.0"},
        compatible_serving_runtimes=[{"device": "cpu", "framework": "1.0.0"}],
    )
    caps = TrainerCapabilitiesV1(
        protocol_version=TRAINER_PROTOCOL_VERSION,
        framework_id="example-trainer",
        framework_version="1.0.0",
        runtime_versions={"python": "3.12"},
        supported_tasks=["detect"],
        supported_dataset_contracts=["bbox-v1"],
        architectures=[
            TrainerArchitectureV1(
                id="demo-n",
                family="demo",
                tasks=["detect"],
                base_asset_filename="demo-n.pt",
                base_asset_sha256=DIGEST_A,
            )
        ],
        policy_digests={"budget": DIGEST_B},
        serving_compatibility=[compat],
    )
    assert caps.protocol_version == TRAINER_PROTOCOL_VERSION
    assert len(canonical_json_digest(caps)) == 64

    valid_cfg = ValidateConfigResultV1(
        protocol_version=TRAINER_PROTOCOL_VERSION,
        valid=True,
        schema_id="example-1.0.0-detect-1",
        schema_digest=DIGEST_A,
        resolved_config={"epochs": 10},
        effective_values={"epochs": 10},
        origins={"epochs": "user"},
    )
    assert valid_cfg.valid is True
    invalid_cfg = ValidateConfigResultV1(
        protocol_version=TRAINER_PROTOCOL_VERSION,
        valid=False,
        issues=[ValidationIssueV1(code="range_error", field="epochs", message="must be >= 1")],
    )
    assert len(invalid_cfg.issues) == 1

    spec = TrainSpecV1(
        protocol_version=TRAINER_PROTOCOL_VERSION,
        run_id="run-1",
        attempt_id="attempt-1",
        framework_id="example-trainer",
        task="detect",
        architecture_id="demo-n",
        dataset_contract="bbox-v1",
        dataset_dir="/work/input/dataset",
        dataset_snapshot_digest=DIGEST_A,
        config_digest=DIGEST_B,
        classes_digest=DIGEST_C,
        schema_id="example-1.0.0-detect-1",
        schema_digest=DIGEST_D,
        parameters={"epochs": 10},
    )
    spec_path = tmp_path / "nested" / "spec.json"
    atomic_write_json(spec_path, spec)
    assert stat.S_IMODE(spec_path.stat().st_mode) == 0o644
    raw_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert raw_spec["attempt_id"] == "attempt-1"
    assert TrainSpecV1.model_validate(raw_spec) == spec

    events_path = tmp_path / "events.jsonl"
    write_event_jsonl(
        events_path,
        TrainEventV1(
            seq=1,
            kind="epoch",
            epoch=1,
            total_epochs=10,
            progress=1,
            metrics={"map50": 0.72},
        ),
    )
    write_event_jsonl(
        events_path,
        TrainEventV1(seq=2, kind="progress", progress=-0.0, metrics={"loss": -0.0}),
    )
    lines = events_path.read_text(encoding="utf-8").splitlines()
    assert "-0.0" not in lines[1]
    rows = [json.loads(line) for line in lines]
    assert rows[0]["progress"] == 1.0
    assert rows[1]["progress"] == 0.0
    assert rows[1]["metrics"]["loss"] == 0.0

    receipt = TrainReceiptV1(
        protocol_version=TRAINER_PROTOCOL_VERSION,
        status="succeeded",
        artifacts=[
            TrainArtifactV1(
                role="weights",
                filename="best.pt",
                sha256=DIGEST_A,
                size_bytes=1024,
            )
        ],
        serving_compatibility=compat,
    )
    receipt_path = tmp_path / "result.json"
    atomic_write_json(receipt_path, receipt, mode=0o600)
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
    raw_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert raw_receipt["serving_compatibility"]["format"] is None
    reloaded = TrainReceiptV1.model_validate(raw_receipt)
    assert reloaded == receipt
    assert canonical_json_digest(reloaded) == canonical_json_digest(raw_receipt)
    assert b"-0.0" not in canonical_json_bytes({"z": -0.0})

    failed_receipt = TrainReceiptV1(
        protocol_version=TRAINER_PROTOCOL_VERSION,
        status="failed",
        failure_code="oom",
        failure_message="CUDA out of memory",
    )
    assert failed_receipt.failure_code == "oom"

    preview_spec = PreviewSpecV1(
        protocol_version=TRAINER_PROTOCOL_VERSION,
        preview_id="preview-1",
        framework_id="example-trainer",
        task="detect",
        architecture_id="demo-n",
        input_digest=DIGEST_E,
        config_digest=DIGEST_A,
        dataset_snapshot_digest=DIGEST_B,
        policy_digest=DIGEST_C,
        seed=42,
        epoch=1,
        count=2,
        sources=[
            PreviewSourceV1(
                task_id="task-1",
                filename="images/a.jpg",
                sha256=DIGEST_D,
                width=640,
                height=480,
                size_bytes=12345,
                box_count=3,
                label_path="labels/a.txt",
                label_sha256=DIGEST_A,
            )
        ],
        parameters={"mosaic": 0.5},
        native_values={"device": "cpu", "workers": 0},
        pool_summary={"candidate_count": 1},
    )
    preview_report = PreviewReportV1(
        protocol_version=TRAINER_PROTOCOL_VERSION,
        status="succeeded",
        input_digest=preview_spec.input_digest,
        policy_digest=DIGEST_C,
        cuda_initialized=False,
        outputs=[
            PreviewOutputV1(
                filename="preview-0.png",
                sha256=DIGEST_A,
                size_bytes=4096,
                width=640,
                height=640,
                box_count=3,
                metadata={"clipped": 0, "filtered": 0},
            )
        ],
        runtime={"cpu_only": True},
        details={"mosaic_closed": False},
    )
    assert preview_report.cuda_initialized is False

    failed_preview = PreviewReportV1(
        protocol_version=TRAINER_PROTOCOL_VERSION,
        status="failed",
        input_digest=preview_spec.input_digest,
        policy_digest=DIGEST_C,
        failure_code="preview_timeout",
        failure_message="exceeded 45s",
    )
    assert failed_preview.status == "failed"

    def _boom(_src: str, _dst: Path) -> None:
        raise RuntimeError("simulated replace error")

    monkeypatch.setattr(os, "replace", _boom)
    fail_target = tmp_path / "cleanup" / "atomic.json"
    with pytest.raises(RuntimeError, match="simulated replace error"):
        atomic_write_json(fail_target, {"ok": True})
    assert not fail_target.exists()
    assert not list(fail_target.parent.glob(".tmp-*"))


def test_trainer_models_emit_valid_json_schema_with_required_protocol_version() -> None:
    for model_cls in (
        TrainerCapabilitiesV1,
        ValidateConfigResultV1,
        TrainSpecV1,
        TrainReceiptV1,
        PreviewSpecV1,
        PreviewReportV1,
    ):
        schema = model_cls.model_json_schema()
        assert "protocol_version" in schema.get("required", [])
        keys = _collect_schema_keys(schema)
        assert "ge" not in keys
        assert "le" not in keys

    event_schema = TrainEventV1.model_json_schema()
    event_keys = _collect_schema_keys(event_schema)
    assert "ge" not in event_keys
    assert "le" not in event_keys
    progress_any_of = event_schema["properties"]["progress"]["anyOf"]
    number_branch = next(item for item in progress_any_of if item.get("type") == "number")
    assert number_branch["minimum"] == 0.0
    assert number_branch["maximum"] == 1.0


def test_trainer_protocol_negative_invariants(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        ServingCompatibilityV1(framework="demo", format="")
    with pytest.raises(ValidationError):
        ServingCompatibilityV1(framework="demo", format="   ")
    with pytest.raises(ValidationError):
        ServingCompatibilityV1.model_validate({"framework": "demo"})
    with pytest.raises(ValidationError):
        ServingCompatibilityV1.model_validate({"framework": "demo", "format": None, "extra": 1})

    with pytest.raises(ValidationError):
        TrainReceiptV1.model_validate({"status": "cancelled"})
    with pytest.raises(ValidationError):
        TrainerArchitectureV1(
            id="demo-n",
            family="demo",
            tasks=["detect"],
            base_asset_sha256=DIGEST_A,
        )
    with pytest.raises(ValidationError):
        TrainerCapabilitiesV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            framework_id="demo",
            framework_version="1.0.0",
            supported_tasks=["detect"],
            supported_dataset_contracts=["bbox-v1"],
            architectures=[
                TrainerArchitectureV1(id="demo-n", family="demo", tasks=["segment"])
            ],
        )
    with pytest.raises(ValidationError):
        TrainerCapabilitiesV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            framework_id="demo",
            framework_version="1.0.0",
            runtime_versions={" bad": "1.0"},
            supported_tasks=["detect"],
            supported_dataset_contracts=["bbox-v1"],
            architectures=[
                TrainerArchitectureV1(id="demo-n", family="demo", tasks=["detect"])
            ],
        )
    with pytest.raises(ValidationError):
        TrainerCapabilitiesV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            framework_id="demo",
            framework_version="1.0.0",
            supported_tasks=["detect"],
            supported_dataset_contracts=["bbox-v1"],
            architectures=[
                TrainerArchitectureV1(id="demo-n", family="demo", tasks=["detect"])
            ],
            policy_digests={"budget": "not-a-sha"},
        )

    with pytest.raises(ValidationError):
        ValidateConfigResultV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            valid=True,
            schema_id="s-1",
        )
    with pytest.raises(ValidationError):
        ValidateConfigResultV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            valid=True,
            schema_id="s-1",
            schema_digest=DIGEST_A,
            issues=[ValidationIssueV1(code="err", message="bad")],
        )
    with pytest.raises(ValidationError):
        ValidateConfigResultV1(protocol_version=TRAINER_PROTOCOL_VERSION, valid=False)

    with pytest.raises(ValidationError):
        TrainSpecV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            run_id="run-1",
            attempt_id="attempt-1",
            framework_id="demo",
            task="detect",
            architecture_id="demo-n",
            dataset_contract="bbox-v1",
            dataset_dir="/work/input/dataset",
            dataset_snapshot_digest=DIGEST_A,
            config_digest=DIGEST_B,
            classes_digest=DIGEST_C,
            resume_checkpoint_path="/work/input/resume.pt",
            init_weights_path="/work/input/init.pt",
        )
    with pytest.raises(ValidationError):
        TrainSpecV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            run_id="run-1",
            attempt_id="attempt-1",
            framework_id="demo",
            task="detect",
            architecture_id="demo-n",
            dataset_contract="bbox-v1",
            dataset_dir="/work/input/dataset",
            dataset_snapshot_digest=DIGEST_A,
            config_digest=DIGEST_B,
            classes_digest=DIGEST_C,
            schema_id="only-id",
        )

    with pytest.raises(ValidationError):
        TrainEventV1(seq=1, kind="epoch", epoch=5, total_epochs=3)
    with pytest.raises(ValidationError):
        TrainEventV1(seq=1, kind="progress", progress=float("nan"))
    with pytest.raises(ValidationError):
        TrainEventV1.model_validate({"seq": 1, "kind": "progress", "progress": True})
    with pytest.raises(ValueError):
        write_event_jsonl(
            tmp_path / "never-created" / "events.jsonl",
            TrainEventV1(seq=1, kind="epoch", metrics={"loss": float("nan")}),
        )
    assert not (tmp_path / "never-created").exists()

    with pytest.raises(ValidationError):
        TrainArtifactV1(
            role="weights",
            filename="../escape.pt",
            sha256=DIGEST_A,
            size_bytes=10,
        )
    with pytest.raises(ValidationError):
        TrainReceiptV1(protocol_version=TRAINER_PROTOCOL_VERSION, status="failed")
    with pytest.raises(ValidationError):
        TrainReceiptV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            status="succeeded",
            failure_code="unexpected",
        )

    with pytest.raises(ValidationError):
        PreviewSourceV1(
            task_id="t-1",
            filename="../secret.jpg",
            sha256=DIGEST_A,
            width=100,
            height=100,
            size_bytes=100,
            box_count=0,
        )
    with pytest.raises(ValidationError):
        PreviewSourceV1(
            task_id="t-1",
            filename="images/huge.jpg",
            sha256=DIGEST_A,
            width=5000,
            height=4000,
            size_bytes=100,
            box_count=0,
        )
    with pytest.raises(ValidationError):
        PreviewSourceV1(
            task_id="t-1",
            filename="images/a.jpg",
            sha256=DIGEST_A,
            width=100,
            height=100,
            size_bytes=100,
            box_count=0,
            label_path="labels/a.txt",
        )
    with pytest.raises(ValidationError):
        PreviewSpecV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            preview_id="p-oversize",
            framework_id="demo",
            task="detect",
            architecture_id="demo-n",
            input_digest=DIGEST_E,
            config_digest=DIGEST_A,
            dataset_snapshot_digest=DIGEST_B,
            policy_digest=DIGEST_C,
            seed=1,
            epoch=1,
            count=1,
            sources=[
                PreviewSourceV1(
                    task_id=f"t-{idx}",
                    filename=f"images/{idx}.jpg",
                    sha256=DIGEST_D,
                    width=100,
                    height=100,
                    size_bytes=PREVIEW_MAX_SOURCE_BYTES,
                    box_count=0,
                )
                for idx in range((PREVIEW_MAX_POOL_BYTES // PREVIEW_MAX_SOURCE_BYTES) + 1)
            ],
        )
    with pytest.raises(ValidationError):
        PreviewReportV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            status="succeeded",
            input_digest=DIGEST_A,
            policy_digest=DIGEST_B,
            cuda_initialized=True,
            outputs=[
                PreviewOutputV1(
                    filename="p0.png",
                    sha256=DIGEST_C,
                    size_bytes=100,
                    width=64,
                    height=64,
                    box_count=1,
                )
            ],
        )
    with pytest.raises(ValidationError):
        PreviewReportV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            status="succeeded",
            input_digest=DIGEST_A,
            policy_digest=DIGEST_B,
            outputs=[],
        )
    with pytest.raises(ValidationError):
        PreviewReportV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            status="failed",
            input_digest=DIGEST_A,
            policy_digest=DIGEST_B,
        )

    bad_dir = tmp_path / "never-mkdir"
    bad_path = bad_dir / "nan.json"
    with pytest.raises(ValueError):
        atomic_write_json(bad_path, {"loss": float("nan")})
    assert not bad_dir.exists()


def test_trainer_package_preserves_exports_and_has_zero_ml_imports() -> None:
    import subprocess
    import sys

    import yoao_plugin_sdk.trainer as trainer_pkg

    legacy_exports = {
        "PREVIEW_MAX_EDGE",
        "PREVIEW_MAX_OUTPUTS",
        "PREVIEW_MAX_OUTPUT_BOXES",
        "PREVIEW_MAX_OUTPUT_BYTES",
        "PREVIEW_MAX_POOL_BOXES",
        "PREVIEW_MAX_POOL_BYTES",
        "PREVIEW_MAX_POOL_PIXELS",
        "PREVIEW_MAX_SOURCES",
        "PREVIEW_MAX_SOURCE_BOXES",
        "PREVIEW_MAX_SOURCE_BYTES",
        "PREVIEW_MAX_SOURCE_PIXELS",
        "PREVIEW_SEED_MAX",
        "SHA256_PATTERN",
        "TRAINER_PROTOCOL_VERSION",
        "PreviewOutputV1",
        "PreviewReportV1",
        "PreviewSourceV1",
        "PreviewSpecV1",
        "ServingCompatibilityV1",
        "TrainArtifactV1",
        "TrainEventV1",
        "TrainReceiptV1",
        "TrainSpecV1",
        "TrainerArchitectureV1",
        "TrainerCapabilitiesV1",
        "ValidateConfigResultV1",
        "ValidationIssueV1",
        "atomic_write_json",
        "canonical_json_bytes",
        "canonical_json_digest",
        "write_event_jsonl",
    }
    assert legacy_exports.issubset(set(trainer_pkg.__all__))

    probe = (
        "import json, sys, yoao_plugin_sdk.trainer; "
        "forbidden = {'torch', 'torchvision', 'ultralytics', 'libreyolo', 'cv2', 'numpy', 'yaml', 'httpx', 'fastapi'}; "
        "loaded = sorted(m for m in forbidden if m in sys.modules); "
        "print(json.dumps(loaded))"
    )
    out = subprocess.check_output([sys.executable, "-c", probe], text=True)
    assert json.loads(out) == []
