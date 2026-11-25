from __future__ import annotations

import json
from pathlib import Path

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
    TRAINER_PROTOCOL_VERSION,
    ServingCompatibilityV1,
    TrainArtifactV1,
    TrainEventV1,
    TrainReceiptV1,
    TrainSpecV1,
    TrainerArchitectureV1,
    TrainerCapabilitiesV1,
    atomic_write_json,
    write_event_jsonl,
)


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
    assert "predict" in SUPPORTED_CAPABILITIES

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


def test_trainer_protocol_models_and_atomic_writers(tmp_path: Path) -> None:
    compat = ServingCompatibilityV1(
        framework="ultralytics",
        format=None,
        training_runtime={"python": "3.12", "ultralytics": "8.4.183"},
        compatible_serving_runtimes=[{"device": "cpu", "ultralytics": "8.4.183"}],
    )
    caps = TrainerCapabilitiesV1(
        framework_id="ultralytics",
        framework_version="8.4.183",
        runtime_versions={"python": "3.12"},
        supported_tasks=["detect"],
        supported_dataset_contracts=["ultralytics-bbox-1"],
        architectures=[
            TrainerArchitectureV1(
                id="yolo11n",
                family="yolo11",
                tasks=["detect"],
                base_asset_filename="yolo11n.pt",
            )
        ],
        serving_compatibility=[compat],
    )
    assert caps.protocol_version == TRAINER_PROTOCOL_VERSION

    spec = TrainSpecV1(
        run_id="run-1",
        attempt_id="attempt-1",
        framework_id="ultralytics",
        task="detect",
        architecture_id="yolo11n",
        dataset_contract="ultralytics-bbox-1",
        dataset_dir="/work/input/dataset",
        parameters={"epochs": 10},
    )
    spec_path = tmp_path / "spec.json"
    atomic_write_json(spec_path, spec)
    assert json.loads(spec_path.read_text(encoding="utf-8"))["attempt_id"] == "attempt-1"

    events_path = tmp_path / "events.jsonl"
    write_event_jsonl(
        events_path,
        TrainEventV1(seq=1, kind="epoch", epoch=1, total_epochs=10, progress=0.1, metrics={"map50": 0.72}),
    )
    rows = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "detail": {},
            "epoch": 1,
            "kind": "epoch",
            "metrics": {"map50": 0.72},
            "progress": 0.1,
            "seq": 1,
            "total_epochs": 10,
        }
    ]

    receipt = TrainReceiptV1(
        status="succeeded",
        artifacts=[
            TrainArtifactV1(
                role="weights",
                filename="best.pt",
                sha256="a" * 64,
                size_bytes=1024,
            )
        ],
        serving_compatibility=compat,
    )
    receipt_path = tmp_path / "result.json"
    atomic_write_json(receipt_path, receipt)
    raw_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert raw_receipt["serving_compatibility"]["format"] is None
    reloaded = TrainReceiptV1.model_validate(raw_receipt)
    assert reloaded.status == "succeeded"
    assert reloaded.serving_compatibility is not None
    assert reloaded.serving_compatibility.format is None

    with pytest.raises(ValidationError):
        ServingCompatibilityV1(framework="ultralytics", format="")
    with pytest.raises(ValidationError):
        ServingCompatibilityV1(framework="ultralytics", format="   ")
    with pytest.raises(ValidationError):
        ServingCompatibilityV1.model_validate({"framework": "ultralytics"})
    with pytest.raises(ValidationError):
        TrainSpecV1(
            run_id="run-1",
            attempt_id="attempt-1",
            framework_id="ultralytics",
            task="detect",
            architecture_id="yolo11n",
            dataset_contract="ultralytics-bbox-1",
            dataset_dir="/work/input/dataset",
            resume_checkpoint_path="/work/input/resume.pt",
            init_weights_path="/work/input/init.pt",
        )
    bad_path = tmp_path / "nan.json"
    with pytest.raises(ValueError):
        atomic_write_json(bad_path, {"loss": float("nan")})
    assert not bad_path.exists()
    assert not list(tmp_path.glob(".tmp-*"))
