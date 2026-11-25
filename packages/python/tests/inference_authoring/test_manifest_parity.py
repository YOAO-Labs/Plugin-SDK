"""Cross-repository schema & round-trip parity tests between Plugin-SDK and yoao-sdk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from yoao_inference_contract import plugin as platform_plugin
from yoao_plugin_sdk import manifest as sdk_manifest
from yoao_plugin_sdk.inference.authoring.manifest import load_manifest
from yoao_task_contract import Task

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def _deref_schema(schema: Any, defs: dict[str, Any] | None = None) -> Any:
    if defs is None:
        defs = schema.get("$defs", {}) if isinstance(schema, dict) else {}
    if isinstance(schema, dict):
        if "$ref" in schema:
            name = schema["$ref"].split("/")[-1]
            return _deref_schema(defs[name], defs)
        return {
            key: _deref_schema(value, defs)
            for key, value in schema.items()
            if key not in {"title", "description", "$defs"}
        }
    if isinstance(schema, list):
        return [_deref_schema(item, defs) for item in schema]
    return schema


def test_inference_manifest_constants_and_json_schemas_match_platform_contract() -> None:
    assert sdk_manifest.EXTENSION_MANIFEST_SCHEMA == platform_plugin.EXTENSION_MANIFEST_SCHEMA
    assert sdk_manifest.INFERENCE_PROTOCOL_VERSION == platform_plugin.INFERENCE_PROTOCOL_VERSION
    assert (
        sdk_manifest.INFERENCE_WIRE_PROTOCOL_VERSION
        == platform_plugin.INFERENCE_WIRE_PROTOCOL_VERSION
    )
    assert sdk_manifest.INFERENCE_EXECUTION_MODES == platform_plugin.INFERENCE_EXECUTION_MODES
    assert sdk_manifest.INFERENCE_CAPABILITIES == platform_plugin.INFERENCE_CAPABILITIES
    assert sdk_manifest.INFERENCE_TASK_VALUES == platform_plugin.INFERENCE_TASK_VALUES
    assert sdk_manifest.INFERENCE_TASK_VALUES == get_args(Task)
    assert "train" not in sdk_manifest.INFERENCE_CAPABILITIES

    pairs = (
        (sdk_manifest.ExtensionResourceV1, platform_plugin.ExtensionResourceV1),
        (sdk_manifest.InferenceComputeV1, platform_plugin.InferenceComputeV1),
        (sdk_manifest.InferenceModelCapabilityV1, platform_plugin.PluginModelCapability),
        (sdk_manifest.InferencePoolTopologyV1, platform_plugin.PluginPoolTopology),
        (
            sdk_manifest.InferenceManagedModelCompatibilityV1,
            platform_plugin.ManagedModelCompatibility,
        ),
        (sdk_manifest.InferenceExtensionV1, platform_plugin.InferenceExtensionV1),
    )
    for sdk_model, platform_model in pairs:
        assert _deref_schema(sdk_model.model_json_schema()) == _deref_schema(
            platform_model.model_json_schema()
        ), f"JSON Schema drift between {sdk_model.__name__} and {platform_model.__name__}"

    sdk_envelope = sdk_manifest.ExtensionManifestV1.model_json_schema()
    platform_envelope = platform_plugin.InferenceExtensionManifestV1.model_json_schema()
    assert set(sdk_envelope["required"]) == {
        "manifest_schema",
        "extension_point",
        "plugin_id",
        "version",
        "publisher",
        "license",
        "protocol_version",
        "capabilities",
    }
    assert set(platform_envelope["required"]) == {
        "manifest_schema",
        "extension_point",
        "plugin_id",
        "version",
        "publisher",
        "license",
        "protocol_version",
        "capabilities",
        "extension",
    }


def test_load_manifest_supports_v1_layered_legacy_flat_and_round_trips(tmp_path: Path) -> None:
    artifact = f"registry.example/yolo@sha256:{DIGEST_B}"
    layered_direct = {
        "manifest_schema": sdk_manifest.EXTENSION_MANIFEST_SCHEMA,
        "extension_point": "inference",
        "plugin_id": "demo.ocr",
        "version": "1.0.0",
        "display_name": "Demo OCR",
        "publisher": "Team",
        "license": "MIT",
        "protocol_version": sdk_manifest.INFERENCE_PROTOCOL_VERSION,
        "capabilities": ["inference", "ocr", "managed_models"],
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
            "compute": {"device": "cpu", "gpu_memory_mb": 0},
            "managed_model_compatibility": [
                {"framework": "custom", "format": None, "runtime_device": "cpu"}
            ],
        },
    }
    v1_file = tmp_path / "v1.json"
    v1_file.write_text(json.dumps(layered_direct), encoding="utf-8")
    loaded_v1 = load_manifest(v1_file)
    assert loaded_v1.execution_mode == "direct"
    assert loaded_v1.artifact is None

    roundtrip_v1_file = tmp_path / "roundtrip-v1.json"
    roundtrip_v1_file.write_text(loaded_v1.model_dump_json(), encoding="utf-8")
    assert load_manifest(roundtrip_v1_file) == loaded_v1

    legacy_flat = {
        "plugin_id": "legacy.det",
        "version": "1.0",
        "publisher": "Example",
        "license": "MIT",
        "artifact": artifact,
        "sdk_compatibility": ">=1",
        "entrypoint": ["python", "-m", "plugin"],
        "capabilities": ["inference", "detection"],
        "models": [
            {
                "id": "detect",
                "task": "detect",
                "is_interactive": False,
                "supported_prompts": ["none"],
                "supported_inputs": ["full_image"],
                "supported_geometric_outputs": ["bbox"],
            }
        ],
        "pools": {"models": {"cap": 1}},
        "resource": {"device": "cpu", "memory_mb": 512, "gpu_memory_mb": 0},
        "asset_manifest_sha256": DIGEST_A,
        "allowed_asset_sources": ["https://example.com/assets"],
        "managed_model_compatibility": [],
    }
    legacy_file = tmp_path / "legacy.json"
    legacy_file.write_text(json.dumps(legacy_flat), encoding="utf-8")
    loaded_legacy = load_manifest(legacy_file)
    assert loaded_legacy.artifact == artifact
    assert set(loaded_legacy.model_dump(mode="json")) == set(legacy_flat)

    roundtrip_legacy_file = tmp_path / "roundtrip-legacy.json"
    roundtrip_legacy_file.write_text(loaded_legacy.model_dump_json(), encoding="utf-8")
    assert load_manifest(roundtrip_legacy_file) == loaded_legacy

    missing_format = json.loads(json.dumps(layered_direct))
    del missing_format["extension"]["managed_model_compatibility"][0]["format"]
    with pytest.raises(ValidationError):
        sdk_manifest.ExtensionManifestV1.model_validate(missing_format)
    with pytest.raises(ValidationError):
        platform_plugin.InferenceExtensionManifestV1.model_validate(missing_format)
