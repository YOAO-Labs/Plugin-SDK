from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from yoao_plugin_sdk import (
    EXTENSION_MANIFEST_SCHEMA,
    ExtensionManifestV1,
    load_manifest_file,
    read_secret,
    verify_bearer_token,
)
from yoao_plugin_sdk.cli import main as cli_main
from yoao_plugin_sdk.format_adapter import ImportParseResponse, ParsedAnnotation, ParsedItem
from yoao_plugin_sdk.notification import DeliverRequest, DeliverResult, DeliveryEnvelope


@pytest.mark.parametrize(
    "extension_point",
    ("inference", "notification_channel", "format_adapter", "theme"),
)
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
