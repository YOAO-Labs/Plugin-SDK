from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from yoao_plugin_sdk._version import __version__
from yoao_plugin_sdk.manifest import (
    EXTENSION_MANIFEST_SCHEMA,
    load_manifest_file,
)


def _template_manifest(extension_point: str, name: str) -> dict[str, Any]:
    slug = name.strip().lower().replace("_", "-")
    if extension_point == "notification_channel":
        return {
            "manifest_schema": EXTENSION_MANIFEST_SCHEMA,
            "extension_point": "notification_channel",
            "plugin_id": f"custom.notification.{slug}",
            "version": "0.1.0",
            "display_name": name,
            "description": f"Custom YOAO notification channel plugin ({name}).",
            "publisher": "Custom Publisher",
            "license": "MIT",
            "protocol_version": "yoao-notification-channel/v1",
            "entrypoint": ["python", "-m", "plugin.server"],
            "capabilities": ["deliver", "verify_address"],
            "resource": {"memory_mb": 128},
            "extension": {
                "channel_id": slug.replace("-", "_"),
                "address_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["recipient_id"],
                    "properties": {
                        "recipient_id": {"type": "string", "minLength": 1, "maxLength": 128}
                    },
                },
                "allowed_egress_hosts": [],
            },
        }
    if extension_point == "format_adapter":
        return {
            "manifest_schema": EXTENSION_MANIFEST_SCHEMA,
            "extension_point": "format_adapter",
            "plugin_id": f"custom.format.{slug}",
            "version": "0.1.0",
            "display_name": name,
            "description": f"Custom YOAO format adapter plugin ({name}).",
            "publisher": "Custom Publisher",
            "license": "MIT",
            "protocol_version": "yoao-format-adapter/v1",
            "entrypoint": ["python", "-m", "plugin.server"],
            "capabilities": ["import", "export"],
            "resource": {"memory_mb": 256},
            "extension": {
                "format_id": slug,
                "adapter_version": "0.1.0",
                "manifest_version": "1",
                "media_types": ["image"],
                "option_schema": {},
                "allowed_egress_hosts": [],
            },
        }
    if extension_point == "theme":
        return {
            "manifest_schema": EXTENSION_MANIFEST_SCHEMA,
            "extension_point": "theme",
            "plugin_id": f"custom.theme.{slug}",
            "version": "0.1.0",
            "display_name": name,
            "description": f"Declarative YOAO theme plugin ({name}).",
            "publisher": "Custom Publisher",
            "license": "MIT",
            "protocol_version": "yoao-theme/v1",
            "capabilities": ["light", "dark"],
            "extension": {
                "theme_id": slug,
                "tokens_file": "tokens.json",
            },
        }
    return {
        "manifest_schema": EXTENSION_MANIFEST_SCHEMA,
        "extension_point": "inference",
        "plugin_id": f"custom.inference.{slug}",
        "version": "0.1.0",
        "display_name": name,
        "description": f"Custom YOAO inference plugin ({name}).",
        "publisher": "Custom Publisher",
        "license": "MIT",
        "protocol_version": "yoao-inference/v2",
        "entrypoint": ["python", "-m", "plugin.server"],
        "capabilities": ["predict", "health"],
        "resource": {"memory_mb": 1024},
        "extension": {},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="yoao-plugin")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="Print yoao-plugin-sdk version")

    val_p = sub.add_parser("validate", help="Validate a yoao-extension-manifest/v1 JSON file")
    val_p.add_argument("manifest_path", type=Path)

    init_p = sub.add_parser("init", help="Scaffold a new YOAO plugin directory")
    init_p.add_argument(
        "--type",
        dest="extension_point",
        choices=("inference", "notification_channel", "format_adapter", "theme"),
        required=True,
    )
    init_p.add_argument("target_dir", type=Path)

    args = parser.parse_args(argv)
    if args.command == "version":
        print(__version__)
        return 0

    if args.command == "validate":
        try:
            manifest = load_manifest_file(args.manifest_path)
        except Exception as exc:
            print(f"INVALID: {exc}", file=sys.stderr)
            return 1
        print(
            json.dumps(
                {
                    "valid": True,
                    "extension_point": manifest.extension_point,
                    "plugin_id": manifest.plugin_id,
                    "protocol_version": manifest.protocol_version,
                }
            )
        )
        return 0

    if args.command == "init":
        target: Path = args.target_dir
        target.mkdir(parents=True, exist_ok=True)
        manifest_path = target / "manifest.json"
        if manifest_path.exists():
            print(f"Refusing to overwrite existing {manifest_path}", file=sys.stderr)
            return 1
        doc = _template_manifest(args.extension_point, target.name)
        manifest_path.write_text(
            json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if args.extension_point == "theme":
            (target / "tokens.json").write_text(
                json.dumps(
                    {
                        "light": {"--sc-bg": "#ffffff", "--sc-fg": "#111827"},
                        "dark": {"--sc-bg": "#0b1120", "--sc-fg": "#f8fafc"},
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        print(f"Scaffolded {args.extension_point} plugin at {target}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
