from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from yoao_plugin_sdk._version import __version__
from yoao_plugin_sdk.manifest import (
    EXTENSION_MANIFEST_SCHEMA,
    EXTENSION_POINTS,
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
    if extension_point == "trainer":
        return {
            "manifest_schema": EXTENSION_MANIFEST_SCHEMA,
            "extension_point": "trainer",
            "plugin_id": f"custom.trainer.{slug}",
            "version": "0.1.0",
            "display_name": name,
            "description": f"Custom YOAO trainer plugin ({name}).",
            "publisher": "Custom Publisher",
            "license": "MIT",
            "protocol_version": "yoao-trainer/v1",
            "entrypoint": ["python", "-m", "plugin.trainer"],
            "capabilities": ["train", "evaluate", "export_weights"],
            "resource": {"memory_mb": 4096},
            "extension": {
                "framework_id": slug.replace("-", "_"),
                "framework_version": "0.1.0",
                "supported_tasks": ["detect"],
                "supported_dataset_contracts": ["ultralytics-bbox-1"],
                "requires_gpu": True,
                "preview_cpu_only": True,
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
        "capabilities": ["inference", "detection"],
        "resource": {"memory_mb": 1024},
        "extension": {
            "execution_mode": "direct",
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
            "compute": {"device": "cpu", "gpu_memory_mb": 0},
            "allowed_asset_sources": [],
            "managed_model_compatibility": [],
        },
    }


def _run_inference_authoring(argv: list[str]) -> int:
    try:
        from yoao_plugin_sdk.inference.authoring import cli as inference_cli
    except ImportError as exc:
        print(
            f"yoao-plugin: '{argv[0]}' requires optional inference dependencies "
            f"(pip install 'yoao-plugin-sdk[inference]'): {exc}",
            file=sys.stderr,
        )
        return 1
    try:
        inference_cli.main(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] in {"dev", "check", "build"}:
        return _run_inference_authoring(args_list)

    parser = argparse.ArgumentParser(
        prog="yoao-plugin",
        description=(
            "Universal YOAO plugin CLI (version, validate, init) plus inference "
            "authoring commands (dev, check, build)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="Print yoao-plugin-sdk version")

    val_p = sub.add_parser("validate", help="Validate a yoao-extension-manifest/v1 JSON file")
    val_p.add_argument("manifest_path", type=Path)

    init_p = sub.add_parser("init", help="Scaffold a new YOAO plugin directory")
    init_p.add_argument(
        "--type",
        dest="extension_point",
        choices=EXTENSION_POINTS,
        required=True,
    )
    init_p.add_argument(
        "--example",
        action="store_true",
        help="For inference plugins, also scaffold the runnable CPU color-region demo package.",
    )
    init_p.add_argument("target_dir", type=Path)

    sub.add_parser("dev", help="Run a local inference plugin development server (requires [inference])")
    sub.add_parser("check", help="Check an inference plugin endpoint or run local GPU qualification")
    sub.add_parser("build", help="Build a plugin wheel or locked OCI container image")

    args = parser.parse_args(args_list)
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
        use_example = bool(args.example and args.extension_point == "inference")
        if use_example:
            from yoao_plugin_sdk.inference._scaffold import init_project

            try:
                init_project(target)
            except FileExistsError:
                print(f"Refusing to overwrite existing directory {target}", file=sys.stderr)
                return 1
        else:
            target.mkdir(parents=True, exist_ok=True)
        manifest_path = target / "manifest.json"
        if manifest_path.exists():
            print(f"Refusing to overwrite existing {manifest_path}", file=sys.stderr)
            return 1
        doc = _template_manifest(args.extension_point, target.name)
        if use_example:
            doc["entrypoint"] = [
                "my-color-region-plugin",
                "serve",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ]
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
