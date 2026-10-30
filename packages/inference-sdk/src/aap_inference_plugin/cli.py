"""Standalone development commands for detection plugins."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import httpx


def init_project(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "pyproject.toml").write_text(
        """[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my-color-region-plugin"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "aap-inference-plugin-sdk==0.1.0",
    "aap-backend-runtime==0.1.0",
    "aap-inference-contract==0.1.0",
    "aap-lifecycle-contract==0.1.0",
    "aap-protocol-v2==0.2.0",
    "aap-task-contract==0.1.0",
    "aap-training-contract==0.1.0",
]

[project.scripts]
my-color-region-plugin = "my_color_region_plugin:main"

[tool.setuptools]
py-modules = ["my_color_region_plugin"]

[tool.uv.sources]
aap-inference-plugin-sdk = { url = "https://github.com/YOAO-Labs/Plugin-SDK/releases/download/python-sdk-0.1.0/aap_inference_plugin_sdk-0.1.0-py3-none-any.whl" }
aap-backend-runtime = { url = "https://github.com/YOAO-Labs/Plugin-SDK/releases/download/python-sdk-0.1.0/aap_backend_runtime-0.1.0-py3-none-any.whl" }
aap-inference-contract = { url = "https://github.com/YOAO-Labs/Plugin-SDK/releases/download/contracts-3d6393e64630/aap_inference_contract-0.1.0-py3-none-any.whl" }
aap-lifecycle-contract = { url = "https://github.com/YOAO-Labs/Plugin-SDK/releases/download/contracts-3d6393e64630/aap_lifecycle_contract-0.1.0-py3-none-any.whl" }
aap-protocol-v2 = { url = "https://github.com/YOAO-Labs/Plugin-SDK/releases/download/contracts-3d6393e64630/aap_protocol_v2-0.2.0-py3-none-any.whl" }
aap-task-contract = { url = "https://github.com/YOAO-Labs/Plugin-SDK/releases/download/contracts-3d6393e64630/aap_task_contract-0.1.0-py3-none-any.whl" }
aap-training-contract = { url = "https://github.com/YOAO-Labs/Plugin-SDK/releases/download/contracts-3d6393e64630/aap_training_contract-0.1.0-py3-none-any.whl" }
""",
        encoding="utf-8",
    )
    (directory / "my_color_region_plugin.py").write_text(
        '''"""CPU color-region algorithm example, not a pretrained model."""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL.Image import Image
from aap_inference_plugin import Detection, DetectionConfig, DetectionModel, create_detection_app
import uvicorn


class ColorRegionModel:
    """One box enclosing all pixels with red >= 200 and green/blue <= 80."""

    classes = ("red-region",)

    def __init__(self) -> None:
        self.closed = False

    def predict(self, image: Image, parameters: Mapping[str, Any]) -> Sequence[Detection]:
        if self.closed:
            raise RuntimeError("model is closed")
        red, green, blue = image.convert("RGB").split()
        mask = red.point(lambda value: 255 if value >= 200 else 0)
        from PIL import ImageChops
        mask = ImageChops.multiply(mask, green.point(lambda value: 255 if value <= 80 else 0))
        mask = ImageChops.multiply(mask, blue.point(lambda value: 255 if value <= 80 else 0))
        box = mask.getbbox()
        return [Detection(*box, class_index=0, score=1.0)] if box else []

    def close(self) -> None:
        self.closed = True


class Provider:
    def load(self, model_id: str, device: str, config: Mapping[str, Any]) -> DetectionModel:
        if model_id != "red" or device != "cpu":
            raise ValueError("color-region example supports only red on cpu")
        return ColorRegionModel()


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU color-region detection demo")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--token-file", type=Path, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    token = args.token_file.read_text(encoding="utf-8").strip()
    config = DetectionConfig(token=token, device="cpu", catalog={"red": {}},
                             default_model="red", series="color-region")
    uvicorn.run(create_detection_app(Provider(), config), host=args.host, port=args.port)
''',
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="aap-inference-plugin")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("directory", type=Path)
    verify = commands.add_parser("check")
    mode = verify.add_mutually_exclusive_group(required=True)
    mode.add_argument("--url", help="Check a running HTTP endpoint only")
    mode.add_argument(
        "--qualify-local",
        action="store_true",
        help="Run full local-exclusive GPU qualification",
    )
    verify.add_argument("--fixtures", required=True, type=Path)
    verify.add_argument(
        "--entrypoint",
        help="Installed author console script for local qualification",
    )
    verify.add_argument("--device")
    verify.add_argument("--gpu-uuid")
    verify.add_argument("--asset-dir", type=Path)
    verify.add_argument("--exclusive-gpu", action="store_true")
    verify.add_argument("--report", type=Path)
    build = commands.add_parser("build")
    build.add_argument("directory", nargs="?", type=Path, default=Path.cwd())
    build.add_argument(
        "--base-image", help="Digest-pinned base image for an offline image build"
    )
    build.add_argument("--wheelhouse", type=Path)
    build.add_argument("--requirements-lock", type=Path)
    build.add_argument(
        "--entrypoint",
        help="Installed author console script (required for image builds)",
    )
    build.add_argument("--tag")
    build.add_argument("--push", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "init":
            init_project(args.directory)
        elif args.command == "check":
            if args.qualify_local:
                if not (
                    args.entrypoint
                    and args.device
                    and args.gpu_uuid
                    and args.asset_dir
                    and args.report
                    and args.exclusive_gpu
                ):
                    raise ValueError(
                        "--qualify-local requires --entrypoint, --device, --gpu-uuid, --asset-dir, --exclusive-gpu and --report"
                    )
                from .qualification import qualify_local

                report = qualify_local(
                    args.fixtures,
                    args.device,
                    args.gpu_uuid,
                    args.asset_dir,
                    args.report,
                    entrypoint=args.entrypoint,
                )
                print(
                    json.dumps(
                        {
                            "mode": report["mode"],
                            "report": str(args.report),
                            "platform_managed_qualified": False,
                        }
                    )
                )
            else:
                if args.entrypoint or args.device or args.gpu_uuid or args.asset_dir or args.exclusive_gpu:
                    raise ValueError(
                        "GPU qualification options require --qualify-local"
                    )
                from .check import check_http

                report = check_http(
                    args.url, os.environ.get("AAP_PLUGIN_TOKEN", ""), args.fixtures
                )
                if args.report:
                    args.report.write_text(
                        json.dumps(report, indent=2) + "\n", encoding="utf-8"
                    )
                print(json.dumps(report))
        else:
            image_options = (
                args.base_image,
                args.wheelhouse,
                args.requirements_lock,
                args.tag,
            )
            if any(image_options) or args.entrypoint or args.push:
                if not all(image_options) or not args.entrypoint:
                    raise ValueError(
                        "image build requires --entrypoint, --base-image, --wheelhouse, --requirements-lock and --tag"
                    )
                from .image_build import build_image

                build_image(
                    base_image=args.base_image,
                    wheelhouse=args.wheelhouse,
                    requirements_lock=args.requirements_lock,
                    entrypoint=args.entrypoint,
                    tag=args.tag,
                    push=args.push,
                )
            else:
                subprocess.run(
                    [sys.executable, "-m", "build", "--wheel", str(args.directory)],
                    check=True,
                )
                for artifact in sorted((args.directory / "dist").glob("*.whl")):
                    print(
                        f"{hashlib.sha256(artifact.read_bytes()).hexdigest()}  {artifact}"
                    )
    except (OSError, ValueError, httpx.HTTPError, subprocess.SubprocessError) as exc:
        print(f"aap-inference-plugin: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
