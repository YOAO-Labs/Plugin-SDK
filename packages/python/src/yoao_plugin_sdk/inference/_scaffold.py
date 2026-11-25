"""Pure-stdlib scaffolding for the CPU color-region detection demo plugin."""

from __future__ import annotations

from pathlib import Path


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
    "yoao-plugin-sdk[inference]==0.1.0",
    "yoao-inference-contract==0.1.0",
    "yoao-lifecycle-contract==0.1.0",
    "yoao-protocol-v2==0.2.0",
    "yoao-task-contract==0.1.0",
    "yoao-training-contract==0.1.0",
]

[project.scripts]
my-color-region-plugin = "my_color_region_plugin:main"

[tool.setuptools]
py-modules = ["my_color_region_plugin"]

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

from PIL import ImageChops
from PIL.Image import Image
from yoao_plugin_sdk.inference import Detection, DetectionConfig, DetectionModel, create_detection_app
import uvicorn


class ColorRegionModel:
    """One enclosing box per dominant red, green or blue region."""

    classes = ("red-region", "green-region", "blue-region")

    def __init__(self) -> None:
        self.closed = False

    def predict(self, image: Image, parameters: Mapping[str, Any]) -> Sequence[Detection]:
        if self.closed:
            raise RuntimeError("model is closed")
        channels = image.convert("RGB").split()
        detections = []
        for index, channel in enumerate(channels):
            mask = channel.point(lambda value: 255 if value >= 200 else 0)
            for other_index, other in enumerate(channels):
                if other_index != index:
                    mask = ImageChops.multiply(mask, other.point(lambda value: 255 if value <= 80 else 0))
            box = mask.getbbox()
            if box:
                detections.append(Detection(*box, class_index=index, score=1.0))
        return detections

    def close(self) -> None:
        self.closed = True


class Provider:
    def load(self, model_id: str, device: str, config: Mapping[str, Any]) -> DetectionModel:
        if model_id != "colors" or device != "cpu":
            raise ValueError("color-region example supports only colors on cpu")
        return ColorRegionModel()


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU color-region detection demo")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--token-file", type=Path, required=True)
    serve.add_argument("--device", choices=["cpu"], default="cpu")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    token = args.token_file.read_text(encoding="utf-8").strip()
    config = DetectionConfig(token=token, device=args.device, catalog={"colors": {}},
                             default_model="colors", series="color-region")
    uvicorn.run(create_detection_app(Provider(), config), host=args.host, port=args.port)
''',
        encoding="utf-8",
    )
