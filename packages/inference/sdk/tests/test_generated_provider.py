"""Behavior of the generated CPU algorithm."""
import importlib.util

import pytest
from PIL import Image

from aap_inference_plugin import Detection
from aap_inference_plugin.cli import init_project


def test_generated_color_region_bounds_and_empty_image(tmp_path):
    project = tmp_path / "project"
    init_project(project)
    spec = importlib.util.spec_from_file_location("color_demo", project / "my_color_region_plugin.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.Provider().load("colors", "cpu", {})
    image = Image.new("RGB", (10, 8), "black")
    image.putpixel((2, 3), (200, 80, 80))
    image.putpixel((8, 6), (255, 0, 0))
    image.putpixel((0, 0), (199, 0, 0))
    assert model.predict(image, {}) == [Detection(2, 3, 9, 7, 0, 1.0)]
    image.putpixel((1, 1), (0, 200, 0))
    image.putpixel((3, 2), (80, 255, 80))
    image.putpixel((9, 7), (0, 0, 255))
    assert model.predict(image, {}) == [
        Detection(2, 3, 9, 7, 0, 1.0),
        Detection(1, 1, 4, 3, 1, 1.0),
        Detection(9, 7, 10, 8, 2, 1.0),
    ]
    assert model.predict(Image.new("RGB", (10, 8), (255, 81, 0)), {}) == []
    with pytest.raises(ValueError):
        module.Provider().load("colors", "cuda:0", {})
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model.predict(image, {})
