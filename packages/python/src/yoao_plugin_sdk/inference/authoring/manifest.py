"""Validate plugin installation intent using the shared inference contract."""
from pathlib import Path

from yoao_inference_contract.plugin import PluginManifest, parse_inference_manifest


def load_manifest(path: str | Path) -> PluginManifest:
    return parse_inference_manifest(Path(path).read_text(encoding="utf-8"))
