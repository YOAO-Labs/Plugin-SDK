from __future__ import annotations

from typing import Any

INFERENCE_PROTOCOL_VERSION = "yoao-inference/v2"
SUPPORTED_CAPABILITIES = ("predict", "train", "track", "health")

_AUTHORING_EXPORTS = frozenset(
    {
        "Detection",
        "DetectionConfig",
        "DetectionModel",
        "DetectionProvider",
        "create_detection_app",
        "load_manifest",
    }
)


def __getattr__(name: str) -> Any:
    if name in _AUTHORING_EXPORTS:
        try:
            from . import authoring
        except ImportError as exc:
            raise ImportError(
                f"yoao_plugin_sdk.inference.{name} requires the inference extra: "
                "pip install 'yoao-plugin-sdk[inference]'"
            ) from exc
        return getattr(authoring, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "INFERENCE_PROTOCOL_VERSION",
    "SUPPORTED_CAPABILITIES",
    "Detection",
    "DetectionConfig",
    "DetectionModel",
    "DetectionProvider",
    "create_detection_app",
    "load_manifest",
]
