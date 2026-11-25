from __future__ import annotations

from typing import Any

from yoao_plugin_sdk.manifest import (
    INFERENCE_CAPABILITIES,
    INFERENCE_EXECUTION_MODES,
    INFERENCE_PROTOCOL_VERSION,
    INFERENCE_TASK_VALUES,
    INFERENCE_WIRE_PROTOCOL_VERSION,
    InferenceComputeV1,
    InferenceExtensionV1,
    InferenceManagedModelCompatibilityV1,
    InferenceModelCapabilityV1,
    InferencePoolTopologyV1,
)

SUPPORTED_CAPABILITIES = INFERENCE_CAPABILITIES

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
    "INFERENCE_CAPABILITIES",
    "INFERENCE_EXECUTION_MODES",
    "INFERENCE_PROTOCOL_VERSION",
    "INFERENCE_TASK_VALUES",
    "INFERENCE_WIRE_PROTOCOL_VERSION",
    "SUPPORTED_CAPABILITIES",
    "InferenceComputeV1",
    "InferenceExtensionV1",
    "InferenceManagedModelCompatibilityV1",
    "InferenceModelCapabilityV1",
    "InferencePoolTopologyV1",
    "Detection",
    "DetectionConfig",
    "DetectionModel",
    "DetectionProvider",
    "create_detection_app",
    "load_manifest",
]
