from __future__ import annotations

from yoao_plugin_sdk._version import __version__
from yoao_plugin_sdk.auth import read_secret, verify_bearer_token
from yoao_plugin_sdk.manifest import (
    EXTENSION_MANIFEST_SCHEMA,
    EXTENSION_POINTS,
    ExtensionManifestV1,
    ExtensionResourceV1,
    load_manifest_file,
    validate_manifest_dict,
)

__all__ = [
    "EXTENSION_MANIFEST_SCHEMA",
    "EXTENSION_POINTS",
    "ExtensionManifestV1",
    "ExtensionResourceV1",
    "__version__",
    "load_manifest_file",
    "read_secret",
    "validate_manifest_dict",
    "verify_bearer_token",
]
