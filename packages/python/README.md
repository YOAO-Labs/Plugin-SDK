# `yoao-plugin-sdk`

English | [简体中文](README.zh-CN.md)

Unified Python SDK (`requires-python >= 3.10`) for authoring YOAO plugins across all five extension points (`inference`, `trainer`, `notification_channel`, `format_adapter`, and `theme`). See the [Plugin-SDK repository README](../../README.md) for workspace overview and related repositories.

## Capabilities

### Included in the base package (`pydantic>=2.5`)

- **Manifest validation** (`yoao_plugin_sdk.manifest`, re-exported from `yoao_plugin_sdk`): `ExtensionManifestV1`, `load_manifest_file`, and `validate_manifest_dict` for strict `yoao-extension-manifest/v1` validation, including `container` and `direct` inference modes.
- **Auth helpers** (`yoao_plugin_sdk.auth`): `verify_bearer_token` for constant-time Bearer token comparison and `read_secret` for direct or `*_FILE` secret loading.
- **Protocol models**:
  - `yoao_plugin_sdk.inference`: `yoao-inference/v2` manifest models and capability constants.
  - `yoao_plugin_sdk.trainer`: `yoao-trainer/v1` contract models (`TrainerCapabilitiesV1`, `ValidateConfigResultV1`, `TrainSpecV1`, `TrainEventV1`, `TrainReceiptV1`, `PreviewSpecV1`, `PreviewReportV1`, and related types), canonical JSON hashing (`canonical_json_bytes`, `canonical_json_digest`), and atomic writers (`atomic_write_json`, `write_event_jsonl`).
  - `yoao_plugin_sdk.notification`: `yoao-notification-channel/v1` models (`DeliveryEnvelope`, `DeliverRequest`, `DeliverResult`).
  - `yoao_plugin_sdk.format_adapter`: `yoao-format-adapter/v1` models (`ParsedAnnotation`, `ParsedItem`, `ImportParseResponse`).
- **Base CLI** (`yoao-plugin`): `version`, `init`, and `validate`.

### Optional extras

- `[inference-runtime]` (`yoao_plugin_sdk.inference.runtime`): `ManagedLruPool`, `TrackerSessionManager`, `ManagedGpuLifecycle`, device and GPU helpers, lifecycle evidence helpers, and `plugin_service` route and auth integration.
- `[inference]` (`yoao_plugin_sdk.inference`, `yoao_plugin_sdk.inference.authoring`): framework-independent detection authoring API (`Detection`, `DetectionConfig`, `DetectionModel`, `DetectionProvider`, `create_detection_app`, `load_manifest`) and `yoao-plugin` `dev`, `check`, and `build` commands.
- `[mask]` (`yoao_plugin_sdk.inference.mask_utils`): `mask_to_polygon`, `mask_to_multi_polygon`, `mask_to_preview_polygon`, `normalize_coords`, COCO RLE codecs (`encode_coco_rle`, `decode_coco_rle`, `validate_coco_rle`), and prompt adapters.
- `[qualification]` (`yoao_plugin_sdk.inference.runtime.qualification_http`): ephemeral loopback HTTP client and route-admission helper for local qualification runs.
- `[notification]`: adds `httpx` for notification channel HTTP servers.
- `[trainer]` and `[format]`: extension-point extras with no extra runtime dependencies beyond the base package.

`[inference-runtime]` and `[inference]` depend on `yoao-sdk[protocol]` from [YOAO](https://github.com/YOAO-Labs/YOAO). All other extras work standalone without `yoao-sdk`. `yoao-plugin-sdk` never pins PyTorch or TorchVision.

## Installation

Install from a local repository checkout (from `packages/python`):

```sh
# Base package (pydantic>=2.5 only)
python -m pip install -e .

# Optional extras
python -m pip install -e ".[mask]"
python -m pip install -e ".[notification]"
python -m pip install -e ".[dev]"
```

When installing from published wheels or a local wheelhouse (with `yoao-sdk[protocol]` available for inference extras):

```sh
python -m pip install yoao-plugin-sdk
python -m pip install "yoao-plugin-sdk[inference-runtime]"
python -m pip install "yoao-plugin-sdk[inference]"
python -m pip install "yoao-plugin-sdk[inference-runtime,mask]"
```

## Quick start

```sh
yoao-plugin version
yoao-plugin init --type trainer ./my-trainer-plugin
yoao-plugin validate ./my-trainer-plugin/manifest.json

# Scaffold the runnable CPU color-region detection demo (running it requires [inference])
yoao-plugin init --type inference --example ./my-detector-plugin
```

For inference authoring commands (after installing `[inference]`):

```sh
yoao-plugin dev --help
yoao-plugin check --help
yoao-plugin build --help
```

## Testing

From `packages/python`:

```sh
# Core tests (base + [dev])
python -m pytest tests/test_sdk.py

# Mask utilities tests ([mask,dev])
python -m pytest --import-mode=importlib tests/mask_utils

# Full suite (requires yoao-sdk[protocol] and [inference,mask,qualification,dev])
python -m pytest --import-mode=importlib tests
```

## License

First-party source in this package is licensed under the [MIT License](LICENSE). Third-party dependencies and plugin assets retain their own licenses and terms.
