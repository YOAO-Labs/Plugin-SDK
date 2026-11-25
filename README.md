# YOAO Plugin SDK

YOAO stands for **You Only Annotate Once**. This repository is the multi-language monorepo for authors of independent YOAO plugins across all extension points: inference (`yoao-inference/v2`), trainers (`yoao-trainer/v1`), notification channels (`yoao-notification-channel/v1`), format adapters (`yoao-format-adapter/v1`), and declarative themes (`yoao-theme/v1`).

`yoao plugins` belongs to the separate [platform repository](https://github.com/YOAO-Labs/ai-annotation-platform) and manages platform installations. The platform owns portable contracts, authorization, and storage. [YOAO-Plugins](https://github.com/YOAO-Labs/YOAO-Plugins) maintains official plugin implementations. First-party SDK source is [MIT](LICENSE); third-party dependencies and assets retain their own terms.

## Workspace layout

| Path | Distribution | Runtime | Purpose |
| --- | --- | --- | --- |
| `packages/python` | `yoao-plugin-sdk` | Python `>=3.10` | Unified Python SDK: `yoao-extension-manifest/v1` validation, auth helpers, `yoao-plugin` CLI, inference authoring/runtime/mask utilities, trainer/notification/format protocol models |
| `packages/js` | `@yoao/plugin-sdk` | Node.js `>=20` / TS | Zero-dependency ESM + `.d.ts` manifest validation, declarative `--sc-*` theme token validation, auth helpers, and `yoao-plugin-js` CLI |

Python plugins with different interpreter versions (3.10, 3.11, 3.12) install the same single distribution `yoao-plugin-sdk` and select only the optional extras they need:

| Extra | Module namespace | Typical consumer |
| --- | --- | --- |
| Base (`yoao-plugin-sdk`) | `yoao_plugin_sdk.{manifest,auth,cli,notification,format_adapter,trainer}` | Lightweight manifest validation, format adapters, declarative theme tooling |
| `[inference-runtime]` | `yoao_plugin_sdk.inference.runtime` | Custom inference plugins (RapidOCR, ONNX Tools, Grounded-SAM2, SAM3, YOLO) |
| `[inference]` | `yoao_plugin_sdk.inference`, `yoao_plugin_sdk.inference.authoring` | Detection authoring API (`create_detection_app`) and `yoao-plugin dev/check/build` |
| `[mask]` | `yoao_plugin_sdk.inference.mask_utils` | Mask-to-polygon, RLE, and prompt adapter helpers |
| `[qualification]` | `yoao_plugin_sdk.inference.runtime.qualification_http` | Local loopback route-admission checks |
| `[notification]` | `yoao_plugin_sdk.notification` | Notification channel HTTP servers (e.g. Feishu DM) |
| `[trainer]` | `yoao_plugin_sdk.trainer` | Isolated trainer plugins (`yoao-trainer/v1`) |
| `[format]` | `yoao_plugin_sdk.format_adapter` | Python format adapter servers (`yoao-format-adapter/v1`) |

Torch and TorchVision are never pinned by `yoao-plugin-sdk`; each plugin container or virtual environment pins its own ML framework and CUDA stack.

## Migration from legacy inference distributions

The three legacy inference-only distributions (`yoao-inference-plugin-sdk`, `yoao-backend-runtime`, `yoao-mask-utils`) were folded into `yoao-plugin-sdk` (`packages/python`). Official plugins in `YOAO-Plugins` pinned to earlier wheels should update imports and environment variables on their next candidate bump:

| Legacy package / import | Unified `yoao-plugin-sdk` target |
| --- | --- |
| `yoao-backend-runtime` / `yoao_backend_runtime.*` | `yoao-plugin-sdk[inference-runtime]` / `yoao_plugin_sdk.inference.runtime.*` |
| `yoao-inference-plugin-sdk` / `yoao_inference_plugin.*` | `yoao-plugin-sdk[inference]` / `yoao_plugin_sdk.inference.*` (or `.inference.authoring.*`) |
| `yoao-mask-utils` / `mask_utils.*` | `yoao-plugin-sdk[mask]` / `yoao_plugin_sdk.inference.mask_utils.*` |
| `AAP_PLUGIN_*` / `AAP_ROUTE_REGISTRY_ID` env vars | `YOAO_PLUGIN_*` / `ROUTE_REGISTRY_ID` (retired `AAP_*` names now fail closed) |

## Portable contracts and local development

The platform owns the portable contract source for inference (`yoao-protocol-v2`, `yoao-lifecycle-contract`, `yoao-inference-contract`, `yoao-task-contract`, `yoao-training-contract`). Base `yoao-plugin-sdk` and `@yoao/plugin-sdk` do not require those contracts; inference extras (`[inference-runtime]`, `[inference]`) do.

Managed checkpoint compatibility requires an explicit `format`: `null` identifies a native checkpoint without a separate export-format identity; named formats use nonempty strings. `null` is not a wildcard, and omitting the field is invalid.

Plugins describe required assets for each model and variant. The platform independently verifies files in the installed container and governs installation-scoped approval, availability, and project access. Publishing a descriptor grants none of those permissions; user-trained checkpoints remain under platform Learning and Assets governance.

```sh
# 1. Core Python SDK (no portable contracts required)
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e "packages/python[dev]"
yoao-plugin version
python -m pytest packages/python/tests/test_sdk.py scripts/test_fetch_contracts.py

# 2. Full Python SDK including inference runtime, authoring, and mask utilities
python -m pip install build 'setuptools>=68' wheel pytest pytest-asyncio PyJWT uvicorn
python scripts/fetch-contracts.py --manifest "$CONTRACT_MANIFEST" --sha256 "$CONTRACT_MANIFEST_SHA256" --output /tmp/yoao-contracts
python scripts/build-sdk-artifacts.py --revision HEAD --output /tmp/yoao-sdk
python -m pip install /tmp/yoao-contracts/*.whl
python -m pip install "/tmp/yoao-sdk/yoao_plugin_sdk-"*.whl"[inference,mask,qualification,dev]"
python -m pytest --import-mode=importlib packages/python/tests scripts/test_fetch_contracts.py

# 3. JavaScript / TypeScript SDK (Node >= 20, zero runtime dependencies)
node --test packages/js/test/sdk.test.mjs
```

## Stateful tracker sessions

`TrackerSessionManager` keeps idle tracker state owned by the exact `ManagedLruPool` entry. Its `start()` borrow factory must yield `(managed_lease, callback_resource)`: the first value is the actual `ResourceLease` returned by that pool, while the second is the resource or adapter passed to `open_session`, `call`, and `close_session` callbacks.

Opening and each command borrow and serialize the entry only while work runs. Eviction, idle unload, drain, or shutdown invalidate its context and await state cleanup before disposing the model; a stale token raises `TrackerSessionLost`, including after the same model key is loaded again. Cleanup runs exactly once; a raised error or `cleanup_uncertain` retains owned roots and unknown residency instead of claiming a safe unload.

## Building source release inputs

Build only from an explicit committed SDK revision; uncommitted working-tree changes are not included:

```sh
python -m pip install build 'setuptools>=68' wheel
python scripts/build-sdk-artifacts.py --revision HEAD --output /tmp/yoao-sdk-release
(cd /tmp/yoao-sdk-release && sha256sum -c SHA256SUMS)
```

The output contains `sdk-source.tar` (restricted to `packages/python`, `packages/js`, root metadata, `LICENSE`, and `README.md`), the `yoao-plugin-sdk` wheel and sdist, `manifest.json`, and `SHA256SUMS`. Local builds do not publish anything.
