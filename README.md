# YOAO Plugin SDK

English | [简体中文](README.zh-CN.md)

Plugin SDKs, CLIs, and shared runtime libraries for authors of independent YOAO (**You Only Annotate Once**) plugins.

This repository covers all five YOAO extension points under the `yoao-extension-manifest/v1` manifest schema:

- `inference` (`yoao-inference/v2`, container and direct modes)
- `trainer` (`yoao-trainer/v1`)
- `notification_channel` (`yoao-notification-channel/v1`)
- `format_adapter` (`yoao-format-adapter/v1`)
- `theme` (`yoao-theme/v1`)

## Packages

| Path | Package | Runtime | Summary |
| --- | --- | --- | --- |
| [`packages/python`](packages/python/README.md) | `yoao-plugin-sdk` | Python `>=3.10` | Manifest validation, Bearer and `*_FILE` auth helpers, protocol models, `yoao-plugin` CLI, and optional inference runtime, authoring, and mask utilities |
| [`packages/js`](packages/js/README.md) | `@yoao/plugin-sdk` | Node.js `>=20` and TypeScript | Zero-dependency ESM and `.d.ts` package for manifest validation, declarative `--sc-*` theme validation, `yoao-trainer/v1` validators, auth helpers, and `yoao-plugin-js` CLI |

## Python optional extras

Python plugins on Python 3.10, 3.11, or 3.12 install the single `yoao-plugin-sdk` distribution and add only the extras they need.

| Extra | Module namespace | Scope |
| --- | --- | --- |
| Base (`yoao-plugin-sdk`) | `yoao_plugin_sdk.{manifest,auth,cli,inference,trainer,notification,format_adapter}` | Manifest validation, auth helpers, `yoao-plugin` base commands (`version`, `validate`, `init`, `trainer check`), protocol models, and dependency-free trainer runtime components (`pydantic>=2.5` only) |
| `[inference-runtime]` | `yoao_plugin_sdk.inference.runtime` | Model LRU pool, tracker sessions, managed GPU lifecycle, and `plugin_service` route and auth helpers |
| `[inference]` | `yoao_plugin_sdk.inference`, `yoao_plugin_sdk.inference.authoring` | Detection authoring API (`create_detection_app`) and `yoao-plugin` `dev`, `check`, and `build` commands |
| `[mask]` | `yoao_plugin_sdk.inference.mask_utils` | Mask-to-polygon conversion, COCO RLE codecs, and prompt adapters |
| `[qualification]` | `yoao_plugin_sdk.inference.runtime.qualification_http` | Local loopback route-admission checks |
| `[notification]` | `yoao_plugin_sdk.notification` | Adds `httpx` for notification channel HTTP servers |
| `[trainer]` | `yoao_plugin_sdk.trainer` | Extension-point extra for trainer plugins (no extra runtime dependencies beyond base) |
| `[format]` | `yoao_plugin_sdk.format_adapter` | Extension-point extra for format adapter plugins (no extra runtime dependencies beyond base) |

`yoao-plugin-sdk` never pins PyTorch or TorchVision. Each plugin environment or container image pins its own ML framework and CUDA stack.

Base `yoao-plugin-sdk`, `[mask]`, `[notification]`, `[trainer]`, `[format]`, and `@yoao/plugin-sdk` work standalone without platform contract packages. Only `[inference-runtime]` and `[inference]` depend on `yoao-sdk[protocol]`, the portable contract distribution owned by [YOAO](https://github.com/YOAO-Labs/YOAO).

## Quick start

### Python SDK

Install the base Python SDK in editable mode from a local checkout, scaffold a plugin manifest, and run the core tests:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e "packages/python[dev]"
yoao-plugin version
yoao-plugin init --type trainer --example ./my-trainer-plugin
yoao-plugin validate ./my-trainer-plugin/manifest.json
python -m pip install --no-deps --no-build-isolation ./my-trainer-plugin
yoao-plugin trainer check --manifest ./my-trainer-plugin/manifest.json --entrypoint my-trainer-plugin --preview
python -m pytest packages/python/tests/test_sdk.py packages/python/tests/trainer scripts/test_fetch_contracts.py
```

To test the mask utilities extra, install `packages/python[mask,dev]` and run `python -m pytest --import-mode=importlib packages/python/tests/mask_utils`. For package details, CLI subcommands, and inference extras, see [`packages/python/README.md`](packages/python/README.md).

### JavaScript and TypeScript SDK

Run the `@yoao/plugin-sdk` test suite and CLI directly with Node.js `>=20` (no install step required):

```sh
node --test packages/js/test/sdk.test.mjs
node packages/js/src/cli.mjs validate ./my-trainer-plugin/manifest.json
```

For exports and CLI usage, see [`packages/js/README.md`](packages/js/README.md).

## Related repositories

| Repository | Role |
| --- | --- |
| [YOAO](https://github.com/YOAO-Labs/YOAO) | Core annotation platform. Owns `yoao plugins` management, portable contract source (`yoao-sdk`), authorization, storage, and asset governance |
| [Plugin-SDK](https://github.com/YOAO-Labs/Plugin-SDK) | This repository. Provides multi-language SDKs, CLIs, and shared runtime utilities for custom plugins |
| [YOAO-Plugins](https://github.com/YOAO-Labs/YOAO-Plugins) | Official plugin implementations maintained by YOAO-Labs |
| [YOAO-Docs](https://github.com/YOAO-Labs/YOAO-Docs) | Official user guides, plugin development documentation, API references, and deployment guides |

## License

First-party SDK source in this repository is licensed under the [MIT License](LICENSE). Third-party dependencies and plugin model assets retain their own licenses and terms.
