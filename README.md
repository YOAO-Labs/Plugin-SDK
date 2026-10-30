# Plugin SDK

This repository publishes capability-scoped SDKs for independent plugin authors; it is not the platform or a distribution of official plugins. [YOAO-Plugins](https://github.com/YOAO-Labs/YOAO-Plugins) owns official implementations and their model-specific assets. The [platform](https://github.com/YOAO-Labs/ai-annotation-platform) owns API SDKs, portable contract **source**, authorization, storage and governance; the SDK consumes published contracts rather than duplicating platform rules.

| Path | Python distribution | Purpose |
| --- | --- | --- |
| `packages/inference/sdk` | `aap-inference-plugin-sdk` | Detection author interfaces, HTTP adapter and author tooling |
| `packages/inference/runtime` | `aap-backend-runtime` | Inference execution, cancellation and managed lifecycle helpers |
| `packages/inference/mask-utils` | `anno-mask-utils` | Mask conversion helpers |

UI, trainers and data export may gain separate capability packages when implemented; they do not inherit detection, HTTP or GPU APIs merely to be plugins. First-party SDK code is [MIT](LICENSE); third-party dependencies and assets retain their own terms.

Inference code is grouped under one capability directory, `packages/inference/`. Its three installable distributions remain separate: runtime and mask consumers support Python 3.10 and must not inherit the author SDK's Python 3.11 requirement or development-service dependencies.

## Portable contracts and local development

Python 3.11 or newer is needed for all three packages together. The platform is the authoritative contract source; its five portable wheels are mirrored for anonymous distribution in this SDK repository's immutable [`contracts-3d6393e64630` release](https://github.com/YOAO-Labs/Plugin-SDK/releases/tag/contracts-3d6393e64630). The fetcher pins the manifest SHA-256 `d3749d2a7429da60c02e222c65e9c4a5ec545b050b797422263d5a3bcb02ddec`, checks platform source revision `3d6393e64630fdeb6ff699b13c2d1f9e198a1a06`, and verifies every wheel against that manifest. It needs no platform checkout or GitHub credentials.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install build 'setuptools>=68' wheel pytest pytest-asyncio
python scripts/fetch-contracts.py /tmp/aap-contracts
python -m pip install /tmp/aap-contracts/*.whl
python -m pip install -e packages/inference/runtime -e packages/inference/sdk -e packages/inference/mask-utils
python -m pytest packages/inference/runtime/tests
python -m pytest packages/inference/sdk/tests
python -m pytest packages/inference/mask-utils/tests
```

For author commands and their arguments, use `aap-inference-plugin --help`, `aap-inference-plugin build --help`, and `aap-inference-plugin check --help`. `aap-inference-plugin build /path/to/your-plugin` builds a wheel. HTTP checks require a running service and fixture data; GPU qualification requires explicit exclusive hardware and evidence. Neither a successful wheel build nor an HTTP check grants platform-managed qualification or permission to redistribute models. See the inference SDK package documentation for author workflow and provider requirements.

## Building source release inputs

The initial SDK extraction is SDK commit `182d20dd0d84abe864d44fdb29d89dcf8de375a2`, extracted from the shared packages in YOAO-Plugins; its own repository history is the provenance for subsequent SDK changes. Platform portable contracts have a separate provenance above. Build only from an explicit SDK commit; uncommitted working-tree changes are not included:

```sh
python -m pip install build 'setuptools>=68' wheel
python scripts/build-sdk-artifacts.py --revision HEAD --output /tmp/aap-sdk-release
(cd /tmp/aap-sdk-release && sha256sum -c SHA256SUMS)
```

The output contains a source tar archive, wheel and sdist for each of the three packages, `manifest.json` with exact source repository/revision, builder versions and artifact SHA-256 hashes, and `SHA256SUMS` (including the manifest). The source archive is restricted to the three SDK package trees and root license/readme. It does not distribute platform source, official plugin implementations, model catalogs, native loaders or model weights. The repository owns publication separately; a local build does not publish anything.
