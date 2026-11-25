# YOAO Plugin SDK

YOAO stands for **You Only Annotate Once**. This repository maintains multi-language SDKs and tooling for authors of independent YOAO plugins across all extension points (inference, notification channels, format adapters, and declarative themes). `yoao plugins` belongs to the separate [platform API client](https://github.com/YOAO-Labs/ai-annotation-platform) and manages installations. The platform owns the portable contracts, authorization and storage. [YOAO-Plugins](https://github.com/YOAO-Labs/YOAO-Plugins) maintains official plugin implementations.

| Path | Distribution | Runtime | Purpose |
| --- | --- | --- | --- |
| `packages/python` | `yoao-plugin-sdk` | Python `>=3.10` | Universal `yoao-extension-manifest/v1` validation, auth helpers, notification/format/inference/theme protocol models, and `yoao-plugin-sdk` CLI |
| `packages/js` | `@yoao/plugin-sdk` | Node.js `>=20` / TS | Zero-dependency ESM + `.d.ts` manifest validation, declarative `--sc-*` theme token validation, auth helpers, and `yoao-plugin-js` CLI |
| `packages/inference/sdk` | `yoao-inference-plugin-sdk` | Python `>=3.11` | Detection author interfaces, HTTP adapter and author tooling |
| `packages/inference/runtime` | `yoao-backend-runtime` | Python `>=3.10` | Inference execution, cancellation and managed lifecycle helpers |
| `packages/inference/mask-utils` | `yoao-mask-utils` | Python `>=3.10` | Mask conversion helpers |

UI, trainers and data export may gain separate capability packages when implemented; they do not inherit detection, HTTP or GPU APIs merely to be plugins. First-party SDK code is [MIT](LICENSE); third-party dependencies and assets retain their own terms.

Inference code is grouped under one capability directory, `packages/inference/`. Its three installable distributions remain separate: runtime and mask consumers support Python 3.10 and must not inherit the author SDK's Python 3.11 requirement or development-service dependencies.

## Portable contracts and local development

Python 3.11 or newer is needed for all three packages together. The platform owns the portable contract source; obtain a manifest and wheels built from the desired platform revision, and the independently verified manifest SHA-256. The fetcher verifies the manifest and each selected wheel. The previous `contracts-906628f2619b` release is historical and does not provide the renamed distributions. CI requires repository variables `CONTRACT_MANIFEST_URL` and `CONTRACT_MANIFEST_SHA256` pointing to matching artifacts; until supplied, contract-dependent jobs fail rather than assume a release exists.

Managed checkpoint compatibility requires an explicit `format`: `null` identifies a native checkpoint without a separate export-format identity; named formats use nonempty strings. Null is not a wildcard, and omitting the field is invalid. The platform still matches the approved manifest, observed capabilities, checkpoint receipt and runtime device.

Plugins describe the required assets for each model and complete variant. The platform independently verifies files in the installed container and governs installation-scoped approval, availability and project access. Publishing a descriptor or displaying a catalog entry grants none of those permissions; user-trained checkpoints remain under platform Learning and Assets governance.

Native Torch lifecycle reports describe their model pools without an empty `session_counts` field. ONNX Runtime plugins additionally record the actual per-pool session counts and CUDA provider observations; the runtime checks those against the complete report rather than treating a missing session as qualified.

Full HTTP implementations can reuse `yoao_backend_runtime.qualification_http` with the runtime's
`qualification` extra for real loopback route-admission checks. The caller owns the dedicated app
lifespan, credentials and actual model/lifecycle scenarios; the helper does not qualify an artifact
or authorize access to a live platform. It supports Python 3.10 without the author SDK.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install build 'setuptools>=68' wheel pytest pytest-asyncio PyJWT uvicorn
python scripts/fetch-contracts.py --manifest "$CONTRACT_MANIFEST" --sha256 "$CONTRACT_MANIFEST_SHA256" --output /tmp/yoao-contracts
python scripts/build-sdk-artifacts.py --revision HEAD --output /tmp/yoao-sdk
python -m pip install /tmp/yoao-contracts/*.whl /tmp/yoao-sdk/*.whl
python -m pytest packages/inference/runtime/tests
python -m pytest packages/inference/sdk/tests
python -m pytest packages/inference/mask-utils/tests
```

For the [detection author workflow](packages/inference/sdk/README.md), use the documentation from the `source_revision` in your verified SDK manifest and `yoao-plugin --help` from the installed wheel. The public historical `aap-inference-plugin` CLI and its templates are not substitutes for this candidate. A wheel build or HTTP check does not qualify managed GPU use or authorize model redistribution.

## Stateful tracker sessions

`TrackerSessionManager` keeps idle tracker state owned by the exact `ManagedLruPool` entry. Its `start()` borrow factory must yield `(managed_lease, callback_resource)`: the first value is the actual `ResourceLease` returned by that pool, while the second is the resource or adapter passed to `open_session`, `call` and `close_session` callbacks. Do not replace the first value with a model-only wrapper.

Opening and each command borrow and serialize the entry only while work runs. A completed command leaves no active borrower, but its idle session remains pool-owned. Continuations reacquire that same entry without rebuilding by key. Eviction, idle unload, drain or shutdown invalidate its context and await state cleanup before disposing the model; a stale token raises `TrackerSessionLost`, including after the same model key is loaded again.

Retirement releases model references before the allocator cleanup callback runs. A completed cleanup must release both live model allocations and allocator caches; an empty pool alone is not physical-memory evidence.

Close callbacks must release their real state and surface failures. Cleanup runs exactly once; a raised error or the resource's `cleanup_uncertain` flag retains owned roots and unknown residency instead of claiming a safe unload. Cancellation waits for already-owned work to settle. Fence new admission before shutting down the session manager, and keep each workload's signed admission and completion outside the pool adapter: a persistent context does not authorize another request or make a consumed grant reusable.

## Building source release inputs

The initial SDK extraction is SDK commit `182d20dd0d84abe864d44fdb29d89dcf8de375a2`, extracted from the shared packages in YOAO-Plugins; its own repository history is the provenance for subsequent SDK changes. Platform portable contracts have a separate provenance above. Build only from an explicit SDK commit; uncommitted working-tree changes are not included:

```sh
python -m pip install build 'setuptools>=68' wheel
python scripts/build-sdk-artifacts.py --revision HEAD --output /tmp/yoao-sdk-release
(cd /tmp/yoao-sdk-release && sha256sum -c SHA256SUMS)
```

The output contains a source tar archive, wheel and sdist for each of the three packages, `manifest.json` with exact source repository/revision, builder versions and artifact SHA-256 hashes, and `SHA256SUMS` (including the manifest). The source archive is restricted to the three SDK package trees and root license/readme. It does not distribute platform source, official plugin implementations, model catalogs, native loaders or model weights. The repository owns publication separately; a local build does not publish anything.
