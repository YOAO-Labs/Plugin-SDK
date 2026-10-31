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

Python 3.11 or newer is needed for all three packages together. The platform is the authoritative contract source; its five portable wheels are mirrored for anonymous distribution in this SDK repository's immutable [`contracts-906628f2619b` release](https://github.com/YOAO-Labs/Plugin-SDK/releases/tag/contracts-906628f2619b). The fetcher pins the manifest SHA-256 `5c26a3829073324bf6f9cc29521457d296920424e5b1b3c34d672408aa956663`, checks platform source revision `906628f2619be01ec8dd96f03385218e3626b64c`, and verifies every wheel against that manifest. It needs no platform checkout or GitHub credentials.

Managed checkpoint compatibility requires an explicit `format`: `null` identifies a native checkpoint without a separate export-format identity; named formats use nonempty strings. Null is not a wildcard, and omitting the field is invalid. The platform still matches the approved manifest, observed capabilities, checkpoint receipt and runtime device.

Plugins describe the required assets for each model and complete variant. The platform independently verifies files in the installed container and governs installation-scoped approval, availability and project access. Publishing a descriptor or displaying a catalog entry grants none of those permissions; user-trained checkpoints remain under platform Learning and Assets governance.

Native Torch lifecycle reports describe their model pools without an empty `session_counts` field. ONNX Runtime plugins additionally record the actual per-pool session counts and CUDA provider observations; the runtime checks those against the complete report rather than treating a missing session as qualified.

Full HTTP implementations can reuse `aap_backend_runtime.qualification_http` with the runtime's
`qualification` extra for real loopback route-admission checks. The caller owns the dedicated app
lifespan, credentials and actual model/lifecycle scenarios; the helper does not qualify an artifact
or authorize access to a live platform. It supports Python 3.10 without the author SDK.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install build 'setuptools>=68' wheel pytest pytest-asyncio
python scripts/fetch-contracts.py /tmp/aap-contracts
python -m pip install /tmp/aap-contracts/*.whl
python -m pip install -e 'packages/inference/runtime[test]' -e packages/inference/sdk -e packages/inference/mask-utils
python -m pytest packages/inference/runtime/tests
python -m pytest packages/inference/sdk/tests
python -m pytest packages/inference/mask-utils/tests
```

For author commands and their arguments, use `aap-inference-plugin --help`, `aap-inference-plugin build --help`, and `aap-inference-plugin check --help`. `aap-inference-plugin build /path/to/your-plugin` builds a wheel. HTTP checks require a running service and fixture data; GPU qualification requires explicit exclusive hardware and evidence. Neither a successful wheel build nor an HTTP check grants platform-managed qualification or permission to redistribute models. See the inference SDK package documentation for author workflow and provider requirements.

## Stateful tracker sessions

`TrackerSessionManager` keeps idle tracker state owned by the exact `ManagedLruPool` entry. Its `start()` borrow factory must yield `(managed_lease, callback_resource)`: the first value is the actual `ResourceLease` returned by that pool, while the second is the resource or adapter passed to `open_session`, `call` and `close_session` callbacks. Do not replace the first value with a model-only wrapper.

Opening and each command borrow and serialize the entry only while work runs. A completed command leaves no active borrower, but its idle session remains pool-owned. Continuations reacquire that same entry without rebuilding by key. Eviction, idle unload, drain or shutdown invalidate its context and await state cleanup before disposing the model; a stale token raises `TrackerSessionLost`, including after the same model key is loaded again.

Close callbacks must release their real state and surface failures. Cleanup runs exactly once; a raised error or the resource's `cleanup_uncertain` flag retains owned roots and unknown residency instead of claiming a safe unload. Cancellation waits for already-owned work to settle. Fence new admission before shutting down the session manager, and keep each workload's signed admission and completion outside the pool adapter: a persistent context does not authorize another request or make a consumed grant reusable.

## Building source release inputs

The initial SDK extraction is SDK commit `182d20dd0d84abe864d44fdb29d89dcf8de375a2`, extracted from the shared packages in YOAO-Plugins; its own repository history is the provenance for subsequent SDK changes. Platform portable contracts have a separate provenance above. Build only from an explicit SDK commit; uncommitted working-tree changes are not included:

```sh
python -m pip install build 'setuptools>=68' wheel
python scripts/build-sdk-artifacts.py --revision HEAD --output /tmp/aap-sdk-release
(cd /tmp/aap-sdk-release && sha256sum -c SHA256SUMS)
```

The output contains a source tar archive, wheel and sdist for each of the three packages, `manifest.json` with exact source repository/revision, builder versions and artifact SHA-256 hashes, and `SHA256SUMS` (including the manifest). The source archive is restricted to the three SDK package trees and root license/readme. It does not distribute platform source, official plugin implementations, model catalogs, native loaders or model weights. The repository owns publication separately; a local build does not publish anything.
