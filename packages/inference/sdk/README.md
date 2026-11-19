# Inference plugin SDK

This package is the detection authoring API and HTTP adapter, not an official plugin, model catalog, training SDK, or general-purpose plugin API. Implement `DetectionProvider.load(model_id, device, config)` and return a `DetectionModel` with `classes`, `predict(image, parameters)`, and `close()`. Return `Detection` boxes in original-image pixel coordinates with a class index and confidence. `create_detection_app` handles protocol envelopes, authentication, bounded image inputs, pooling, and lifecycle; configure it with `DetectionConfig`. Provider code owns model loading, inference, and cleanup. CPU providers do not need Torch, GPU drivers, or weights.

## Runnable CPU example

Prepare a complete wheelhouse for your Python version and a trusted SHA-256 requirements lock for the SDK, runtime, contracts and third-party wheels. Follow the [repository instructions](../../../README.md#portable-contracts-and-local-development) from the same SDK revision. Include `build`, `setuptools>=68` and `wheel` in the lock so the install below puts them in the active environment: `yoao-plugin build` does not use build isolation, and the SDK's `dev` extra installs only `build`. The generated project contains no release URL or editable checkout path.

```sh
python -m pip install --no-index --find-links="$WHEELHOUSE" --require-hashes -r "$REQUIREMENTS_LOCK"
yoao-plugin init my-provider
yoao-plugin build my-provider
python -m pip install --no-deps my-provider/dist/*.whl
umask 077
python -c 'import secrets; from pathlib import Path; Path("token").write_text(secrets.token_urlsafe(40))'
yoao-plugin dev --entrypoint my-color-region-plugin --device cpu --token-file token
```

The generated editable `my_color_region_plugin.py` is a color-region **algorithm demo**, not a pretrained model. For each RGB channel, it finds pixels with that channel >= 200 and the other channels <= 80. It returns one enclosing box per present color (even if its regions are disconnected), classes `red-region`, `green-region`, `blue-region`, score 1.0, or no detections. The exclusive right/bottom coordinates are image pixels. It ignores inference parameters; replace the model/provider and rename the distribution, module and console script before publishing your own plugin. It has no assets and supports only `cpu`. Authenticate non-health requests with `Authorization: Bearer <token>`; `/predict` accepts bounded data-image URLs. Host URLs require an explicit `DetectionConfig.allowed_image_hosts` allowlist.

`dev` starts the installed provider on loopback, warms its default detection variant, and verifies residency on the explicitly requested device. The provider implements `serve --device --token-file --host --port`; extra provider arguments follow `--` and cannot override those values. For example an asset-backed provider may need `-- --asset-dir /approved/assets --local`. Inherited platform route/plugin authority is removed; a private temporary data directory is provided. Ctrl-C stops the child. This is development preflight, not managed GPU qualification. Direct provider serving beyond loopback requires appropriate network and secret controls.

## Build and optional checks

In another shell using the same environment, run `AAP_PLUGIN_TOKEN="$(cat token)" yoao-plugin check --url http://127.0.0.1:8000 --fixtures fixtures.json`. `AAP_PLUGIN_TOKEN` is the token the installed checker reads for this provider, not a platform API key. The fixture JSON requires `multi` with at least two expected detections and `empty` with none; each case has `image`, `detections`, `box_tolerance`, and `score_tolerance`. A detection supplies `native_class_index`, `label`, percentage `[x,y,width,height]` `box`, and `score`. For the demo, use an independently specified red/green image and a black image. Expected values must come from the algorithm or a separately reviewed model baseline, not the HTTP response under test.

GPU qualification is optional and absent from the CPU example. `check --qualify-local` requires explicit `--entrypoint`, `--device cuda:N`, `--gpu-uuid`, `--asset-dir`, `--fixtures`, `--exclusive-gpu`, and `--report`. The provider implements `probe` and `qualification-server` hooks called in `qualification.py`: missing/wrong-device weights, residency, multi/empty inference, release and signed lifecycle checks. These hooks are only for providers with applicable GPU/weight semantics. Local-isolated evidence does not grant platform-managed qualification.

`yoao-plugin build DIRECTORY` builds a wheel using the installed build tools. To build an image, supply a digest-pinned `--base-image`, `--wheelhouse`, `--requirements-lock`, `--entrypoint` (the installed provider console script) and `--tag`; `--push` is optional. Only the pip install layer runs with networking disabled: the wheelhouse and hash lock must cover every Python dependency, including the provider. The separate APT layer uses the base image's configured repositories for `libgl1` and `libglib2.0-0`; the full image build is not guaranteed offline. The base must support Debian package and user management. The image runs as UID/GID 10001 with writable `AAP_PLUGIN_DATA_DIR=/var/lib/aap-inference`. Supply the provider's serve arguments, including its token file, at runtime.

