# Inference plugin SDK

This package is the detection authoring API and HTTP adapter, not an official plugin, model catalog, training SDK, or general-purpose plugin API. Implement `DetectionProvider.load(model_id, device, config)` and return a `DetectionModel` with `classes`, `predict(image, parameters)`, and `close()`. Return `Detection` boxes in original-image pixel coordinates with a class index and confidence. `create_detection_app` handles protocol envelopes, authentication, bounded image inputs, pooling, and lifecycle; configure it with `DetectionConfig`. Provider code owns model loading, inference, and cleanup. CPU providers do not need Torch, GPU drivers, or weights.

## Runnable CPU example

Install the SDK and runtime from the Plugin-SDK `python-sdk-0.1.0` release and the five portable contracts from the `contracts-3d6393e64630` release. The generated project's `[tool.uv.sources]` pins these release URLs for `uv`; ordinary pip does not consume that table, so install those wheels explicitly before installing the generated wheel. Publication of the SDK release is managed separately.

```sh
aap-inference-plugin init my-provider
aap-inference-plugin build my-provider
# Install my-provider/dist/*.whl into the SDK environment.
printf '%s' 'replace-with-a-secret-at-least-32-characters' > token
my-color-region-plugin serve --token-file token --host 127.0.0.1 --port 8000
```

The generated editable `my_color_region_plugin.py` is a color-region **algorithm demo**, not a pretrained model. It finds pixels where R >= 200 and G and B <= 80, returns one bounding box enclosing all matching pixels (even disconnected regions), class `red-region`, score 1.0, or no detections if none match. The exclusive right/bottom box coordinates are in image pixels. It ignores inference parameters; replace the model and provider to implement your own semantics. It has no model assets and only supports `cpu`. Authenticate non-health requests with `Authorization: Bearer <token>`; `/predict` accepts protocol tasks with data-image URLs. Host URLs require an explicit `DetectionConfig.allowed_image_hosts` allowlist. Expose the service beyond loopback only with appropriate network and secret controls.

## Build and optional checks

`aap-inference-plugin check --url URL --fixtures FILE` checks a running provider, using `AAP_PLUGIN_TOKEN`. The JSON fixture schema in `check.py` requires a multi-detection case with at least two detections and an empty case, so it is not applicable to the single-box demo without changing the algorithm. GPU qualification is optional and is not implemented by the CPU example. `check --qualify-local` requires an explicit `--entrypoint`, `--device cuda:N`, `--gpu-uuid`, `--asset-dir`, `--fixtures`, `--exclusive-gpu`, and `--report`. That provider must implement the `probe` and `qualification-server` command hooks called in `qualification.py`: the probe reports missing weights, wrong device, GPU residency, multi/empty inference and GPU release; the qualification server supports the signed lifecycle exercised there. These hooks are for providers with applicable GPU/weight semantics, not a requirement for detection authors. The report is local-isolated evidence, not platform-managed qualification.

`aap-inference-plugin build DIRECTORY` builds a wheel. For a container, provide `--base-image` pinned by SHA-256, `--wheelhouse`, `--requirements-lock`, `--entrypoint` (the installed provider console script), and `--tag`; `--push` is optional. The wheelhouse must contain the complete binary dependency closure and the lock must pin and hash every dependency including the provider. Installation is offline, and the image runs as UID/GID 10001 with writable `AAP_PLUGIN_DATA_DIR=/var/lib/aap-inference`. Supply the provider's serve arguments at container runtime, including its token file. The base image must support the builder's Debian user-management commands.

