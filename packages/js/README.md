# `@yoao/plugin-sdk`

English | [简体中文](README.zh-CN.md)

Zero-dependency Node.js (`>=20`) and TypeScript (ESM + `.d.ts`) SDK for YOAO plugins. See the [Plugin-SDK repository README](../../README.md) for workspace overview and related repositories.

## Capabilities

- **Manifest validation**: `validateManifest`, `EXTENSION_MANIFEST_SCHEMA`, and `yoao-inference/v2` constants (`INFERENCE_PROTOCOL_VERSION`, `INFERENCE_WIRE_PROTOCOL_VERSION`, `INFERENCE_EXECUTION_MODES`, `INFERENCE_TASK_VALUES`, `INFERENCE_CAPABILITIES`). Validates `yoao-extension-manifest/v1` offline across `inference` (`container` and `direct` modes), `trainer`, `notification_channel`, `format_adapter`, and `theme`.
- **Declarative theme validation**: `validateThemeTokens` enforces `--sc-*` semantic CSS tokens for `light` and `dark` modes and rejects `url(...)`, `@import`, `javascript:`, CSS injection constructs, and unsafe values without executing third-party JavaScript.
- **Trainer contract validation**: `TRAINER_PROTOCOL_VERSION`, `SHA256_PATTERN`, preview limit constants (`PREVIEW_*`), `canonicalJsonDigest`, and `yoao-trainer/v1` validators (`validateServingCompatibility`, `validateTrainerCapabilities`, `validateConfigResult`, `validateTrainSpec`, `validateTrainEvent`, `validateTrainReceipt`, `validatePreviewSpec`, `validatePreviewReport`).
- **Auth helpers**: `verifyBearerToken` (constant-time Bearer token check via `crypto.timingSafeEqual`) and `readSecret` (direct environment variable or `*_FILE` secret loading).
- **CLI (`yoao-plugin-js`)**: validates plugin manifests and declarative theme token files from the command line.

## Usage

### Programmatic API (ESM and TypeScript)

```js
import { validateManifest, validateThemeTokens } from "@yoao/plugin-sdk";

const manifest = validateManifest({
  manifest_schema: "yoao-extension-manifest/v1",
  extension_point: "theme",
  plugin_id: "custom.theme.midnight",
  version: "0.1.0",
  publisher: "Custom Publisher",
  license: "MIT",
  protocol_version: "yoao-theme/v1",
  capabilities: ["dark"],
  extension: { theme_id: "midnight", tokens_file: "tokens.json" },
});

const tokens = validateThemeTokens({
  dark: { "--sc-bg": "#0b1120", "--sc-fg": "#f8fafc" },
});
```

### CLI

When installed as a package dependency:

```sh
yoao-plugin-js validate ./manifest.json
yoao-plugin-js validate-theme ./tokens.json
```

Or run directly from a local repository checkout (from `packages/js`):

```sh
node src/cli.mjs validate ./manifest.json
node src/cli.mjs validate-theme ./tokens.json
```

## Testing

From `packages/js`:

```sh
node --test test/sdk.test.mjs
```

From the repository root:

```sh
node --test packages/js/test/sdk.test.mjs
```

## License

First-party source in this package is licensed under the [MIT License](LICENSE). Third-party dependencies and plugin assets retain their own licenses and terms.
