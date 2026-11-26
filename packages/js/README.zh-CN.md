# `@yoao/plugin-sdk`

[English](README.md) | 简体中文

零运行时依赖的 Node.js（`>=20`）与 TypeScript（ESM + `.d.ts`）YOAO 插件开发包。仓库整体架构与相关仓库关系请参阅 [Plugin-SDK 仓库根目录说明](../../README.zh-CN.md)。

## 核心能力

- **清单校验**：导出 `validateManifest`、`EXTENSION_MANIFEST_SCHEMA` 及 `yoao-inference/v2` 常量（`INFERENCE_PROTOCOL_VERSION`、`INFERENCE_WIRE_PROTOCOL_VERSION`、`INFERENCE_EXECUTION_MODES`、`INFERENCE_TASK_VALUES`、`INFERENCE_CAPABILITIES`），支持离线校验 `inference`（含 `container` 与 `direct` 模式）、`trainer`、`notification_channel`、`format_adapter`、`theme` 五类 `yoao-extension-manifest/v1` 清单。
- **声明式主题校验**：导出 `validateThemeTokens`，校验 `light` 与 `dark` 模式下的 `--sc-*` 语义 CSS Token，拒绝 `url(...)`、`@import`、`javascript:` 等 CSS 注入与非法值，不执行第三方 JavaScript。
- **训练器契约校验**：导出 `TRAINER_PROTOCOL_VERSION`、`SHA256_PATTERN`、预览限额常量（`PREVIEW_*`）、`canonicalJsonDigest` 及 `yoao-trainer/v1` 校验函数（`validateServingCompatibility`、`validateTrainerCapabilities`、`validateConfigResult`、`validateTrainSpec`、`validateTrainEvent`、`validateTrainReceipt`、`validatePreviewSpec`、`validatePreviewReport`）。
- **鉴权辅助**：导出 `verifyBearerToken`（基于 `crypto.timingSafeEqual` 的恒定时间 Bearer 鉴权）与 `readSecret`（支持直传环境变量或 `*_FILE` 只读凭据读取）。
- **CLI（`yoao-plugin-js`）**：提供命令行清单校验与声明式主题 Token 校验。

## 使用方法

### ESM / TypeScript 导入

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

作为依赖包安装后：

```sh
yoao-plugin-js validate ./manifest.json
yoao-plugin-js validate-theme ./tokens.json
```

或在本地源码目录（`packages/js`）中直接运行：

```sh
node src/cli.mjs validate ./manifest.json
node src/cli.mjs validate-theme ./tokens.json
```

## 测试

在 `packages/js` 目录下运行：

```sh
node --test test/sdk.test.mjs
```

或在仓库根目录运行：

```sh
node --test packages/js/test/sdk.test.mjs
```

## 许可证

本包第一方源码采用 [MIT License](LICENSE)。第三方依赖与插件资产保留各自原有的许可证与条款。
