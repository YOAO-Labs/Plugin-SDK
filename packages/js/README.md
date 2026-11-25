# `@yoao/plugin-sdk`

零第三方依赖的 Node.js (`>=20`) / TypeScript 插件开发包，提供：

- `validateManifest`：`yoao-extension-manifest/v1` 清单离线校验（支持 `inference`、`notification_channel`、`format_adapter`、`theme`）；
- `validateThemeTokens`：纯声明式主题插件（`--sc-*` 语义 CSS Token）安全校验，拒绝 `url(...)`、CSS 注入与非法色值，不执行第三方 JS；
- `verifyBearerToken` / `readSecret`：恒定时间 Bearer 鉴权与 `*_FILE` 只读凭据读取；
- `yoao-plugin-js` CLI：`yoao-plugin-js validate manifest.json` 与 `yoao-plugin-js validate-theme tokens.json`。

## 测试

```bash
node --test test/sdk.test.mjs
```
