# `yoao-plugin-sdk`

YOAO 通用 Python 插件开发包（`requires-python >= 3.10`），面向全部扩展点：

- 通用清单模型 `ExtensionManifestV1`（`yoao-extension-manifest/v1`），支持 `inference`、`notification_channel`、`format_adapter`、`theme`；
- 恒定时间 Bearer 鉴权与 `*_FILE` 凭据读取（`yoao_plugin_sdk.auth`）；
- 各扩展点协议常量与基类（`yoao_plugin_sdk.inference`、`yoao_plugin_sdk.notification`、`yoao_plugin_sdk.format_adapter`）；
- 脚手架与离线校验 CLI：`yoao-plugin init`、`yoao-plugin validate`、`yoao-plugin version`。

## 安装

```bash
pip install yoao-plugin-sdk
# 按扩展点选择可选依赖
pip install "yoao-plugin-sdk[inference]"
pip install "yoao-plugin-sdk[notification]"
```

## 快速开始

```bash
yoao-plugin init --type format_adapter ./my-format-plugin
yoao-plugin validate ./my-format-plugin/manifest.json
```
