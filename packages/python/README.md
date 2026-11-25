# `yoao-plugin-sdk`

YOAO 统一 Python 插件开发包（`requires-python >= 3.10`），面向全部扩展点：

- 通用清单模型 `ExtensionManifestV1`（`yoao-extension-manifest/v1`），支持 `inference`、`notification_channel`、`format_adapter`、`trainer`、`theme`；
- 恒定时间 Bearer 鉴权与 `*_FILE` 只读凭据读取（`yoao_plugin_sdk.auth`）；
- 推理插件全套能力（按需安装 extra）：
  - `yoao_plugin_sdk.inference.runtime`（`[inference-runtime]`）：模型池、有状态追踪会话、受控 GPU 生命周期与 `plugin_service` 路由/鉴权组合；
  - `yoao_plugin_sdk.inference` / `.authoring`（`[inference]`）：框架无关检测插件接口（`Detection`、`create_detection_app`）与 `yoao-plugin dev/check/build`；
  - `yoao_plugin_sdk.inference.mask_utils`（`[mask]`）：掩码转多边形、RLE 与交互式提示适配器；
- 训练器、通知渠道、格式适配器协议模型：`yoao_plugin_sdk.trainer`、`yoao_plugin_sdk.notification`、`yoao_plugin_sdk.format_adapter`；
- 统一 CLI：`yoao-plugin version`、`yoao-plugin validate`、`yoao-plugin init --type ...`、`yoao-plugin dev`、`yoao-plugin check`、`yoao-plugin build`。

## 安装

```bash
# 基础包（仅依赖 pydantic>=2.5）
pip install yoao-plugin-sdk

# 按扩展点选择可选依赖
pip install "yoao-plugin-sdk[inference-runtime]"
pip install "yoao-plugin-sdk[inference]"
pip install "yoao-plugin-sdk[inference-runtime,mask]"
pip install "yoao-plugin-sdk[trainer]"
pip install "yoao-plugin-sdk[notification]"
```

## 快速开始

```bash
yoao-plugin init --type trainer ./my-trainer-plugin
yoao-plugin validate ./my-trainer-plugin/manifest.json

# 生成可运行的 CPU 颜色区域检测示例（需要 [inference]）
yoao-plugin init --type inference --example ./my-detector-plugin
```
