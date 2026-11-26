# YOAO Plugin SDK

[English](README.md) | 简体中文

面向独立 YOAO（**You Only Annotate Once**）插件开发者的多语言插件 SDK、CLI 与共享运行时仓库。

本仓库基于统一的 `yoao-extension-manifest/v1` 清单规范，覆盖 YOAO 全部五个扩展点：

- `inference`：推理服务（`yoao-inference/v2`，支持 `container` 与 `direct` 模式）
- `trainer`：训练器（`yoao-trainer/v1`）
- `notification_channel`：通知渠道（`yoao-notification-channel/v1`）
- `format_adapter`：格式适配器（`yoao-format-adapter/v1`）
- `theme`：声明式主题（`yoao-theme/v1`）

## 包结构

| 路径 | 分发包 | 运行时要求 | 说明 |
| --- | --- | --- | --- |
| [`packages/python`](packages/python/README.zh-CN.md) | `yoao-plugin-sdk` | Python `>=3.10` | 清单校验、Bearer 与 `*_FILE` 鉴权辅助、各扩展点协议模型、`yoao-plugin` CLI，以及按需安装的推理运行时、检测开发接口与掩码工具 |
| [`packages/js`](packages/js/README.zh-CN.md) | `@yoao/plugin-sdk` | Node.js `>=20` / TypeScript | 零运行时依赖的 ESM 与 `.d.ts` 包，提供清单校验、声明式 `--sc-*` 主题校验、`yoao-trainer/v1` 契约校验、鉴权辅助与 `yoao-plugin-js` CLI |

## Python 可选依赖

Python 3.10、3.11 或 3.12 插件均安装统一的 `yoao-plugin-sdk` 分发包，并按需选择对应 extra：

| Extra | 模块命名空间 | 适用范围 |
| --- | --- | --- |
| 基础包（`yoao-plugin-sdk`） | `yoao_plugin_sdk.{manifest,auth,cli,inference,trainer,notification,format_adapter}` | 清单校验、鉴权辅助、`yoao-plugin` 基础命令与各扩展点协议模型（仅依赖 `pydantic>=2.5`） |
| `[inference-runtime]` | `yoao_plugin_sdk.inference.runtime` | 模型 LRU 池、有状态追踪会话、受管 GPU 生命周期与 `plugin_service` 路由及鉴权集成 |
| `[inference]` | `yoao_plugin_sdk.inference`、`yoao_plugin_sdk.inference.authoring` | 检测插件开发接口（`create_detection_app`）及 `yoao-plugin` 的 `dev`、`check`、`build` 命令 |
| `[mask]` | `yoao_plugin_sdk.inference.mask_utils` | 掩码转多边形、COCO RLE 编解码与交互式提示适配器 |
| `[qualification]` | `yoao_plugin_sdk.inference.runtime.qualification_http` | 本地环回路由准入校验辅助 |
| `[notification]` | `yoao_plugin_sdk.notification` | 附加 `httpx` 依赖，用于通知渠道 HTTP 服务 |
| `[trainer]` | `yoao_plugin_sdk.trainer` | 训练器扩展点 extra（当前无基础包之外的额外运行时依赖） |
| `[format]` | `yoao_plugin_sdk.format_adapter` | 格式适配器扩展点 extra（当前无基础包之外的额外运行时依赖） |

`yoao-plugin-sdk` 不会锁定 PyTorch 或 TorchVision 版本，每个插件的虚拟环境或容器镜像自行管理其机器学习框架与 CUDA 栈。

基础包 `yoao-plugin-sdk`、`[mask]`、`[notification]`、`[trainer]`、`[format]` 以及 `@yoao/plugin-sdk` 均可独立使用，不依赖平台契约包；仅 `[inference-runtime]` 和 `[inference]` 需要依赖由 [YOAO](https://github.com/YOAO-Labs/YOAO) 提供的便携契约包 `yoao-sdk[protocol]`。

## 快速上手

### Python SDK

在本地仓库中以可编辑模式安装基础 Python SDK，生成插件模板并运行核心测试：

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e "packages/python[dev]"
yoao-plugin version
yoao-plugin init --type trainer ./my-trainer-plugin
yoao-plugin validate ./my-trainer-plugin/manifest.json
python -m pytest packages/python/tests/test_sdk.py scripts/test_fetch_contracts.py
```

如需测试掩码工具 extra，安装 `packages/python[mask,dev]` 后运行 `python -m pytest --import-mode=importlib packages/python/tests/mask_utils`。更多模块说明、CLI 子命令及推理 extra 用法请参阅 [`packages/python/README.zh-CN.md`](packages/python/README.zh-CN.md)。

### JavaScript / TypeScript SDK

使用 Node.js `>=20` 可直接运行 `@yoao/plugin-sdk` 测试与 CLI（无需安装第三方依赖）：

```sh
node --test packages/js/test/sdk.test.mjs
node packages/js/src/cli.mjs validate ./my-trainer-plugin/manifest.json
```

完整导出列表与 CLI 说明请参阅 [`packages/js/README.zh-CN.md`](packages/js/README.zh-CN.md)。

## 相关仓库

| 仓库 | 定位 |
| --- | --- |
| [YOAO](https://github.com/YOAO-Labs/YOAO) | 核心标注平台，负责 `yoao plugins` 安装管理、便携契约源码（`yoao-sdk`）、权限授权、存储与资产治理 |
| [Plugin-SDK](https://github.com/YOAO-Labs/Plugin-SDK) | 本仓库，为自定义插件开发者提供多语言 SDK、CLI 与共享运行时库 |
| [YOAO-Plugins](https://github.com/YOAO-Labs/YOAO-Plugins) | YOAO-Labs 维护的官方插件实现 |
| [YOAO-Docs](https://github.com/YOAO-Labs/YOAO-Docs) | 官方文档站，包含用户指南、插件开发文档、API 参考与部署指南 |

## 许可证

本仓库第一方 SDK 源码采用 [MIT License](LICENSE)。第三方依赖与插件模型资产保留各自原有的许可证与条款。
