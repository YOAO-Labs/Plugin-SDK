# `yoao-plugin-sdk`

[English](README.md) | 简体中文

YOAO 统一 Python 插件开发包（`requires-python >= 3.10`），覆盖全部五个扩展点（`inference`、`trainer`、`notification_channel`、`format_adapter`、`theme`）。仓库整体架构与相关仓库关系请参阅 [Plugin-SDK 仓库根目录说明](../../README.zh-CN.md)。

## 核心能力

### 基础包内置能力（仅依赖 `pydantic>=2.5`）

- **清单校验**（`yoao_plugin_sdk.manifest`，并在 `yoao_plugin_sdk` 顶层导出）：提供 `ExtensionManifestV1`、`load_manifest_file`、`validate_manifest_dict`，严格校验 `yoao-extension-manifest/v1` 清单（包含 `container` 与 `direct` 推理模式）。
- **鉴权辅助**（`yoao_plugin_sdk.auth`）：提供恒定时间 Bearer 鉴权 `verify_bearer_token` 与支持直传环境变量或 `*_FILE` 的只读凭据读取 `read_secret`。
- **各扩展点协议模型**：
  - `yoao_plugin_sdk.inference`：`yoao-inference/v2` 清单模型与能力常量。
  - `yoao_plugin_sdk.trainer`：`yoao-trainer/v1` 契约模型（`TrainerCapabilitiesV1`、`ValidateConfigResultV1`、`TrainSpecV1`、`TrainEventV1`、`TrainReceiptV1`、`PreviewSpecV1`、`PreviewReportV1` 等）、原子文件与不可变产物工具（`digest_file`、`atomic_write_bytes`、`atomic_write_json`、`immutable_copy`、`sync_directory`）、结构化异常与 OOM 分类（`TrainerError`、`InputMismatchError`、`NativeConfigurationMismatch`、`NativeRuntimeMismatch`、`DatasetContractError`、`classify_failure`、`sanitize_failure_message`）、训练运行时辅助（`EventEmitter`、`CheckpointSlotStore`、`SUPERVISOR_OWNED_ARTIFACTS`、`build_artifact_entry`、`load_dataset_identity`、`run_with_signal_receipt`）、冻结输入成员证明器（`InputMembership`、`POLICY_DIGEST`）、纯标准库数据集与谱系校验（`validate_source_lineage`、`validate_frozen_manifest_core`、`validate_sample_files`、`validate_yolo_bbox_label`）、严格参数与指标工具（`validate_value`、`normalized`、`metrics_of`、`hash_source_files`），以及 `TrainerPlugin` CLI 驱动（`run_trainer_cli`、`build_trainer_parser`）。
  - `yoao_plugin_sdk.notification`：`yoao-notification-channel/v1` 模型（`DeliveryEnvelope`、`DeliverRequest`、`DeliverResult`）。
  - `yoao_plugin_sdk.format_adapter`：`yoao-format-adapter/v1` 模型（`ParsedAnnotation`、`ParsedItem`、`ImportParseResponse`）。
- **基础 CLI**（`yoao-plugin`）：`version`、`init`（支持 `--type trainer --example`）、`validate` 与 `trainer check`。

### 可选依赖（extras）

- `[inference-runtime]`（`yoao_plugin_sdk.inference.runtime`）：`ManagedLruPool`、`TrackerSessionManager`、`ManagedGpuLifecycle`、设备与 GPU 工具、生命周期证据辅助及 `plugin_service` 路由与鉴权集成。
- `[inference]`（`yoao_plugin_sdk.inference`、`yoao_plugin_sdk.inference.authoring`）：框架无关的检测插件开发接口（`Detection`、`DetectionConfig`、`DetectionModel`、`DetectionProvider`、`create_detection_app`、`load_manifest`）以及 `yoao-plugin` 的 `dev`、`check`、`build` 命令。
- `[mask]`（`yoao_plugin_sdk.inference.mask_utils`）：掩码转多边形（`mask_to_polygon`、`mask_to_multi_polygon`、`mask_to_preview_polygon`）、坐标归一化（`normalize_coords`）、COCO RLE 编解码（`encode_coco_rle`、`decode_coco_rle`、`validate_coco_rle`）及提示适配器。
- `[qualification]`（`yoao_plugin_sdk.inference.runtime.qualification_http`）：用于本地准入验证的环回 HTTP 客户端与路由准入辅助。
- `[notification]`：附加 `httpx` 依赖，用于通知渠道 HTTP 服务。
- `[trainer]` 与 `[format]`：扩展点 extra，当前无基础包之外的额外运行时依赖。

`[inference-runtime]` 与 `[inference]` 依赖 [YOAO](https://github.com/YOAO-Labs/YOAO) 提供的 `yoao-sdk[protocol]`；其余 extra 均可独立使用，无需安装 `yoao-sdk`。`yoao-plugin-sdk` 不会锁定 PyTorch 或 TorchVision 版本。

## 安装

在本地源码目录（`packages/python`）中以可编辑模式安装：

```sh
# 基础包（仅依赖 pydantic>=2.5）
python -m pip install -e .

# 按需安装可选依赖
python -m pip install -e ".[mask]"
python -m pip install -e ".[notification]"
python -m pip install -e ".[dev]"
```

从已发布的 wheel 或本地 wheelhouse 安装（使用推理 extra 时需同时提供 `yoao-sdk[protocol]`）：

```sh
python -m pip install yoao-plugin-sdk
python -m pip install "yoao-plugin-sdk[inference-runtime]"
python -m pip install "yoao-plugin-sdk[inference]"
python -m pip install "yoao-plugin-sdk[inference-runtime,mask]"
```

## 快速开始

```sh
yoao-plugin version
yoao-plugin init --type trainer --example ./my-trainer-plugin
yoao-plugin validate ./my-trainer-plugin/manifest.json
python -m pip install --no-deps --no-build-isolation ./my-trainer-plugin
yoao-plugin trainer check --manifest ./my-trainer-plugin/manifest.json --entrypoint my-trainer-plugin --preview

# 生成 CPU 颜色区域检测示例（运行示例需要安装 [inference]）
yoao-plugin init --type inference --example ./my-detector-plugin
```

安装 `[inference]` 后可使用推理插件开发命令：

```sh
yoao-plugin dev --help
yoao-plugin check --help
yoao-plugin build --help
```

## 测试

在 `packages/python` 目录下运行：

```sh
# 核心与训练器 SDK 测试（基础包 + [dev]）
python -m pytest tests/test_sdk.py tests/trainer

# 掩码工具测试（[mask,dev]）
python -m pytest --import-mode=importlib tests/mask_utils

# 完整测试集（需安装 yoao-sdk[protocol] 与 [inference,mask,qualification,dev]）
python -m pytest --import-mode=importlib tests
```

## 许可证

本包第一方源码采用 [MIT License](LICENSE)。第三方依赖与插件资产保留各自原有的许可证与条款。
