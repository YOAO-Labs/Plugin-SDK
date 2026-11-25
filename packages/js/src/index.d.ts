export const EXTENSION_MANIFEST_SCHEMA: "yoao-extension-manifest/v1";
export const INFERENCE_PROTOCOL_VERSION: "yoao-inference/v2";
export const INFERENCE_WIRE_PROTOCOL_VERSION: "2.0";
export const INFERENCE_EXECUTION_MODES: readonly ["container", "direct"];
export const INFERENCE_TASK_VALUES: readonly [
  "detect",
  "segment",
  "classify",
  "pose",
  "obb",
  "ocr",
  "doc_layout",
  "tracker",
  "interactive_seg",
];
export const INFERENCE_CAPABILITIES: readonly [
  "inference",
  "detection",
  "segmentation",
  "interactive",
  "video_tracking",
  "ocr",
  "classification",
  "obb",
  "pose",
  "doc_layout",
  "managed_models",
  "managed_evaluation",
];
export const TRAINER_PROTOCOL_VERSION: "yoao-trainer/v1";
export const SHA256_PATTERN: "^[0-9a-f]{64}$";
export const PREVIEW_SEED_MAX: number;
export const PREVIEW_MAX_SOURCES: number;
export const PREVIEW_MAX_OUTPUTS: number;
export const PREVIEW_MAX_EDGE: number;
export const PREVIEW_MAX_SOURCE_BYTES: number;
export const PREVIEW_MAX_POOL_BYTES: number;
export const PREVIEW_MAX_SOURCE_PIXELS: number;
export const PREVIEW_MAX_POOL_PIXELS: number;
export const PREVIEW_MAX_SOURCE_BOXES: number;
export const PREVIEW_MAX_POOL_BOXES: number;
export const PREVIEW_MAX_OUTPUT_BOXES: number;
export const PREVIEW_MAX_OUTPUT_BYTES: number;

export type ExtensionPoint =
  | "inference"
  | "notification_channel"
  | "format_adapter"
  | "trainer"
  | "theme";

export type InferenceExecutionMode = "container" | "direct";
export type InferenceTask =
  | "detect"
  | "segment"
  | "classify"
  | "pose"
  | "obb"
  | "ocr"
  | "doc_layout"
  | "tracker"
  | "interactive_seg";
export type InferenceCapability =
  | "inference"
  | "detection"
  | "segmentation"
  | "interactive"
  | "video_tracking"
  | "ocr"
  | "classification"
  | "obb"
  | "pose"
  | "doc_layout"
  | "managed_models"
  | "managed_evaluation";

export interface InferenceComputeV1 {
  device: "cpu" | "cuda";
  gpu_memory_mb?: number;
}

export interface InferenceModelCapabilityV1 {
  id: string;
  task: InferenceTask;
  is_interactive: boolean;
  supported_prompts: string[];
  supported_inputs: string[];
  supported_geometric_outputs: string[];
}

export interface InferencePoolTopologyV1 {
  cap: number;
}

export interface InferenceManagedModelCompatibilityV1 {
  framework: string;
  format: string | null;
  runtime_device: "cpu" | "cuda";
}

export interface InferenceExtensionV1 {
  execution_mode?: InferenceExecutionMode;
  models: InferenceModelCapabilityV1[];
  pools: Record<string, InferencePoolTopologyV1>;
  compute: InferenceComputeV1;
  sdk_compatibility?: string | null;
  asset_manifest_sha256?: string | null;
  allowed_asset_sources?: string[];
  managed_model_compatibility?: InferenceManagedModelCompatibilityV1[];
}

export interface ExtensionManifestV1 {
  manifest_schema: "yoao-extension-manifest/v1";
  extension_point: ExtensionPoint;
  plugin_id: string;
  version: string;
  display_name?: string | null;
  description?: string | null;
  homepage?: string | null;
  publisher: string;
  license: string;
  protocol_version: string;
  entrypoint?: string[] | null;
  capabilities: string[];
  resource?: { memory_mb: number } | null;
  extension?: InferenceExtensionV1 | Record<string, unknown>;
  artifact?: string | null;
}

export interface ThemeTokensDocument {
  light?: Record<string, string>;
  dark?: Record<string, string>;
}

export interface ServingCompatibilityV1 {
  framework: string;
  format: string | null;
  training_runtime?: Record<string, unknown>;
  compatible_serving_runtimes?: Array<Record<string, unknown>>;
}

export interface TrainerArchitectureV1 {
  id: string;
  family: string;
  tasks: string[];
  end2end?: boolean;
  base_asset_filename?: string | null;
  base_asset_sha256?: string | null;
}

export interface TrainerCapabilitiesV1 {
  protocol_version: "yoao-trainer/v1";
  framework_id: string;
  framework_version: string;
  runtime_versions?: Record<string, string>;
  supported_tasks: string[];
  supported_dataset_contracts: string[];
  architectures: TrainerArchitectureV1[];
  parameter_schema?: Record<string, unknown>;
  evaluator_descriptors?: Record<string, Record<string, unknown>>;
  policy_digests?: Record<string, string>;
  serving_compatibility?: ServingCompatibilityV1[];
}

export interface ValidationIssueV1 {
  code: string;
  message: string;
  field?: string | null;
}

export interface ValidateConfigResultV1 {
  protocol_version: "yoao-trainer/v1";
  valid: boolean;
  schema_id?: string | null;
  schema_digest?: string | null;
  resolved_config?: Record<string, unknown>;
  effective_values?: Record<string, unknown>;
  origins?: Record<string, string>;
  inactive?: Record<string, string>;
  issues?: ValidationIssueV1[];
}

export interface TrainSpecV1 {
  protocol_version: "yoao-trainer/v1";
  run_id: string;
  attempt_id: string;
  framework_id: string;
  task: string;
  architecture_id: string;
  dataset_contract: string;
  dataset_dir: string;
  dataset_snapshot_digest: string;
  config_digest: string;
  classes_digest: string;
  schema_id?: string | null;
  schema_digest?: string | null;
  parameters?: Record<string, unknown>;
  base_weights_path?: string | null;
  resume_checkpoint_path?: string | null;
  init_weights_path?: string | null;
  native_budget?: Record<string, unknown> | null;
}

export interface TrainEventV1 {
  seq: number;
  kind: "phase" | "epoch" | "progress" | "checkpoint" | "warning";
  phase?: string | null;
  epoch?: number | null;
  total_epochs?: number | null;
  progress?: number | null;
  metrics?: Record<string, unknown>;
  detail?: Record<string, unknown>;
}

export interface TrainArtifactV1 {
  role: "weights" | "checkpoint" | "metrics" | "config" | "input_membership" | "extra";
  filename: string;
  sha256: string;
  size_bytes: number;
}

export interface TrainReceiptV1 {
  protocol_version: "yoao-trainer/v1";
  status: "succeeded" | "failed" | "cancelled";
  artifacts?: TrainArtifactV1[];
  serving_compatibility?: ServingCompatibilityV1 | null;
  failure_code?: string | null;
  failure_message?: string | null;
}

export interface PreviewSourceV1 {
  task_id: string;
  filename: string;
  sha256: string;
  width: number;
  height: number;
  size_bytes: number;
  box_count: number;
  label_path?: string | null;
  label_sha256?: string | null;
}

export interface PreviewSpecV1 {
  protocol_version: "yoao-trainer/v1";
  preview_id: string;
  framework_id: string;
  task: string;
  architecture_id: string;
  input_digest: string;
  config_digest: string;
  dataset_snapshot_digest: string;
  policy_digest: string;
  seed: number;
  epoch: number;
  count: number;
  sources: PreviewSourceV1[];
  parameters?: Record<string, unknown>;
  native_values?: Record<string, unknown>;
  pool_summary?: Record<string, unknown>;
}

export interface PreviewOutputV1 {
  filename: string;
  sha256: string;
  size_bytes: number;
  width: number;
  height: number;
  box_count: number;
  metadata?: Record<string, unknown>;
}

export interface PreviewReportV1 {
  protocol_version: "yoao-trainer/v1";
  status: "succeeded" | "failed";
  input_digest: string;
  policy_digest: string;
  cuda_initialized?: boolean;
  outputs?: PreviewOutputV1[];
  warnings?: string[];
  runtime?: Record<string, unknown>;
  details?: Record<string, unknown>;
  failure_code?: string | null;
  failure_message?: string | null;
}

export function validateManifest(manifest: unknown): ExtensionManifestV1;
export function validateThemeTokens(themeDoc: unknown): ThemeTokensDocument;
export function readSecret(
  env: Record<string, string | undefined> | undefined,
  valueKey: string,
  fileKey: string,
): string;
export function verifyBearerToken(
  authorizationHeader: string | undefined,
  expectedToken: string,
): boolean;
export function canonicalJsonDigest(payload: object): string;
export function validateServingCompatibility(payload: unknown): ServingCompatibilityV1;
export function validateTrainerCapabilities(payload: unknown): TrainerCapabilitiesV1;
export function validateConfigResult(payload: unknown): ValidateConfigResultV1;
export function validateTrainSpec(payload: unknown): TrainSpecV1;
export function validateTrainEvent(payload: unknown): TrainEventV1;
export function validateTrainReceipt(payload: unknown): TrainReceiptV1;
export function validatePreviewSpec(payload: unknown): PreviewSpecV1;
export function validatePreviewReport(payload: unknown): PreviewReportV1;
