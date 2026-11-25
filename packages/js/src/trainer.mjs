import crypto from "node:crypto";

export const TRAINER_PROTOCOL_VERSION = "yoao-trainer/v1";
export const SHA256_PATTERN = "^[0-9a-f]{64}$";

export const PREVIEW_SEED_MAX = 2147483647;
export const PREVIEW_MAX_SOURCES = 16;
export const PREVIEW_MAX_OUTPUTS = 4;
export const PREVIEW_MAX_EDGE = 2560;
export const PREVIEW_MAX_SOURCE_BYTES = 8 * 1024 * 1024;
export const PREVIEW_MAX_POOL_BYTES = 64 * 1024 * 1024;
export const PREVIEW_MAX_SOURCE_PIXELS = 16_000_000;
export const PREVIEW_MAX_POOL_PIXELS = 32_000_000;
export const PREVIEW_MAX_SOURCE_BOXES = 256;
export const PREVIEW_MAX_POOL_BOXES = 4096;
export const PREVIEW_MAX_OUTPUT_BOXES = 4096;
export const PREVIEW_MAX_OUTPUT_BYTES = 24 * 1024 * 1024;

const SHA256_RE = new RegExp(SHA256_PATTERN);

const ARTIFACT_ROLES = new Set([
  "weights",
  "checkpoint",
  "metrics",
  "config",
  "input_membership",
  "extra",
]);
const EVENT_KINDS = new Set(["phase", "epoch", "progress", "checkpoint", "warning"]);
const RECEIPT_STATUSES = new Set(["succeeded", "failed", "cancelled"]);
const PREVIEW_STATUSES = new Set(["succeeded", "failed"]);

function isPlainObject(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

function assertAllowedKeys(obj, allowedKeys, label) {
  for (const key of Object.keys(obj)) {
    if (!allowedKeys.has(key)) {
      throw new Error(`${label} contains unexpected field: ${key}`);
    }
  }
}

function hasControlChars(value) {
  for (let i = 0; i < value.length; i += 1) {
    const code = value.charCodeAt(i);
    if (code < 0x20 || code === 0x7f) return true;
  }
  return false;
}

function assertTrimmedString(value, field, maxLen = 1024) {
  if (
    typeof value !== "string" ||
    !value.trim() ||
    value !== value.trim() ||
    value.length > maxLen ||
    hasControlChars(value)
  ) {
    throw new Error(`${field} must be a non-empty trimmed string (max ${maxLen} chars)`);
  }
}

function assertOptionalTrimmedString(value, field, maxLen = 1024) {
  if (value === undefined || value === null) return;
  assertTrimmedString(value, field, maxLen);
}

function assertSha256(value, field) {
  if (typeof value !== "string" || !SHA256_RE.test(value)) {
    throw new Error(`${field} must be a 64-character lowercase hex SHA-256`);
  }
}

function assertSafeFilename(value, field = "filename") {
  assertTrimmedString(value, field, 255);
  if (value.includes("/") || value.includes("\\") || value === "." || value === "..") {
    throw new Error(`${field} must be a single safe filename without path separators`);
  }
}

function assertSafeRelativePath(value, field = "filename") {
  assertTrimmedString(value, field, 255);
  if (value.startsWith("/") || value.includes("\\")) {
    throw new Error(`${field} must be a relative POSIX path`);
  }
  const segments = value.split("/");
  if (segments.some((part) => part === "" || part === "." || part === "..")) {
    throw new Error(`${field} must not contain empty or traversal path segments`);
  }
}

function assertUniqueTrimmedStringArray(value, field) {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error(`${field} must be a non-empty string array`);
  }
  const seen = new Set();
  for (const item of value) {
    assertTrimmedString(item, field, 128);
    if (seen.has(item)) {
      throw new Error(`${field} must not contain duplicates: ${item}`);
    }
    seen.add(item);
  }
}

function assertIntInRange(value, field, min, max = Number.MAX_SAFE_INTEGER) {
  if (!Number.isInteger(value) || value < min || value > max) {
    throw new Error(`${field} must be an integer in [${min}, ${max}]`);
  }
}

function compareCodePoints(a, b) {
  const iterA = Array.from(a);
  const iterB = Array.from(b);
  const len = Math.min(iterA.length, iterB.length);
  for (let i = 0; i < len; i += 1) {
    const diff = iterA[i].codePointAt(0) - iterB[i].codePointAt(0);
    if (diff !== 0) return diff;
  }
  return iterA.length - iterB.length;
}

function encodeCanonicalJson(value) {
  if (value === null) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new Error("canonical JSON forbids NaN and Infinity");
    }
    return JSON.stringify(Object.is(value, -0) ? 0 : value);
  }
  if (typeof value === "string") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    const items = [];
    for (let i = 0; i < value.length; i += 1) {
      if (!(i in value)) {
        throw new Error(`canonical JSON forbids sparse array hole at index ${i}`);
      }
      items.push(encodeCanonicalJson(value[i]));
    }
    return `[${items.join(",")}]`;
  }
  if (isPlainObject(value)) {
    const keys = Object.keys(value).sort(compareCodePoints);
    const parts = [];
    for (const key of keys) {
      if (value[key] === undefined) {
        throw new Error(`canonical JSON forbids undefined value at key ${key}`);
      }
      parts.push(`${JSON.stringify(key)}:${encodeCanonicalJson(value[key])}`);
    }
    return `{${parts.join(",")}}`;
  }
  throw new Error(`unsupported value type in canonical JSON: ${typeof value}`);
}

export function canonicalJsonDigest(payload) {
  if (!isPlainObject(payload)) {
    throw new Error("payload must be a plain JSON object");
  }
  const encoded = encodeCanonicalJson(payload);
  return crypto.createHash("sha256").update(encoded, "utf8").digest("hex");
}

const SERVING_COMPAT_KEYS = new Set([
  "framework",
  "format",
  "training_runtime",
  "compatible_serving_runtimes",
]);

export function validateServingCompatibility(payload) {
  if (!isPlainObject(payload)) {
    throw new Error("serving compatibility must be a JSON object");
  }
  assertAllowedKeys(payload, SERVING_COMPAT_KEYS, "ServingCompatibilityV1");
  assertTrimmedString(payload.framework, "framework", 64);
  if (!Object.prototype.hasOwnProperty.call(payload, "format")) {
    throw new Error("format must be explicitly provided (string or null)");
  }
  if (payload.format !== null) {
    assertTrimmedString(payload.format, "format", 64);
  }
  if (payload.training_runtime !== undefined && !isPlainObject(payload.training_runtime)) {
    throw new Error("training_runtime must be an object");
  }
  if (payload.compatible_serving_runtimes !== undefined) {
    if (
      !Array.isArray(payload.compatible_serving_runtimes) ||
      !payload.compatible_serving_runtimes.every(isPlainObject)
    ) {
      throw new Error("compatible_serving_runtimes must be an array of objects");
    }
  }
  return payload;
}

const ARCH_KEYS = new Set([
  "id",
  "family",
  "tasks",
  "end2end",
  "base_asset_filename",
  "base_asset_sha256",
]);

const CAPABILITIES_KEYS = new Set([
  "protocol_version",
  "framework_id",
  "framework_version",
  "runtime_versions",
  "supported_tasks",
  "supported_dataset_contracts",
  "architectures",
  "parameter_schema",
  "evaluator_descriptors",
  "policy_digests",
  "serving_compatibility",
]);

export function validateTrainerCapabilities(payload) {
  if (!isPlainObject(payload)) {
    throw new Error("capabilities must be a JSON object");
  }
  assertAllowedKeys(payload, CAPABILITIES_KEYS, "TrainerCapabilitiesV1");
  if (payload.protocol_version !== TRAINER_PROTOCOL_VERSION) {
    throw new Error(`protocol_version must be ${TRAINER_PROTOCOL_VERSION}`);
  }
  assertTrimmedString(payload.framework_id, "framework_id", 64);
  assertTrimmedString(payload.framework_version, "framework_version", 64);
  assertUniqueTrimmedStringArray(payload.supported_tasks, "supported_tasks");
  assertUniqueTrimmedStringArray(
    payload.supported_dataset_contracts,
    "supported_dataset_contracts",
  );
  if (!Array.isArray(payload.architectures) || payload.architectures.length === 0) {
    throw new Error("architectures must be a non-empty array");
  }
  const supportedTasks = new Set(payload.supported_tasks);
  const seenArch = new Set();
  for (const arch of payload.architectures) {
    if (!isPlainObject(arch)) throw new Error("architecture entry must be an object");
    assertAllowedKeys(arch, ARCH_KEYS, "TrainerArchitectureV1");
    assertTrimmedString(arch.id, "architecture.id", 128);
    if (seenArch.has(arch.id)) throw new Error(`duplicate architecture id: ${arch.id}`);
    seenArch.add(arch.id);
    assertTrimmedString(arch.family, "architecture.family", 64);
    assertUniqueTrimmedStringArray(arch.tasks, "architecture.tasks");
    for (const task of arch.tasks) {
      if (!supportedTasks.has(task)) {
        throw new Error(`architecture ${arch.id} references unsupported task: ${task}`);
      }
    }
    if (arch.end2end !== undefined && typeof arch.end2end !== "boolean") {
      throw new Error("architecture.end2end must be a boolean");
    }
    if (arch.base_asset_filename !== undefined && arch.base_asset_filename !== null) {
      assertSafeFilename(arch.base_asset_filename, "base_asset_filename");
    }
    if (arch.base_asset_sha256 !== undefined && arch.base_asset_sha256 !== null) {
      assertSha256(arch.base_asset_sha256, "base_asset_sha256");
      if (!arch.base_asset_filename) {
        throw new Error("base_asset_sha256 requires base_asset_filename");
      }
    }
  }
  if (payload.runtime_versions !== undefined) {
    if (!isPlainObject(payload.runtime_versions)) {
      throw new Error("runtime_versions must be an object");
    }
    for (const [k, v] of Object.entries(payload.runtime_versions)) {
      assertTrimmedString(k, "runtime_versions key", 64);
      assertTrimmedString(v, `runtime_versions[${k}]`, 128);
    }
  }
  if (payload.parameter_schema !== undefined && !isPlainObject(payload.parameter_schema)) {
    throw new Error("parameter_schema must be an object");
  }
  if (payload.evaluator_descriptors !== undefined) {
    if (!isPlainObject(payload.evaluator_descriptors)) {
      throw new Error("evaluator_descriptors must be an object");
    }
    for (const [k, v] of Object.entries(payload.evaluator_descriptors)) {
      if (!isPlainObject(v)) throw new Error(`evaluator_descriptors[${k}] must be an object`);
    }
  }
  if (payload.policy_digests !== undefined) {
    if (!isPlainObject(payload.policy_digests)) {
      throw new Error("policy_digests must be an object");
    }
    for (const [k, v] of Object.entries(payload.policy_digests)) {
      assertTrimmedString(k, "policy_digests key", 64);
      assertSha256(v, `policy_digests[${k}]`);
    }
  }
  if (payload.serving_compatibility !== undefined) {
    if (!Array.isArray(payload.serving_compatibility)) {
      throw new Error("serving_compatibility must be an array");
    }
    for (const entry of payload.serving_compatibility) {
      validateServingCompatibility(entry);
    }
  }
  return payload;
}

const ISSUE_KEYS = new Set(["code", "message", "field"]);
const VALIDATE_RESULT_KEYS = new Set([
  "protocol_version",
  "valid",
  "schema_id",
  "schema_digest",
  "resolved_config",
  "effective_values",
  "origins",
  "inactive",
  "issues",
]);

export function validateConfigResult(payload) {
  if (!isPlainObject(payload)) {
    throw new Error("validate-config result must be a JSON object");
  }
  assertAllowedKeys(payload, VALIDATE_RESULT_KEYS, "ValidateConfigResultV1");
  if (payload.protocol_version !== TRAINER_PROTOCOL_VERSION) {
    throw new Error(`protocol_version must be ${TRAINER_PROTOCOL_VERSION}`);
  }
  if (typeof payload.valid !== "boolean") {
    throw new Error("valid must be a boolean");
  }
  assertOptionalTrimmedString(payload.schema_id, "schema_id", 128);
  if (payload.schema_digest !== undefined && payload.schema_digest !== null) {
    assertSha256(payload.schema_digest, "schema_digest");
  }
  if (payload.resolved_config !== undefined && !isPlainObject(payload.resolved_config)) {
    throw new Error("resolved_config must be an object");
  }
  if (payload.effective_values !== undefined && !isPlainObject(payload.effective_values)) {
    throw new Error("effective_values must be an object");
  }
  for (const mapField of ["origins", "inactive"]) {
    if (payload[mapField] !== undefined) {
      if (!isPlainObject(payload[mapField])) throw new Error(`${mapField} must be an object`);
      for (const [k, v] of Object.entries(payload[mapField])) {
        assertTrimmedString(k, `${mapField} key`, 128);
        assertTrimmedString(v, `${mapField}[${k}]`, 512);
      }
    }
  }
  const issues = payload.issues === undefined ? [] : payload.issues;
  if (!Array.isArray(issues)) throw new Error("issues must be an array");
  for (const issue of issues) {
    if (!isPlainObject(issue)) throw new Error("validation issue must be an object");
    assertAllowedKeys(issue, ISSUE_KEYS, "ValidationIssueV1");
    assertTrimmedString(issue.code, "issue.code", 64);
    assertTrimmedString(issue.message, "issue.message", 512);
    assertOptionalTrimmedString(issue.field, "issue.field", 128);
  }
  if (payload.valid) {
    if (issues.length > 0) {
      throw new Error("valid=true must not include validation issues");
    }
    if (!payload.schema_id || !payload.schema_digest) {
      throw new Error("valid=true requires both schema_id and schema_digest");
    }
  } else if (issues.length === 0) {
    throw new Error("valid=false requires at least one validation issue");
  }
  return payload;
}

const TRAIN_SPEC_KEYS = new Set([
  "protocol_version",
  "run_id",
  "attempt_id",
  "framework_id",
  "task",
  "architecture_id",
  "dataset_contract",
  "dataset_dir",
  "dataset_snapshot_digest",
  "config_digest",
  "classes_digest",
  "schema_id",
  "schema_digest",
  "parameters",
  "base_weights_path",
  "resume_checkpoint_path",
  "init_weights_path",
  "native_budget",
]);

export function validateTrainSpec(payload) {
  if (!isPlainObject(payload)) {
    throw new Error("train spec must be a JSON object");
  }
  assertAllowedKeys(payload, TRAIN_SPEC_KEYS, "TrainSpecV1");
  if (payload.protocol_version !== TRAINER_PROTOCOL_VERSION) {
    throw new Error(`protocol_version must be ${TRAINER_PROTOCOL_VERSION}`);
  }
  assertTrimmedString(payload.run_id, "run_id", 128);
  assertTrimmedString(payload.attempt_id, "attempt_id", 128);
  assertTrimmedString(payload.framework_id, "framework_id", 64);
  assertTrimmedString(payload.task, "task", 64);
  assertTrimmedString(payload.architecture_id, "architecture_id", 128);
  assertTrimmedString(payload.dataset_contract, "dataset_contract", 128);
  assertTrimmedString(payload.dataset_dir, "dataset_dir", 1024);
  assertSha256(payload.dataset_snapshot_digest, "dataset_snapshot_digest");
  assertSha256(payload.config_digest, "config_digest");
  assertSha256(payload.classes_digest, "classes_digest");
  assertOptionalTrimmedString(payload.schema_id, "schema_id", 128);
  if (payload.schema_digest !== undefined && payload.schema_digest !== null) {
    assertSha256(payload.schema_digest, "schema_digest");
  }
  const hasSchemaId = payload.schema_id !== undefined && payload.schema_id !== null;
  const hasSchemaDigest = payload.schema_digest !== undefined && payload.schema_digest !== null;
  if (hasSchemaId !== hasSchemaDigest) {
    throw new Error("schema_id and schema_digest must be provided together");
  }
  assertOptionalTrimmedString(payload.base_weights_path, "base_weights_path", 1024);
  assertOptionalTrimmedString(payload.resume_checkpoint_path, "resume_checkpoint_path", 1024);
  assertOptionalTrimmedString(payload.init_weights_path, "init_weights_path", 1024);
  if (payload.resume_checkpoint_path && payload.init_weights_path) {
    throw new Error("resume_checkpoint_path and init_weights_path are mutually exclusive");
  }
  if (payload.parameters !== undefined && !isPlainObject(payload.parameters)) {
    throw new Error("parameters must be an object");
  }
  if (
    payload.native_budget !== undefined &&
    payload.native_budget !== null &&
    !isPlainObject(payload.native_budget)
  ) {
    throw new Error("native_budget must be an object or null");
  }
  return payload;
}

const EVENT_KEYS = new Set([
  "seq",
  "kind",
  "phase",
  "epoch",
  "total_epochs",
  "progress",
  "metrics",
  "detail",
]);

export function validateTrainEvent(payload) {
  if (!isPlainObject(payload)) {
    throw new Error("train event must be a JSON object");
  }
  assertAllowedKeys(payload, EVENT_KEYS, "TrainEventV1");
  assertIntInRange(payload.seq, "seq", 1);
  if (typeof payload.kind !== "string" || !EVENT_KINDS.has(payload.kind)) {
    throw new Error(`unsupported event kind: ${String(payload.kind)}`);
  }
  assertOptionalTrimmedString(payload.phase, "phase", 128);
  if (payload.epoch !== undefined && payload.epoch !== null) {
    assertIntInRange(payload.epoch, "epoch", 0);
  }
  if (payload.total_epochs !== undefined && payload.total_epochs !== null) {
    assertIntInRange(payload.total_epochs, "total_epochs", 1);
  }
  if (
    payload.epoch !== undefined &&
    payload.epoch !== null &&
    payload.total_epochs !== undefined &&
    payload.total_epochs !== null &&
    payload.epoch > payload.total_epochs
  ) {
    throw new Error("epoch cannot exceed total_epochs");
  }
  if (payload.progress !== undefined && payload.progress !== null) {
    if (
      typeof payload.progress !== "number" ||
      !Number.isFinite(payload.progress) ||
      payload.progress < 0 ||
      payload.progress > 1
    ) {
      throw new Error("progress must be a finite number in [0, 1]");
    }
  }
  if (payload.metrics !== undefined && !isPlainObject(payload.metrics)) {
    throw new Error("metrics must be an object");
  }
  if (payload.detail !== undefined && !isPlainObject(payload.detail)) {
    throw new Error("detail must be an object");
  }
  return payload;
}

const ARTIFACT_KEYS = new Set(["role", "filename", "sha256", "size_bytes"]);
const RECEIPT_KEYS = new Set([
  "protocol_version",
  "status",
  "artifacts",
  "serving_compatibility",
  "failure_code",
  "failure_message",
]);

export function validateTrainReceipt(payload) {
  if (!isPlainObject(payload)) {
    throw new Error("train receipt must be a JSON object");
  }
  assertAllowedKeys(payload, RECEIPT_KEYS, "TrainReceiptV1");
  if (payload.protocol_version !== TRAINER_PROTOCOL_VERSION) {
    throw new Error(`protocol_version must be ${TRAINER_PROTOCOL_VERSION}`);
  }
  if (typeof payload.status !== "string" || !RECEIPT_STATUSES.has(payload.status)) {
    throw new Error(`unsupported receipt status: ${String(payload.status)}`);
  }
  const artifacts = payload.artifacts === undefined ? [] : payload.artifacts;
  if (!Array.isArray(artifacts)) throw new Error("artifacts must be an array");
  const seenNames = new Set();
  for (const artifact of artifacts) {
    if (!isPlainObject(artifact)) throw new Error("artifact must be an object");
    assertAllowedKeys(artifact, ARTIFACT_KEYS, "TrainArtifactV1");
    if (typeof artifact.role !== "string" || !ARTIFACT_ROLES.has(artifact.role)) {
      throw new Error(`unsupported artifact role: ${String(artifact.role)}`);
    }
    assertSafeFilename(artifact.filename, "artifact.filename");
    if (seenNames.has(artifact.filename)) {
      throw new Error(`duplicate artifact filename: ${artifact.filename}`);
    }
    seenNames.add(artifact.filename);
    assertSha256(artifact.sha256, "artifact.sha256");
    assertIntInRange(artifact.size_bytes, "artifact.size_bytes", 0);
  }
  if (payload.serving_compatibility !== undefined && payload.serving_compatibility !== null) {
    validateServingCompatibility(payload.serving_compatibility);
  }
  assertOptionalTrimmedString(payload.failure_code, "failure_code", 64);
  assertOptionalTrimmedString(payload.failure_message, "failure_message", 2048);
  if (payload.status === "failed" && !payload.failure_code) {
    throw new Error("status='failed' requires failure_code");
  }
  if (
    payload.status === "succeeded" &&
    (payload.failure_code || payload.failure_message)
  ) {
    throw new Error("status='succeeded' must not set failure_code or failure_message");
  }
  return payload;
}

const PREVIEW_SOURCE_KEYS = new Set([
  "task_id",
  "filename",
  "sha256",
  "width",
  "height",
  "size_bytes",
  "box_count",
  "label_path",
  "label_sha256",
]);
const PREVIEW_SPEC_KEYS = new Set([
  "protocol_version",
  "preview_id",
  "framework_id",
  "task",
  "architecture_id",
  "input_digest",
  "config_digest",
  "dataset_snapshot_digest",
  "policy_digest",
  "seed",
  "epoch",
  "count",
  "sources",
  "parameters",
  "native_values",
  "pool_summary",
]);

export function validatePreviewSpec(payload) {
  if (!isPlainObject(payload)) {
    throw new Error("preview spec must be a JSON object");
  }
  assertAllowedKeys(payload, PREVIEW_SPEC_KEYS, "PreviewSpecV1");
  if (payload.protocol_version !== TRAINER_PROTOCOL_VERSION) {
    throw new Error(`protocol_version must be ${TRAINER_PROTOCOL_VERSION}`);
  }
  assertTrimmedString(payload.preview_id, "preview_id", 128);
  assertTrimmedString(payload.framework_id, "framework_id", 64);
  assertTrimmedString(payload.task, "task", 64);
  assertTrimmedString(payload.architecture_id, "architecture_id", 128);
  assertSha256(payload.input_digest, "input_digest");
  assertSha256(payload.config_digest, "config_digest");
  assertSha256(payload.dataset_snapshot_digest, "dataset_snapshot_digest");
  assertSha256(payload.policy_digest, "policy_digest");
  assertIntInRange(payload.seed, "seed", 0, PREVIEW_SEED_MAX);
  assertIntInRange(payload.epoch, "epoch", 1);
  assertIntInRange(payload.count, "count", 1, PREVIEW_MAX_OUTPUTS);
  if (
    !Array.isArray(payload.sources) ||
    payload.sources.length < 1 ||
    payload.sources.length > PREVIEW_MAX_SOURCES
  ) {
    throw new Error(`sources must contain 1..${PREVIEW_MAX_SOURCES} entries`);
  }
  const seenSources = new Set();
  let totalBytes = 0;
  let totalPixels = 0;
  let totalBoxes = 0;
  for (const source of payload.sources) {
    if (!isPlainObject(source)) throw new Error("preview source must be an object");
    assertAllowedKeys(source, PREVIEW_SOURCE_KEYS, "PreviewSourceV1");
    assertTrimmedString(source.task_id, "source.task_id", 128);
    assertSafeRelativePath(source.filename, "source.filename");
    if (seenSources.has(source.filename)) {
      throw new Error(`duplicate preview source filename: ${source.filename}`);
    }
    seenSources.add(source.filename);
    assertSha256(source.sha256, "source.sha256");
    assertIntInRange(source.width, "source.width", 1, PREVIEW_MAX_SOURCE_PIXELS);
    assertIntInRange(source.height, "source.height", 1, PREVIEW_MAX_SOURCE_PIXELS);
    const pixels = source.width * source.height;
    if (pixels > PREVIEW_MAX_SOURCE_PIXELS) {
      throw new Error(`source pixels exceed ${PREVIEW_MAX_SOURCE_PIXELS}`);
    }
    assertIntInRange(source.size_bytes, "source.size_bytes", 1, PREVIEW_MAX_SOURCE_BYTES);
    assertIntInRange(source.box_count, "source.box_count", 0, PREVIEW_MAX_SOURCE_BOXES);
    const hasLabelPath = source.label_path !== undefined && source.label_path !== null;
    const hasLabelSha = source.label_sha256 !== undefined && source.label_sha256 !== null;
    if (hasLabelPath !== hasLabelSha) {
      throw new Error("label_path and label_sha256 must be provided together");
    }
    if (hasLabelPath) {
      assertSafeRelativePath(source.label_path, "source.label_path");
      assertSha256(source.label_sha256, "source.label_sha256");
    }
    totalBytes += source.size_bytes;
    totalPixels += pixels;
    totalBoxes += source.box_count;
  }
  if (totalBytes > PREVIEW_MAX_POOL_BYTES) {
    throw new Error(`preview pool bytes exceed ${PREVIEW_MAX_POOL_BYTES}`);
  }
  if (totalPixels > PREVIEW_MAX_POOL_PIXELS) {
    throw new Error(`preview pool pixels exceed ${PREVIEW_MAX_POOL_PIXELS}`);
  }
  if (totalBoxes > PREVIEW_MAX_POOL_BOXES) {
    throw new Error(`preview pool boxes exceed ${PREVIEW_MAX_POOL_BOXES}`);
  }
  for (const objField of ["parameters", "native_values", "pool_summary"]) {
    if (payload[objField] !== undefined && !isPlainObject(payload[objField])) {
      throw new Error(`${objField} must be an object`);
    }
  }
  return payload;
}

const PREVIEW_OUTPUT_KEYS = new Set([
  "filename",
  "sha256",
  "size_bytes",
  "width",
  "height",
  "box_count",
  "metadata",
]);
const PREVIEW_REPORT_KEYS = new Set([
  "protocol_version",
  "status",
  "input_digest",
  "policy_digest",
  "cuda_initialized",
  "outputs",
  "warnings",
  "runtime",
  "details",
  "failure_code",
  "failure_message",
]);

export function validatePreviewReport(payload) {
  if (!isPlainObject(payload)) {
    throw new Error("preview report must be a JSON object");
  }
  assertAllowedKeys(payload, PREVIEW_REPORT_KEYS, "PreviewReportV1");
  if (payload.protocol_version !== TRAINER_PROTOCOL_VERSION) {
    throw new Error(`protocol_version must be ${TRAINER_PROTOCOL_VERSION}`);
  }
  if (typeof payload.status !== "string" || !PREVIEW_STATUSES.has(payload.status)) {
    throw new Error(`unsupported preview report status: ${String(payload.status)}`);
  }
  assertSha256(payload.input_digest, "input_digest");
  assertSha256(payload.policy_digest, "policy_digest");
  const cudaInit = payload.cuda_initialized === undefined ? false : payload.cuda_initialized;
  if (typeof cudaInit !== "boolean") {
    throw new Error("cuda_initialized must be a boolean");
  }
  const outputs = payload.outputs === undefined ? [] : payload.outputs;
  if (!Array.isArray(outputs) || outputs.length > PREVIEW_MAX_OUTPUTS) {
    throw new Error(`outputs must be an array of at most ${PREVIEW_MAX_OUTPUTS} items`);
  }
  const seenOutputs = new Set();
  for (const output of outputs) {
    if (!isPlainObject(output)) throw new Error("preview output must be an object");
    assertAllowedKeys(output, PREVIEW_OUTPUT_KEYS, "PreviewOutputV1");
    assertSafeFilename(output.filename, "output.filename");
    if (seenOutputs.has(output.filename)) {
      throw new Error(`duplicate preview output filename: ${output.filename}`);
    }
    seenOutputs.add(output.filename);
    assertSha256(output.sha256, "output.sha256");
    assertIntInRange(output.size_bytes, "output.size_bytes", 1, PREVIEW_MAX_OUTPUT_BYTES);
    assertIntInRange(output.width, "output.width", 1, PREVIEW_MAX_EDGE);
    assertIntInRange(output.height, "output.height", 1, PREVIEW_MAX_EDGE);
    assertIntInRange(output.box_count, "output.box_count", 0, PREVIEW_MAX_OUTPUT_BOXES);
    if (output.metadata !== undefined && !isPlainObject(output.metadata)) {
      throw new Error("output.metadata must be an object");
    }
  }
  if (payload.warnings !== undefined) {
    if (!Array.isArray(payload.warnings)) throw new Error("warnings must be an array");
    for (const warning of payload.warnings) {
      assertTrimmedString(warning, "warnings", 1024);
    }
  }
  for (const objField of ["runtime", "details"]) {
    if (payload[objField] !== undefined && !isPlainObject(payload[objField])) {
      throw new Error(`${objField} must be an object`);
    }
  }
  assertOptionalTrimmedString(payload.failure_code, "failure_code", 64);
  assertOptionalTrimmedString(payload.failure_message, "failure_message", 2048);
  if (payload.status === "succeeded") {
    if (cudaInit) {
      throw new Error("status='succeeded' preview must not initialize CUDA");
    }
    if (outputs.length === 0) {
      throw new Error("status='succeeded' preview requires at least one output");
    }
    if (payload.failure_code || payload.failure_message) {
      throw new Error("status='succeeded' must not set failure_code or failure_message");
    }
  } else if (!payload.failure_code) {
    throw new Error("status='failed' requires failure_code");
  }
  return payload;
}
