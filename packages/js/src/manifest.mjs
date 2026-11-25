export const EXTENSION_MANIFEST_SCHEMA = "yoao-extension-manifest/v1";
export const INFERENCE_PROTOCOL_VERSION = "yoao-inference/v2";
export const INFERENCE_WIRE_PROTOCOL_VERSION = "2.0";

export const INFERENCE_EXECUTION_MODES = Object.freeze(["container", "direct"]);
export const INFERENCE_TASK_VALUES = Object.freeze([
  "detect",
  "segment",
  "classify",
  "pose",
  "obb",
  "ocr",
  "doc_layout",
  "tracker",
  "interactive_seg",
]);
export const INFERENCE_CAPABILITIES = Object.freeze([
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
]);

const PROTOCOL_BY_POINT = Object.freeze({
  inference: INFERENCE_PROTOCOL_VERSION,
  notification_channel: "yoao-notification-channel/v1",
  format_adapter: "yoao-format-adapter/v1",
  trainer: "yoao-trainer/v1",
  theme: "yoao-theme/v1",
});

const CAPABILITIES_BY_POINT = Object.freeze({
  inference: new Set(INFERENCE_CAPABILITIES),
  notification_channel: new Set(["deliver", "verify_address"]),
  format_adapter: new Set(["import", "export"]),
  trainer: new Set(["train", "evaluate", "preview", "export_weights"]),
  theme: new Set(["light", "dark"]),
});

const TOP_LEVEL_MANIFEST_KEYS = new Set([
  "manifest_schema",
  "extension_point",
  "plugin_id",
  "version",
  "display_name",
  "description",
  "homepage",
  "publisher",
  "license",
  "protocol_version",
  "entrypoint",
  "capabilities",
  "resource",
  "extension",
  "artifact",
]);
const INFERENCE_TASKS = new Set(INFERENCE_TASK_VALUES);
const INFERENCE_EXTENSION_KEYS = new Set([
  "execution_mode",
  "models",
  "pools",
  "compute",
  "sdk_compatibility",
  "asset_manifest_sha256",
  "allowed_asset_sources",
  "managed_model_compatibility",
]);
const MODEL_KEYS = new Set([
  "id",
  "task",
  "is_interactive",
  "supported_prompts",
  "supported_inputs",
  "supported_geometric_outputs",
]);
const COMPUTE_KEYS = new Set(["device", "gpu_memory_mb"]);
const COMPAT_KEYS = new Set(["framework", "format", "runtime_device"]);

const PLUGIN_ID_RE = /^[a-z0-9]+(?:[._-][a-z0-9]+)*$/;
const OCI_DIGEST_RE =
  /^[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[0-9]+)?(?:\/[a-z0-9]+(?:[._-][a-z0-9]+)*)*(?::[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})?@sha256:[0-9a-f]{64}$/;
const SHA256_HEX_RE = /^[0-9a-f]{64}$/;
const ENTRYPOINT_HEAD_RE = /^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$/;
const NUMERIC_HOST_RE = /^(?:[0-9]+|0[xX][0-9a-fA-F]+)(?:\.(?:[0-9]+|0[xX][0-9a-fA-F]+)){0,3}$/;
const IPV4_RE = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/;
const SHELL_EXECUTABLES = new Set(["sh", "bash", "dash", "ash", "zsh", "ksh", "fish"]);

function assertPlainObject(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be a JSON object`);
  }
}

function assertAllowedKeys(obj, allowed, label) {
  for (const key of Object.keys(obj)) {
    if (!allowed.has(key)) {
      throw new Error(`${label} contains unexpected key: ${key}`);
    }
  }
}

function hasControlChars(value) {
  for (let i = 0; i < value.length; i += 1) {
    const code = value.charCodeAt(i);
    if (code < 0x20 || code === 0x7f) {
      return true;
    }
  }
  return false;
}

function validatePrintableText(value, maxLength, label) {
  if (typeof value !== "string") {
    throw new Error(`${label} must be a string`);
  }
  const cleaned = value.trim();
  if (!cleaned || cleaned.length > maxLength || hasControlChars(value)) {
    throw new Error(`${label} must be 1..${maxLength} printable characters`);
  }
}

function isForbiddenIpv4(octets) {
  const [a, b, c] = octets;
  if (a === 0) return true;
  if (a === 10) return true;
  if (a === 100 && b >= 64 && b <= 127) return true;
  if (a === 127) return true;
  if (a === 169 && b === 254) return true;
  if (a === 172 && b >= 16 && b <= 31) return true;
  if (a === 192 && b === 0 && c === 0) return true;
  if (a === 192 && b === 0 && c === 2) return true;
  if (a === 192 && b === 168) return true;
  if (a === 198 && (b === 18 || b === 19)) return true;
  if (a === 198 && b === 51 && c === 100) return true;
  if (a === 203 && b === 0 && c === 113) return true;
  if (a >= 224) return true;
  return false;
}

function parseIpv6Bytes(raw) {
  const ip = raw.startsWith("[") && raw.endsWith("]") ? raw.slice(1, -1) : raw;
  const parts = ip.split("::");
  if (parts.length > 2) return null;
  const parseSide = (side) => {
    if (!side) return [];
    const tokens = side.split(":");
    const words = [];
    for (let i = 0; i < tokens.length; i += 1) {
      const tok = tokens[i];
      if (tok.includes(".")) {
        if (i !== tokens.length - 1) return null;
        const m = IPV4_RE.exec(tok);
        if (!m) return null;
        const octs = m.slice(1).map((n) => Number(n));
        if (octs.some((n) => n > 255)) return null;
        words.push((octs[0] << 8) | octs[1], (octs[2] << 8) | octs[3]);
      } else {
        if (!/^[0-9a-fA-F]{1,4}$/.test(tok)) return null;
        words.push(parseInt(tok, 16));
      }
    }
    return words;
  };
  const left = parseSide(parts[0]);
  const right = parts.length === 2 ? parseSide(parts[1]) : [];
  if (left === null || right === null) return null;
  if (parts.length === 1 && left.length !== 8) return null;
  if (parts.length === 2 && left.length + right.length >= 8) return null;
  const zeros = parts.length === 2 ? new Array(8 - left.length - right.length).fill(0) : [];
  return [...left, ...zeros, ...right];
}

function isForbiddenPublicHost(hostname) {
  const lowered = hostname.toLowerCase().replace(/\.+$/, "");
  if (!lowered || lowered === "localhost" || lowered.endsWith(".localhost")) {
    return true;
  }
  const m4 = IPV4_RE.exec(lowered);
  if (m4) {
    const octets = m4.slice(1).map((n) => Number(n));
    if (octets.some((n) => n > 255)) return true;
    return isForbiddenIpv4(octets);
  }
  if (NUMERIC_HOST_RE.test(lowered)) {
    return true;
  }
  if (lowered.includes(":")) {
    const words = parseIpv6Bytes(lowered);
    if (!words) return true;
    const isAllZero = words.every((w) => w === 0);
    const isLoopback = words.slice(0, 7).every((w) => w === 0) && words[7] === 1;
    if (isAllZero || isLoopback) return true;
    if (
      words.slice(0, 5).every((w) => w === 0) &&
      words[5] === 0xffff
    ) {
      const octets = [
        (words[6] >> 8) & 0xff,
        words[6] & 0xff,
        (words[7] >> 8) & 0xff,
        words[7] & 0xff,
      ];
      return isForbiddenIpv4(octets);
    }
    if ((words[0] & 0xfe00) === 0xfc00) return true;
    if ((words[0] & 0xffc0) === 0xfe80) return true;
    if ((words[0] & 0xff00) === 0xff00) return true;
    if (words[0] === 0x2001 && words[1] === 0x0db8) return true;
    if (words[0] < 0x2000 || words[0] > 0x3fff) return true;
  }
  return false;
}

function validateHomepage(value) {
  if (value === undefined || value === null) {
    return;
  }
  if (typeof value !== "string" || !value || value !== value.trim() || value.length > 255) {
    throw new Error("homepage must be a valid https:// URL up to 255 ASCII chars");
  }
  for (let i = 0; i < value.length; i += 1) {
    const code = value.charCodeAt(i);
    if (code < 0x21 || code > 0x7e) {
      throw new Error("homepage must be a valid https:// URL up to 255 ASCII chars");
    }
  }
  if (!value.startsWith("https://")) {
    throw new Error("homepage must be a public https:// URL");
  }
  const authority = value.slice("https://".length).split(/[/?#]/, 1)[0];
  if (!authority || authority.includes("@") || authority.endsWith(":")) {
    throw new Error("homepage must be a public https:// URL without userinfo or empty port");
  }
  let parsed;
  try {
    parsed = new URL(value);
  } catch {
    throw new Error("homepage must be a valid https:// URL");
  }
  if (parsed.protocol !== "https:" || !parsed.hostname || parsed.username || parsed.password) {
    throw new Error("homepage must be a public https:// URL");
  }
  if (parsed.port) {
    const portNum = Number(parsed.port);
    if (!Number.isInteger(portNum) || portNum < 1 || portNum > 65535) {
      throw new Error("homepage port out of range");
    }
  }
  if (isForbiddenPublicHost(parsed.hostname)) {
    throw new Error("homepage must be a public https:// URL");
  }
}

function validateEntrypoint(entrypoint, point) {
  if (!Array.isArray(entrypoint) || entrypoint.length === 0 || entrypoint.length > 64) {
    throw new Error(`${point} manifest requires entrypoint with 1..64 arguments`);
  }
  for (const arg of entrypoint) {
    if (typeof arg !== "string" || !arg || !arg.trim() || /[\x00\r\n]/.test(arg)) {
      throw new Error("entrypoint arguments must be non-empty single-line strings");
    }
  }
  const head = entrypoint[0];
  if (!ENTRYPOINT_HEAD_RE.test(head) || SHELL_EXECUTABLES.has(head.toLowerCase())) {
    throw new Error("entrypoint must not invoke a shell");
  }
  if (
    head.toLowerCase().startsWith("python") &&
    entrypoint
      .slice(1)
      .some((arg) => arg === "-c" || (arg.startsWith("-") && !arg.startsWith("--") && arg.slice(1).includes("c")))
  ) {
    throw new Error("entrypoint must not invoke python -c");
  }
}

function validateResource(resource, point) {
  if (
    !resource ||
    typeof resource !== "object" ||
    Array.isArray(resource) ||
    Object.keys(resource).some((k) => k !== "memory_mb") ||
    !Number.isInteger(resource.memory_mb) ||
    resource.memory_mb < 16 ||
    resource.memory_mb > 65536
  ) {
    throw new Error(`${point} manifest requires resource.memory_mb in 16..65536`);
  }
}

function validateCanonicalUniqueStrings(values, label, { minLength = 0 } = {}) {
  if (!Array.isArray(values) || values.length < minLength) {
    throw new Error(`${label} must be an array with at least ${minLength} item(s)`);
  }
  const seen = new Set();
  for (const item of values) {
    if (typeof item !== "string" || !item || item.trim() !== item) {
      throw new Error(`${label} must contain nonempty canonical strings`);
    }
    if (seen.has(item)) {
      throw new Error(`${label} must not contain duplicates`);
    }
    seen.add(item);
  }
}

function validateInferenceExtension(ext, manifest, seenCaps) {
  assertPlainObject(ext, "inference extension");
  assertAllowedKeys(ext, INFERENCE_EXTENSION_KEYS, "inference extension");

  const executionMode = ext.execution_mode === undefined ? "container" : ext.execution_mode;
  if (executionMode !== "container" && executionMode !== "direct") {
    throw new Error(`unsupported inference execution_mode: ${String(executionMode)}`);
  }

  if (!Array.isArray(ext.models) || ext.models.length === 0 || ext.models.length > 128) {
    throw new Error("inference extension.models must contain 1..128 entries");
  }
  const modelIds = new Set();
  for (const model of ext.models) {
    assertPlainObject(model, "inference model");
    assertAllowedKeys(model, MODEL_KEYS, "inference model");
    if (typeof model.id !== "string" || !model.id || model.id.length > 128 || model.id.trim() !== model.id) {
      throw new Error("model.id must be a canonical string (1..128)");
    }
    if (modelIds.has(model.id)) {
      throw new Error(`duplicate model id: ${model.id}`);
    }
    modelIds.add(model.id);
    if (typeof model.task !== "string" || !INFERENCE_TASKS.has(model.task)) {
      throw new Error(`unsupported inference model task: ${String(model.task)}`);
    }
    if (typeof model.is_interactive !== "boolean") {
      throw new Error("model.is_interactive must be a boolean");
    }
    validateCanonicalUniqueStrings(model.supported_prompts, "model.supported_prompts", { minLength: 1 });
    validateCanonicalUniqueStrings(model.supported_inputs, "model.supported_inputs", { minLength: 1 });
    validateCanonicalUniqueStrings(model.supported_geometric_outputs, "model.supported_geometric_outputs");
  }

  assertPlainObject(ext.pools, "inference extension.pools");
  const poolKeys = Object.keys(ext.pools);
  if (poolKeys.length === 0 || poolKeys.length > 128) {
    throw new Error("inference extension.pools must contain 1..128 pools");
  }
  for (const poolId of poolKeys) {
    if (!poolId || poolId.length > 128 || poolId.trim() !== poolId) {
      throw new Error("pool ids must be nonempty and canonical");
    }
    const pool = ext.pools[poolId];
    assertPlainObject(pool, `pool ${poolId}`);
    if (Object.keys(pool).some((k) => k !== "cap") || !Number.isInteger(pool.cap) || pool.cap <= 0) {
      throw new Error(`pool ${poolId} requires positive integer cap`);
    }
  }

  assertPlainObject(ext.compute, "inference extension.compute");
  assertAllowedKeys(ext.compute, COMPUTE_KEYS, "inference extension.compute");
  if (ext.compute.device !== "cpu" && ext.compute.device !== "cuda") {
    throw new Error("inference extension.compute.device must be 'cpu' or 'cuda'");
  }
  const gpuMemoryMb = ext.compute.gpu_memory_mb === undefined ? 0 : ext.compute.gpu_memory_mb;
  if (!Number.isInteger(gpuMemoryMb) || gpuMemoryMb < 0) {
    throw new Error("inference extension.compute.gpu_memory_mb must be a non-negative integer");
  }
  if (ext.compute.device === "cpu" && gpuMemoryMb !== 0) {
    throw new Error("CPU compute must have gpu_memory_mb == 0");
  }

  if (ext.sdk_compatibility !== undefined && ext.sdk_compatibility !== null) {
    if (typeof ext.sdk_compatibility !== "string" || !ext.sdk_compatibility.trim() || ext.sdk_compatibility.length > 255) {
      throw new Error("sdk_compatibility must be a nonempty string up to 255 chars");
    }
  }
  if (ext.asset_manifest_sha256 !== undefined && ext.asset_manifest_sha256 !== null) {
    if (typeof ext.asset_manifest_sha256 !== "string" || !SHA256_HEX_RE.test(ext.asset_manifest_sha256)) {
      throw new Error("asset_manifest_sha256 must be a 64-char lowercase hex sha256");
    }
  }
  if (ext.allowed_asset_sources !== undefined) {
    if (!Array.isArray(ext.allowed_asset_sources)) {
      throw new Error("allowed_asset_sources must be an array");
    }
    for (const src of ext.allowed_asset_sources) {
      if (typeof src !== "string" || !src.trim()) {
        throw new Error("allowed_asset_sources must not contain blank entries");
      }
    }
  }
  const compat = ext.managed_model_compatibility === undefined ? [] : ext.managed_model_compatibility;
  if (!Array.isArray(compat)) {
    throw new Error("managed_model_compatibility must be an array");
  }
  for (const item of compat) {
    assertPlainObject(item, "managed_model_compatibility entry");
    assertAllowedKeys(item, COMPAT_KEYS, "managed_model_compatibility entry");
    if (
      typeof item.framework !== "string" ||
      !item.framework ||
      item.framework.length > 64 ||
      item.framework.trim() !== item.framework
    ) {
      throw new Error("managed_model_compatibility.framework must be canonical");
    }
    if (!Object.prototype.hasOwnProperty.call(item, "format") || item.format === undefined) {
      throw new Error("managed_model_compatibility.format is required (use null for native checkpoints)");
    }
    if (
      item.format !== null &&
      (typeof item.format !== "string" || !item.format || item.format.length > 64 || item.format.trim() !== item.format)
    ) {
      throw new Error("managed_model_compatibility.format must be null or canonical string");
    }
    if (item.runtime_device !== "cpu" && item.runtime_device !== "cuda") {
      throw new Error("managed_model_compatibility.runtime_device must be 'cpu' or 'cuda'");
    }
    if (item.runtime_device !== ext.compute.device) {
      throw new Error("managed_model_compatibility.runtime_device must match compute.device");
    }
  }

  if (!seenCaps.has("inference")) {
    throw new Error("inference manifest must include 'inference' capability");
  }
  if (seenCaps.has("managed_models") !== (compat.length > 0)) {
    throw new Error("managed_models capability and managed_model_compatibility declarations must agree");
  }
  if (seenCaps.has("managed_evaluation") && !seenCaps.has("managed_models")) {
    throw new Error("managed_evaluation capability requires managed_models capability");
  }

  if (executionMode === "container") {
    validateEntrypoint(manifest.entrypoint, "container inference");
    validateResource(manifest.resource, "container inference");
    if (typeof ext.asset_manifest_sha256 !== "string") {
      throw new Error("container inference manifest requires extension.asset_manifest_sha256");
    }
    if (ext.compute.device === "cuda" && gpuMemoryMb <= 0) {
      throw new Error("container CUDA inference manifest requires extension.compute.gpu_memory_mb > 0");
    }
  } else {
    if (manifest.artifact !== undefined && manifest.artifact !== null) {
      throw new Error("direct inference manifest must not carry an OCI artifact");
    }
    if (manifest.entrypoint !== undefined && manifest.entrypoint !== null) {
      validateEntrypoint(manifest.entrypoint, "direct inference");
    }
    if (manifest.resource !== undefined && manifest.resource !== null) {
      validateResource(manifest.resource, "direct inference");
    }
  }
}

export function validateManifest(manifest) {
  assertPlainObject(manifest, "manifest");
  assertAllowedKeys(manifest, TOP_LEVEL_MANIFEST_KEYS, "manifest");
  if (manifest.manifest_schema !== EXTENSION_MANIFEST_SCHEMA) {
    throw new Error(`manifest_schema must be ${EXTENSION_MANIFEST_SCHEMA}`);
  }
  const point = manifest.extension_point;
  if (!Object.prototype.hasOwnProperty.call(PROTOCOL_BY_POINT, point)) {
    throw new Error(`unsupported extension_point: ${String(point)}`);
  }
  if (
    typeof manifest.plugin_id !== "string" ||
    manifest.plugin_id.length > 128 ||
    !PLUGIN_ID_RE.test(manifest.plugin_id)
  ) {
    throw new Error("plugin_id must match ^[a-z0-9]+(?:[._-][a-z0-9]+)*$ (1..128)");
  }
  validatePrintableText(manifest.version, 128, "version");
  validatePrintableText(manifest.publisher, 255, "publisher");
  validatePrintableText(manifest.license, 255, "license");
  if (manifest.display_name !== undefined && manifest.display_name !== null) {
    validatePrintableText(manifest.display_name, 64, "display_name");
  }
  if (manifest.description !== undefined && manifest.description !== null) {
    validatePrintableText(manifest.description, 280, "description");
  }
  validateHomepage(manifest.homepage);
  if (manifest.protocol_version !== PROTOCOL_BY_POINT[point]) {
    throw new Error(
      `extension_point ${point} requires protocol_version ${PROTOCOL_BY_POINT[point]}`,
    );
  }
  if (manifest.artifact !== undefined && manifest.artifact !== null) {
    if (typeof manifest.artifact !== "string" || !OCI_DIGEST_RE.test(manifest.artifact)) {
      throw new Error("artifact must be a digest-pinned OCI reference (@sha256:...)");
    }
  }
  if (
    !Array.isArray(manifest.capabilities) ||
    manifest.capabilities.length === 0 ||
    manifest.capabilities.length > 16
  ) {
    throw new Error("capabilities must be an array with 1..16 entries");
  }
  const allowedCaps = CAPABILITIES_BY_POINT[point];
  const seenCaps = new Set();
  for (const cap of manifest.capabilities) {
    if (typeof cap !== "string" || !allowedCaps.has(cap)) {
      throw new Error(`unsupported capability ${String(cap)} for ${point}`);
    }
    if (seenCaps.has(cap)) {
      throw new Error(`duplicate capability: ${cap}`);
    }
    seenCaps.add(cap);
  }
  if (manifest.extension !== undefined) {
    assertPlainObject(manifest.extension, "extension");
    if (Object.prototype.hasOwnProperty.call(manifest.extension, "source_tier")) {
      throw new Error("manifest must not self-declare source_tier");
    }
  }
  if (point === "inference") {
    validateInferenceExtension(manifest.extension, manifest, seenCaps);
  } else if (point === "theme") {
    if (manifest.entrypoint !== undefined && manifest.entrypoint !== null) {
      validateEntrypoint(manifest.entrypoint, "theme");
    }
    if (manifest.resource !== undefined && manifest.resource !== null) {
      validateResource(manifest.resource, "theme");
    }
  } else {
    validateEntrypoint(manifest.entrypoint, point);
    validateResource(manifest.resource, point);
  }
  return manifest;
}
