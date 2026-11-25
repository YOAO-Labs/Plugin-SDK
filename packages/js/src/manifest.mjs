export const EXTENSION_MANIFEST_SCHEMA = "yoao-extension-manifest/v1";

const PROTOCOL_BY_POINT = Object.freeze({
  inference: "yoao-inference/v2",
  notification_channel: "yoao-notification-channel/v1",
  format_adapter: "yoao-format-adapter/v1",
  theme: "yoao-theme/v1",
});

const CAPABILITIES_BY_POINT = Object.freeze({
  inference: new Set(["predict", "train", "track", "health"]),
  notification_channel: new Set(["deliver", "verify_address"]),
  format_adapter: new Set(["import", "export"]),
  theme: new Set(["light", "dark"]),
});

const PLUGIN_ID_RE = /^[a-z0-9]+(?:[._-][a-z0-9]+)*$/;
const ENTRYPOINT_HEAD_RE = /^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$/;
const SHELL_EXECUTABLES = new Set(["sh", "bash", "dash", "ash", "zsh", "ksh", "fish"]);

export function validateManifest(manifest) {
  if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)) {
    throw new Error("manifest must be a JSON object");
  }
  if (manifest.manifest_schema !== EXTENSION_MANIFEST_SCHEMA) {
    throw new Error(`manifest_schema must be ${EXTENSION_MANIFEST_SCHEMA}`);
  }
  const point = manifest.extension_point;
  if (!Object.prototype.hasOwnProperty.call(PROTOCOL_BY_POINT, point)) {
    throw new Error(`unsupported extension_point: ${String(point)}`);
  }
  if (typeof manifest.plugin_id !== "string" || !PLUGIN_ID_RE.test(manifest.plugin_id)) {
    throw new Error("plugin_id must match ^[a-z0-9]+(?:[._-][a-z0-9]+)*$");
  }
  if (typeof manifest.version !== "string" || !manifest.version.trim()) {
    throw new Error("version is required");
  }
  if (typeof manifest.publisher !== "string" || !manifest.publisher.trim()) {
    throw new Error("publisher is required");
  }
  if (typeof manifest.license !== "string" || !manifest.license.trim()) {
    throw new Error("license is required");
  }
  if (manifest.protocol_version !== PROTOCOL_BY_POINT[point]) {
    throw new Error(
      `extension_point ${point} requires protocol_version ${PROTOCOL_BY_POINT[point]}`,
    );
  }
  if (!Array.isArray(manifest.capabilities) || manifest.capabilities.length === 0) {
    throw new Error("capabilities must be a non-empty array");
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
  if (manifest.extension && typeof manifest.extension === "object") {
    if (Object.prototype.hasOwnProperty.call(manifest.extension, "source_tier")) {
      throw new Error("manifest must not self-declare source_tier");
    }
  }
  if (point !== "theme") {
    if (!Array.isArray(manifest.entrypoint) || manifest.entrypoint.length === 0) {
      throw new Error(`${point} manifest requires entrypoint`);
    }
    const head = manifest.entrypoint[0];
    if (
      typeof head !== "string" ||
      !ENTRYPOINT_HEAD_RE.test(head) ||
      SHELL_EXECUTABLES.has(head.toLowerCase())
    ) {
      throw new Error("entrypoint must not invoke a shell");
    }
    if (
      !manifest.resource ||
      typeof manifest.resource !== "object" ||
      !Number.isInteger(manifest.resource.memory_mb) ||
      manifest.resource.memory_mb < 16
    ) {
      throw new Error(`${point} manifest requires resource.memory_mb >= 16`);
    }
  }
  return manifest;
}
