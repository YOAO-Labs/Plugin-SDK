import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  EXTENSION_MANIFEST_SCHEMA,
  readSecret,
  validateManifest,
  validateThemeTokens,
  verifyBearerToken,
} from "../src/index.mjs";

test("validateManifest accepts theme and format_adapter manifests and rejects shell/source_tier", () => {
  const themeManifest = validateManifest({
    manifest_schema: EXTENSION_MANIFEST_SCHEMA,
    extension_point: "theme",
    plugin_id: "yoao.theme.midnight",
    version: "1.0.0",
    display_name: "Midnight",
    publisher: "YOAO Official",
    license: "MIT",
    protocol_version: "yoao-theme/v1",
    capabilities: ["dark"],
    extension: { theme_id: "midnight", tokens_file: "tokens.json" },
  });
  assert.equal(themeManifest.extension_point, "theme");

  assert.throws(
    () =>
      validateManifest({
        manifest_schema: EXTENSION_MANIFEST_SCHEMA,
        extension_point: "format_adapter",
        plugin_id: "bad.shell",
        version: "1.0.0",
        publisher: "Test",
        license: "MIT",
        protocol_version: "yoao-format-adapter/v1",
        entrypoint: ["sh", "-c", "id"],
        capabilities: ["import"],
        resource: { memory_mb: 128 },
        extension: {},
      }),
    /shell/,
  );

  assert.throws(
    () =>
      validateManifest({
        manifest_schema: EXTENSION_MANIFEST_SCHEMA,
        extension_point: "theme",
        plugin_id: "spoof.tier",
        version: "1.0.0",
        publisher: "Test",
        license: "MIT",
        protocol_version: "yoao-theme/v1",
        capabilities: ["dark"],
        extension: { source_tier: "official" },
      }),
    /source_tier/,
  );
});

test("validateThemeTokens enforces --sc-* semantic tokens and blocks CSS injection", () => {
  const ok = validateThemeTokens({
    light: { "--sc-bg": "#ffffff", "--sc-accent": "oklch(0.62 0.21 255)" },
    dark: { "--sc-bg": "#0b1120", "--sc-radius": "6px" },
  });
  assert.equal(ok.dark["--sc-bg"], "#0b1120");

  assert.throws(
    () => validateThemeTokens({ dark: { "--color-bg": "#000" } }),
    /must start with --sc-/,
  );
  assert.throws(
    () => validateThemeTokens({ dark: { "--sc-bg": "url(https://evil.example/x)" } }),
    /forbidden CSS construct/,
  );
});

test("auth helpers validate bearer tokens and *_FILE secrets", () => {
  const tmpFile = path.join(os.tmpdir(), `yoao-sdk-secret-${Date.now()}.txt`);
  fs.writeFileSync(tmpFile, "  tok-123  \n", "utf8");
  try {
    assert.equal(readSecret({ TOK_FILE: tmpFile }, "TOK", "TOK_FILE"), "tok-123");
    assert.equal(verifyBearerToken("Bearer tok-123", "tok-123"), true);
    assert.equal(verifyBearerToken("Bearer wrong", "tok-123"), false);
  } finally {
    fs.unlinkSync(tmpFile);
  }
});
