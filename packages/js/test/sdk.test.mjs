import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  EXTENSION_MANIFEST_SCHEMA,
  PREVIEW_MAX_POOL_BYTES,
  PREVIEW_MAX_SOURCE_BYTES,
  SHA256_PATTERN,
  TRAINER_PROTOCOL_VERSION,
  canonicalJsonDigest,
  readSecret,
  validateConfigResult,
  validateManifest,
  validatePreviewReport,
  validatePreviewSpec,
  validateServingCompatibility,
  validateThemeTokens,
  validateTrainEvent,
  validateTrainReceipt,
  validateTrainSpec,
  validateTrainerCapabilities,
  verifyBearerToken,
} from "../src/index.mjs";

const DIGEST_A = "a".repeat(64);
const DIGEST_B = "b".repeat(64);
const DIGEST_C = "c".repeat(64);
const DIGEST_D = "d".repeat(64);
const DIGEST_E = "e".repeat(64);

test("validateManifest accepts theme and trainer manifests and rejects shell/source_tier", () => {
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

  const trainerManifest = validateManifest({
    manifest_schema: EXTENSION_MANIFEST_SCHEMA,
    extension_point: "trainer",
    plugin_id: "yoao.trainer.example",
    version: "1.0.0",
    display_name: "Example Trainer",
    publisher: "YOAO Official",
    license: "MIT",
    protocol_version: TRAINER_PROTOCOL_VERSION,
    entrypoint: ["yoao-trainer-example"],
    capabilities: ["train", "evaluate", "preview", "export_weights"],
    resource: { memory_mb: 8192 },
    extension: { framework_id: "example" },
  });
  assert.equal(trainerManifest.extension_point, "trainer");

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
        extension_point: "format_adapter",
        plugin_id: "bad.python",
        version: "1.0.0",
        publisher: "Test",
        license: "MIT",
        protocol_version: "yoao-format-adapter/v1",
        entrypoint: ["python3", "-c", "print(1)"],
        capabilities: ["import"],
        resource: { memory_mb: 128 },
        extension: {},
      }),
    /python -c/,
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

test("yoao-trainer/v1 validators and canonicalJsonDigest enforce frozen contract invariants", () => {
  assert.equal(TRAINER_PROTOCOL_VERSION, "yoao-trainer/v1");
  assert.equal(SHA256_PATTERN, "^[0-9a-f]{64}$");

  const compat = validateServingCompatibility({
    framework: "example-trainer",
    format: null,
    training_runtime: { python: "3.12" },
    compatible_serving_runtimes: [{ device: "cpu" }],
  });
  assert.equal(compat.format, null);
  assert.throws(() => validateServingCompatibility({ framework: "demo" }), /format/);
  assert.throws(() => validateServingCompatibility({ framework: "demo", format: "" }), /format/);
  assert.throws(
    () => validateServingCompatibility({ framework: "demo", format: null, extra: 1 }),
    /unexpected field/,
  );

  const caps = validateTrainerCapabilities({
    protocol_version: TRAINER_PROTOCOL_VERSION,
    framework_id: "example-trainer",
    framework_version: "1.0.0",
    runtime_versions: { python: "3.12" },
    supported_tasks: ["detect"],
    supported_dataset_contracts: ["bbox-v1"],
    architectures: [
      {
        id: "demo-n",
        family: "demo",
        tasks: ["detect"],
        base_asset_filename: "demo-n.pt",
        base_asset_sha256: DIGEST_A,
      },
    ],
    policy_digests: { budget: DIGEST_B },
    serving_compatibility: [compat],
  });
  assert.equal(caps.framework_id, "example-trainer");
  assert.throws(
    () =>
      validateTrainerCapabilities({
        ...caps,
        architectures: [{ id: "demo-n", family: "demo", tasks: ["segment"] }],
      }),
    /unsupported task/,
  );

  assert.equal(
    validateConfigResult({
      protocol_version: TRAINER_PROTOCOL_VERSION,
      valid: true,
      schema_id: "example-1",
      schema_digest: DIGEST_A,
      effective_values: { epochs: 10 },
    }).valid,
    true,
  );
  assert.throws(
    () => validateConfigResult({ protocol_version: TRAINER_PROTOCOL_VERSION, valid: false }),
    /at least one validation issue/,
  );
  assert.throws(
    () =>
      validateConfigResult({
        protocol_version: TRAINER_PROTOCOL_VERSION,
        valid: true,
        schema_id: "example-1",
        schema_digest: DIGEST_A,
        issues: null,
      }),
    /issues must be an array/,
  );

  const spec = validateTrainSpec({
    protocol_version: TRAINER_PROTOCOL_VERSION,
    run_id: "run-1",
    attempt_id: "attempt-1",
    framework_id: "example-trainer",
    task: "detect",
    architecture_id: "demo-n",
    dataset_contract: "bbox-v1",
    dataset_dir: "/work/input/dataset",
    dataset_snapshot_digest: DIGEST_A,
    config_digest: DIGEST_B,
    classes_digest: DIGEST_C,
  });
  assert.equal(spec.run_id, "run-1");
  assert.throws(
    () =>
      validateTrainSpec({
        ...spec,
        resume_checkpoint_path: "/work/input/a.pt",
        init_weights_path: "/work/input/b.pt",
      }),
    /mutually exclusive/,
  );

  assert.equal(
    validateTrainEvent({ seq: 1, kind: "epoch", epoch: 1, total_epochs: 10, progress: 0.1 }).seq,
    1,
  );
  assert.throws(
    () => validateTrainEvent({ seq: 1, kind: "epoch", epoch: 11, total_epochs: 10 }),
    /epoch cannot exceed/,
  );
  assert.throws(
    () => validateTrainEvent({ seq: 1, kind: "phase", metrics: new Date() }),
    /metrics must be an object/,
  );

  const receipt = validateTrainReceipt({
    protocol_version: TRAINER_PROTOCOL_VERSION,
    status: "succeeded",
    artifacts: [{ role: "weights", filename: "best.pt", sha256: DIGEST_A, size_bytes: 100 }],
    serving_compatibility: compat,
  });
  assert.equal(receipt.status, "succeeded");
  assert.throws(
    () => validateTrainReceipt({ protocol_version: TRAINER_PROTOCOL_VERSION, status: "failed" }),
    /requires failure_code/,
  );
  assert.throws(
    () =>
      validateTrainReceipt({
        protocol_version: TRAINER_PROTOCOL_VERSION,
        status: "cancelled",
        artifacts: null,
      }),
    /artifacts must be an array/,
  );
  assert.throws(
    () =>
      validateTrainReceipt({
        protocol_version: TRAINER_PROTOCOL_VERSION,
        status: "succeeded",
        artifacts: [{ role: "weights", filename: "../bad.pt", sha256: DIGEST_A, size_bytes: 1 }],
      }),
    /safe filename/,
  );

  const previewSpec = validatePreviewSpec({
    protocol_version: TRAINER_PROTOCOL_VERSION,
    preview_id: "preview-1",
    framework_id: "example-trainer",
    task: "detect",
    architecture_id: "demo-n",
    input_digest: DIGEST_E,
    config_digest: DIGEST_A,
    dataset_snapshot_digest: DIGEST_B,
    policy_digest: DIGEST_C,
    seed: 42,
    epoch: 1,
    count: 2,
    sources: [
      {
        task_id: "task-1",
        filename: "images/a.jpg",
        sha256: DIGEST_D,
        width: 640,
        height: 480,
        size_bytes: 1024,
        box_count: 2,
        label_path: "labels/a.txt",
        label_sha256: DIGEST_A,
      },
    ],
  });
  assert.equal(previewSpec.input_digest, DIGEST_E);
  assert.throws(
    () =>
      validatePreviewSpec({
        ...previewSpec,
        sources: Array.from(
          { length: PREVIEW_MAX_POOL_BYTES / PREVIEW_MAX_SOURCE_BYTES + 1 },
          (_, idx) => ({
            task_id: `t-${idx}`,
            filename: `images/${idx}.jpg`,
            sha256: DIGEST_D,
            width: 100,
            height: 100,
            size_bytes: PREVIEW_MAX_SOURCE_BYTES,
            box_count: 0,
          }),
        ),
      }),
    /pool bytes exceed/,
  );

  assert.equal(
    canonicalJsonDigest({ b: 1, a: { y: null, x: "测试" }, c: [0.25, true], z: -0 }),
    canonicalJsonDigest({ z: 0, c: [0.25, true], a: { x: "测试", y: null }, b: 1 }),
  );
  assert.throws(() => canonicalJsonDigest({ bad: Number.NaN }), /NaN/);
  assert.throws(() => canonicalJsonDigest(new Date()), /plain JSON object/);
  assert.throws(() => canonicalJsonDigest({ bad: new Map() }), /unsupported value type/);
  const sparse = [];
  sparse[1] = 1;
  assert.throws(() => canonicalJsonDigest({ arr: sparse }), /sparse array hole/);

  const report = validatePreviewReport({
    protocol_version: TRAINER_PROTOCOL_VERSION,
    status: "succeeded",
    input_digest: previewSpec.input_digest,
    policy_digest: DIGEST_C,
    cuda_initialized: false,
    outputs: [
      {
        filename: "preview-0.png",
        sha256: DIGEST_A,
        size_bytes: 2048,
        width: 640,
        height: 640,
        box_count: 2,
        metadata: { clipped: 0 },
      },
    ],
    runtime: { cpu_only: true },
    details: { mosaic_closed: false },
  });
  assert.equal(report.cuda_initialized, false);
  assert.throws(
    () => validatePreviewReport({ ...report, cuda_initialized: true }),
    /must not initialize CUDA/,
  );
  assert.throws(
    () => validatePreviewReport({ ...report, cuda_initialized: null }),
    /cuda_initialized must be a boolean/,
  );
  assert.throws(
    () =>
      validatePreviewReport({
        protocol_version: TRAINER_PROTOCOL_VERSION,
        status: "failed",
        input_digest: previewSpec.input_digest,
        policy_digest: DIGEST_C,
        failure_code: "err",
        outputs: null,
      }),
    /outputs must be an array/,
  );
});
