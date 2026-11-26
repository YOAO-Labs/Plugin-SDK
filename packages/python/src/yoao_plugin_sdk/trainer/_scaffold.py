"""Pure-stdlib scaffolding for the runnable CPU trainer example plugin."""

from __future__ import annotations

import json
from pathlib import Path
import re

from yoao_plugin_sdk.manifest import EXTENSION_MANIFEST_SCHEMA


def _slugify(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9-]+", "-", name.strip().lower().replace("_", "-")).strip("-")
    return cleaned or "demo-trainer"


def init_trainer_example(directory: Path) -> None:
    slug = _slugify(directory.name)
    framework_id = slug.replace("-", "_")
    directory.mkdir(parents=True, exist_ok=False)
    tests_dir = directory / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "manifest_schema": EXTENSION_MANIFEST_SCHEMA,
        "extension_point": "trainer",
        "plugin_id": f"custom.trainer.{slug}",
        "version": "0.1.0",
        "display_name": directory.name,
        "description": f"Runnable pure-Python CPU trainer example plugin ({directory.name}).",
        "publisher": "Custom Publisher",
        "license": "MIT",
        "protocol_version": "yoao-trainer/v1",
        "entrypoint": [slug],
        "capabilities": ["train", "evaluate", "preview", "export_weights"],
        "resource": {"memory_mb": 512},
        "extension": {
            "framework_id": framework_id,
            "framework_version": "0.1.0",
            "supported_tasks": ["detect"],
            "supported_dataset_contracts": ["ultralytics-bbox-1"],
            "requires_gpu": False,
            "preview_cpu_only": True,
            "allowed_egress_hosts": [],
        },
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    (directory / "pyproject.toml").write_text(
        f"""[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{slug}"
version = "0.1.0"
description = "Runnable pure-Python CPU trainer example for yoao-trainer/v1."
requires-python = ">=3.10"
dependencies = [
    "yoao-plugin-sdk[trainer]>=0.1.0",
]

[project.scripts]
{slug} = "demo_trainer:main"

[tool.setuptools]
py-modules = ["demo_trainer"]
""",
        encoding="utf-8",
    )

    (directory / "demo_trainer.py").write_text(
        f'''"""Runnable pure-Python CPU trainer example implementing yoao-trainer/v1."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import struct
import time
from typing import Any
import zlib

from yoao_plugin_sdk.trainer import (
    POLICY_DIGEST,
    TRAINER_PROTOCOL_VERSION,
    CheckpointSlotStore,
    EventEmitter,
    InputMembership,
    NativeConfigurationMismatch,
    PreviewOutputV1,
    PreviewReportV1,
    PreviewSpecV1,
    ServingCompatibilityV1,
    TrainerArchitectureV1,
    TrainerCapabilitiesV1,
    TrainReceiptV1,
    TrainSpecV1,
    ValidateConfigResultV1,
    ValidationIssueV1,
    atomic_write_json,
    build_artifact_entry,
    canonical_json_bytes,
    canonical_json_digest,
    digest_file,
    hash_source_files,
    load_dataset_identity,
    metrics_of,
    run_trainer_cli,
    validate_frozen_manifest_core,
    validate_sample_files,
    validate_value,
    validate_yolo_bbox_label,
)

FRAMEWORK_ID = {framework_id!r}
FRAMEWORK_VERSION = "0.1.0"
SUPPORTED_TASKS = ("detect",)
SUPPORTED_DATASET_CONTRACTS = ("ultralytics-bbox-1",)
ARCHITECTURE_ID = "cpu-demo-n"
SCHEMA_ID = f"{{FRAMEWORK_ID}}-detect-1"
PREVIEW_POLICY_DIGEST = canonical_json_digest({{"revision": 1, "purpose": "cpu-demo-preview"}})

PARAMETER_RULES: dict[str, dict[str, Any]] = {{
    "epochs": {{"type": "integer", "minimum": 1, "maximum": 100}},
    "seed": {{"type": "integer", "minimum": 0, "maximum": 2147483647}},
}}
PARAMETER_DEFAULTS: dict[str, Any] = {{
    "epochs": 2,
    "seed": 42,
}}
PARAMETER_SCHEMA: dict[str, Any] = {{
    "schema_id": SCHEMA_ID,
    "fields": [
        {{"key": key, "status": "editable", "default": PARAMETER_DEFAULTS[key], "rules": rule}}
        for key, rule in PARAMETER_RULES.items()
    ],
}}
SCHEMA_DIGEST = canonical_json_digest(PARAMETER_SCHEMA)


def _encode_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    raw = bytearray()
    row = bytes([0]) + bytes(rgb) * width
    for _ in range(height):
        raw.extend(row)

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\\x89PNG\\r\\n\\x1a\\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(bytes(raw), level=1))
        + _chunk(b"IEND", b"")
    )


class DemoTrainerPlugin:
    """Zero-ML-dependency CPU reference trainer for yoao-trainer/v1."""

    def describe(self) -> TrainerCapabilitiesV1:
        return TrainerCapabilitiesV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            framework_id=FRAMEWORK_ID,
            framework_version=FRAMEWORK_VERSION,
            runtime_versions={{"python": platform.python_version()}},
            supported_tasks=list(SUPPORTED_TASKS),
            supported_dataset_contracts=list(SUPPORTED_DATASET_CONTRACTS),
            architectures=[
                TrainerArchitectureV1(
                    id=ARCHITECTURE_ID,
                    family="cpu-demo",
                    tasks=["detect"],
                )
            ],
            parameter_schema={{**PARAMETER_SCHEMA, "schema_digest": SCHEMA_DIGEST}},
            evaluator_descriptors={{
                "detect": {{
                    "framework": FRAMEWORK_ID,
                    "version": FRAMEWORK_VERSION,
                    "task": "detect",
                    "source_sha256": hash_source_files([Path(__file__).resolve()]),
                }}
            }},
            policy_digests={{
                "input_membership": POLICY_DIGEST,
                "preview": PREVIEW_POLICY_DIGEST,
            }},
            serving_compatibility=[
                ServingCompatibilityV1(
                    framework=FRAMEWORK_ID,
                    format="json",
                    training_runtime={{"python": platform.python_version(), "device": "cpu"}},
                    compatible_serving_runtimes=[{{"device": "cpu", "framework": FRAMEWORK_ID}}],
                )
            ],
        )

    def validate_config(self, payload: Any) -> ValidateConfigResultV1:
        if not isinstance(payload, dict):
            return ValidateConfigResultV1(
                protocol_version=TRAINER_PROTOCOL_VERSION,
                valid=False,
                schema_id=SCHEMA_ID,
                schema_digest=SCHEMA_DIGEST,
                issues=[
                    ValidationIssueV1(
                        code="invalid_spec", message="Config payload must be a JSON object"
                    )
                ],
            )
        issues: list[ValidationIssueV1] = []
        framework_id = payload.get("framework_id")
        if framework_id is not None and framework_id != FRAMEWORK_ID:
            issues.append(
                ValidationIssueV1(
                    code="unsupported_framework",
                    field="framework_id",
                    message=f"Expected framework_id {{FRAMEWORK_ID!r}}",
                )
            )
        task = payload.get("task")
        if task not in SUPPORTED_TASKS:
            issues.append(
                ValidationIssueV1(
                    code="unsupported_task",
                    field="task",
                    message=f"Unsupported task {{task!r}}",
                )
            )
        arch = payload.get("architecture_id")
        if arch != ARCHITECTURE_ID:
            issues.append(
                ValidationIssueV1(
                    code="unsupported_architecture",
                    field="architecture_id",
                    message=f"Unsupported architecture {{arch!r}}",
                )
            )
        contract = payload.get("dataset_contract")
        if contract not in SUPPORTED_DATASET_CONTRACTS:
            issues.append(
                ValidationIssueV1(
                    code="unsupported_dataset_contract",
                    field="dataset_contract",
                    message=f"Unsupported dataset_contract {{contract!r}}",
                )
            )
        submitted = payload.get("parameters", {{}})
        if not isinstance(submitted, dict):
            issues.append(
                ValidationIssueV1(
                    code="invalid_parameters",
                    field="parameters",
                    message="parameters must be a JSON object",
                )
            )
            submitted = {{}}
        effective = dict(PARAMETER_DEFAULTS)
        origins = {{key: "default" for key in PARAMETER_DEFAULTS}}
        for key, value in submitted.items():
            if key not in PARAMETER_RULES:
                issues.append(
                    ValidationIssueV1(
                        code="unsupported_parameter",
                        field=str(key),
                        message=f"Unknown parameter {{key!r}}",
                    )
                )
                continue
            try:
                validate_value(value, PARAMETER_RULES[key])
            except ValueError as exc:
                issues.append(
                    ValidationIssueV1(
                        code="invalid_parameter_value",
                        field=key,
                        message=str(exc),
                    )
                )
                continue
            effective[key] = value
            origins[key] = "user"
        if issues:
            return ValidateConfigResultV1(
                protocol_version=TRAINER_PROTOCOL_VERSION,
                valid=False,
                schema_id=SCHEMA_ID,
                schema_digest=SCHEMA_DIGEST,
                issues=issues,
            )
        return ValidateConfigResultV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            valid=True,
            schema_id=SCHEMA_ID,
            schema_digest=SCHEMA_DIGEST,
            resolved_config={{
                "framework_id": FRAMEWORK_ID,
                "task": task,
                "architecture_id": arch,
                "dataset_contract": contract,
                "parameters": effective,
            }},
            effective_values=effective,
            origins=origins,
        )

    def train(
        self,
        spec: TrainSpecV1,
        output_dir: Path,
        *,
        spec_path: Path | None = None,
    ) -> TrainReceiptV1:
        output_dir.mkdir(parents=True, exist_ok=True)
        emitter = EventEmitter(output_dir)
        emitter.phase("preparing")

        validation = self.validate_config(
            {{
                "framework_id": spec.framework_id,
                "task": spec.task,
                "architecture_id": spec.architecture_id,
                "dataset_contract": spec.dataset_contract,
                "parameters": spec.parameters,
            }}
        )
        if not validation.valid:
            msg = "; ".join(f"{{i.field or i.code}}: {{i.message}}" for i in validation.issues)
            raise NativeConfigurationMismatch(msg)

        epochs = int(validation.effective_values["epochs"])
        seed = int(validation.effective_values["seed"])
        dataset_dir = Path(spec.dataset_dir)
        manifest, classes = validate_frozen_manifest_core(
            dataset_dir,
            expected_classes_digest=spec.classes_digest,
            expected_snapshot_digest=spec.dataset_snapshot_digest,
        )
        rows = validate_sample_files(dataset_dir, manifest)
        box_counts = [0] * len(classes)
        for _sample, _img, lbl in rows:
            text = lbl.read_text(encoding="utf-8")
            validate_yolo_bbox_label(text, len(classes))
            for line in text.splitlines():
                box_counts[int(line.split()[0])] += 1

        identity = load_dataset_identity(spec_path)
        membership = InputMembership(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            manifest=manifest,
            run_id=spec.run_id,
            attempt_id=spec.attempt_id,
            config_digest=spec.config_digest,
            dataset_snapshot_digest=spec.dataset_snapshot_digest,
            dataset_id=identity.get("dataset_id"),
            dataset_sha256=identity.get("dataset_sha256"),
            require_identities=False,
        )
        membership.observe(
            "train", [img for sample, img, _ in rows if sample["split"] == "train"]
        )
        membership.observe(
            "val", [img for sample, img, _ in rows if sample["split"] == "val"]
        )

        emitter.phase("training", total_epochs=epochs)
        store = CheckpointSlotStore(output_dir, extension=".pt")
        last_ckpt: dict[str, Any] | None = None
        final_metrics: dict[str, float] = {{}}
        for ep in range(1, epochs + 1):
            time.sleep(0.05)
            loss = round(1.0 / (ep + 1), 4)
            map50 = round(min(0.95, 0.5 + 0.1 * ep), 4)
            final_metrics = metrics_of({{"train/loss": loss, "metrics/mAP50": map50}})
            payload = canonical_json_bytes(
                {{
                    "framework": FRAMEWORK_ID,
                    "epoch": ep,
                    "seed": seed,
                    "classes": classes,
                    "box_counts": box_counts,
                    "metrics": final_metrics,
                }}
            )
            last_ckpt = store.commit_via_writer(ep, lambda s, data=payload: s.write(data))
            emitter.epoch(epoch=ep, total_epochs=epochs, metrics=final_metrics)
            emitter.checkpoint(
                epoch=ep,
                total_epochs=epochs,
                filename=last_ckpt["name"],
                sha256=last_ckpt["sha256"],
            )

        assert last_ckpt is not None
        emitter.phase("evaluating")
        weights_path = output_dir / "best.pt"
        weights_path.write_bytes((output_dir / last_ckpt["name"]).read_bytes())

        metrics_path = output_dir / "metrics.json"
        atomic_write_json(
            metrics_path,
            {{
                "summary": final_metrics,
                "classes": classes,
                "box_counts": box_counts,
            }},
        )

        artifacts = [
            build_artifact_entry("weights", weights_path),
            build_artifact_entry("checkpoint", output_dir / last_ckpt["name"]),
            build_artifact_entry("metrics", metrics_path),
        ]
        membership_result = membership.complete()
        if membership_result is not None:
            artifacts.append(build_artifact_entry("input_membership", membership_result.path))

        receipt = TrainReceiptV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            status="succeeded",
            artifacts=artifacts,
            serving_compatibility=ServingCompatibilityV1(
                framework=FRAMEWORK_ID,
                format="json",
                training_runtime={{"python": platform.python_version(), "device": "cpu"}},
                compatible_serving_runtimes=[{{"device": "cpu", "framework": FRAMEWORK_ID}}],
            ),
        )
        atomic_write_json(output_dir / "result.json", receipt)
        return receipt

    def preview(
        self,
        spec: PreviewSpecV1,
        output_dir: Path,
        *,
        spec_path: Path | None = None,
    ) -> PreviewReportV1:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_img = output_dir / "preview-0.png"
        out_img.write_bytes(_encode_png(16, 16, (60, 140, 220)))
        report = PreviewReportV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            status="succeeded",
            input_digest=spec.input_digest,
            policy_digest=spec.policy_digest,
            cuda_initialized=False,
            outputs=[
                PreviewOutputV1(
                    filename=out_img.name,
                    sha256=digest_file(out_img),
                    size_bytes=out_img.stat().st_size,
                    width=16,
                    height=16,
                    box_count=spec.sources[0].box_count,
                    metadata={{"seed": spec.seed, "epoch": spec.epoch}},
                )
            ],
            runtime={{"cpu_only": True, "python": platform.python_version()}},
        )
        atomic_write_json(output_dir / "report.json", report)
        return report


def main(argv: list[str] | None = None) -> int:
    return run_trainer_cli(
        DemoTrainerPlugin(),
        prog={slug!r},
        description="Runnable pure-Python CPU trainer example (yoao-trainer/v1).",
        supports_preview=True,
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
''',
        encoding="utf-8",
    )

    (tests_dir / "test_demo_trainer.py").write_text(
        '''from __future__ import annotations

import json
from pathlib import Path
import uuid

from demo_trainer import ARCHITECTURE_ID, FRAMEWORK_ID, DemoTrainerPlugin
from yoao_plugin_sdk.trainer import (
    POLICY_DIGEST,
    TRAINER_PROTOCOL_VERSION,
    PreviewSourceV1,
    PreviewSpecV1,
    TrainSpecV1,
    canonical_json_digest,
    digest_file,
)


def _write_dataset(root: Path) -> tuple[dict, list[str]]:
    classes = ["cat", "dog"]
    samples = []
    for split, stem, label in (("train", "t0", "0 0.5 0.5 0.4 0.4\\n"), ("val", "v0", "1 0.5 0.5 0.3 0.3\\n")):
        img = root / "images" / split / f"{stem}.png"
        lbl = root / "labels" / split / f"{stem}.txt"
        img.parent.mkdir(parents=True, exist_ok=True)
        lbl.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b"\\x89PNG\\r\\n\\x1a\\n" + stem.encode())
        lbl.write_text(label, encoding="utf-8")
        samples.append(
            {
                "task_id": str(uuid.uuid4()),
                "split": split,
                "image_path": f"images/{split}/{stem}.png",
                "label_path": f"labels/{split}/{stem}.txt",
                "sha256": digest_file(img),
                "label_sha256": digest_file(lbl),
                "width": 8,
                "height": 8,
                "size_bytes": img.stat().st_size,
            }
        )
    content_digest = canonical_json_digest({"classes": classes, "samples": samples})
    manifest = {
        "schema": 3,
        "task": "detect",
        "dataset_id": str(uuid.uuid4()),
        "dataset_sha256": content_digest,
        "content_digest": content_digest,
        "classes": classes,
        "samples": samples,
    }
    (root / "classes.json").write_text(json.dumps(classes), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest, classes


def test_demo_trainer_describe_validate_train_and_preview(tmp_path: Path) -> None:
    plugin = DemoTrainerPlugin()
    caps = plugin.describe()
    assert caps.framework_id == FRAMEWORK_ID
    assert caps.policy_digests["input_membership"] == POLICY_DIGEST

    valid = plugin.validate_config(
        {
            "framework_id": FRAMEWORK_ID,
            "task": "detect",
            "architecture_id": ARCHITECTURE_ID,
            "dataset_contract": "ultralytics-bbox-1",
            "parameters": {"epochs": 1},
        }
    )
    assert valid.valid is True
    invalid = plugin.validate_config({"task": "unknown"})
    assert invalid.valid is False and invalid.issues

    ds = tmp_path / "dataset"
    manifest, classes = _write_dataset(ds)
    out = tmp_path / "out"
    spec = TrainSpecV1(
        protocol_version=TRAINER_PROTOCOL_VERSION,
        run_id=str(uuid.uuid4()),
        attempt_id=str(uuid.uuid4()),
        framework_id=FRAMEWORK_ID,
        task="detect",
        architecture_id=ARCHITECTURE_ID,
        dataset_contract="ultralytics-bbox-1",
        dataset_dir=str(ds),
        dataset_snapshot_digest=manifest["content_digest"],
        config_digest="a" * 64,
        classes_digest=canonical_json_digest(classes),
        schema_id=valid.schema_id,
        schema_digest=valid.schema_digest,
        parameters={"epochs": 1},
    )
    receipt = plugin.train(spec, out)
    assert receipt.status == "succeeded"
    assert (out / "result.json").is_file()
    assert (out / "input-membership.json").is_file()

    preview_spec = PreviewSpecV1(
        protocol_version=TRAINER_PROTOCOL_VERSION,
        preview_id="p-1",
        framework_id=FRAMEWORK_ID,
        task="detect",
        architecture_id=ARCHITECTURE_ID,
        input_digest="b" * 64,
        config_digest="a" * 64,
        dataset_snapshot_digest=manifest["content_digest"],
        policy_digest=caps.policy_digests["preview"],
        seed=42,
        epoch=1,
        count=1,
        sources=[
            PreviewSourceV1(
                task_id=manifest["samples"][0]["task_id"],
                filename=manifest["samples"][0]["image_path"],
                sha256=manifest["samples"][0]["sha256"],
                width=8,
                height=8,
                size_bytes=manifest["samples"][0]["size_bytes"],
                box_count=1,
            )
        ],
    )
    report = plugin.preview(preview_spec, tmp_path / "preview-out")
    assert report.status == "succeeded" and report.cuda_initialized is False
''',
        encoding="utf-8",
    )


__all__ = ["init_trainer_example"]
