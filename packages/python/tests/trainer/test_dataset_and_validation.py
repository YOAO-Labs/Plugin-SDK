from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
from typing import Any
import uuid

import pytest

from yoao_plugin_sdk.trainer import (
    SOURCE_LINEAGE_CONTRACT,
    DatasetContractError,
    InputMismatchError,
    canonical_json_digest,
    digest_file,
    hash_source_files,
    metrics_of,
    normalized,
    optional_package_version,
    ordered_names,
    validate_frozen_manifest_core,
    validate_sample_files,
    validate_source_lineage,
    validate_value,
    validate_yolo_bbox_label,
)

DIGEST_A = "a" * 64


def _valid_lineage() -> dict[str, Any]:
    return {
        "source_id": str(uuid.uuid4()),
        "location_id": str(uuid.uuid4()),
        "path_digest": DIGEST_A,
        "scan_run_id": str(uuid.uuid4()),
        "accepted_revision_id": str(uuid.uuid4()),
        "revision_number": 2,
        "previous_revision_id": None,
        "accepted_at": "2026-11-26T10:00:00Z",
        "scan_completed_at": "2026-11-26T10:00:00+08:00",
        "content_verified_at": None,
        "source_config_revision": 1,
        "location_config_revision": 3,
    }


def _build_frozen_dataset(root: Path, *, classes: list[str] | None = None) -> tuple[dict[str, Any], list[str]]:
    cls = classes or ["cat", "dog"]
    root.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    for split, stem, label_text in (
        ("train", "t0", "0 0.5 0.5 0.4 0.4\n"),
        ("val", "v0", "1 0.25 0.25 0.2 0.2\n"),
    ):
        img = root / "images" / split / f"{stem}.jpg"
        lbl = root / "labels" / split / f"{stem}.txt"
        img.parent.mkdir(parents=True, exist_ok=True)
        lbl.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(f"img-{stem}".encode("utf-8"))
        lbl.write_text(label_text, encoding="utf-8")
        samples.append(
            {
                "task_id": str(uuid.uuid4()),
                "split": split,
                "image_path": f"images/{split}/{stem}.jpg",
                "label_path": f"labels/{split}/{stem}.txt",
                "sha256": digest_file(img),
                "label_sha256": digest_file(lbl),
                "width": 64,
                "height": 64,
                "size_bytes": img.stat().st_size,
            }
        )
    manifest: dict[str, Any] = {
        "schema": 3,
        "task": "detect",
        "classes": cls,
        "content_digest": canonical_json_digest(samples),
        "samples": samples,
    }
    (root / "classes.json").write_text(json.dumps(cls), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest, cls


def test_validate_source_lineage_accepts_valid_and_rejects_violations() -> None:
    lineage = _valid_lineage()
    valid_manifest = {
        "source_lineage_contract": dict(SOURCE_LINEAGE_CONTRACT),
        "samples": [
            {"asset_reference": {"id": "ref-1"}, "source_lineage": lineage},
            {"asset_reference": None},
        ],
    }
    validate_source_lineage(valid_manifest)
    validate_source_lineage({"samples": [{"asset_reference": {"id": "legacy"}}]})

    bad_cases: list[dict[str, Any]] = [
        {"source_lineage_contract": {"schema": 2, "scope": "accepted-reference"}, "samples": []},
        {"source_lineage_contract": {"schema": True, "scope": "accepted-reference"}, "samples": []},
        {"samples": [{"asset_reference": {"id": "ref-1"}, "source_lineage": lineage}]},
        {
            "source_lineage_contract": dict(SOURCE_LINEAGE_CONTRACT),
            "samples": [{"asset_reference": {"id": "ref-1"}}],
        },
        {
            "source_lineage_contract": dict(SOURCE_LINEAGE_CONTRACT),
            "samples": [{"asset_reference": None, "source_lineage": lineage}],
        },
        {
            "source_lineage_contract": dict(SOURCE_LINEAGE_CONTRACT),
            "samples": [
                {"asset_reference": {"id": "ref-1"}, "source_lineage": {**lineage, "extra": 1}}
            ],
        },
        {
            "source_lineage_contract": dict(SOURCE_LINEAGE_CONTRACT),
            "samples": [
                {
                    "asset_reference": {"id": "ref-1"},
                    "source_lineage": {**lineage, "source_id": lineage["source_id"].upper()},
                }
            ],
        },
        {
            "source_lineage_contract": dict(SOURCE_LINEAGE_CONTRACT),
            "samples": [
                {
                    "asset_reference": {"id": "ref-1"},
                    "source_lineage": {**lineage, "accepted_at": "2026-11-26T10:00:00"},
                }
            ],
        },
        {
            "source_lineage_contract": dict(SOURCE_LINEAGE_CONTRACT),
            "samples": [
                {
                    "asset_reference": {"id": "ref-1"},
                    "source_lineage": {**lineage, "source_config_revision": 0},
                }
            ],
        },
        {
            "source_lineage_contract": dict(SOURCE_LINEAGE_CONTRACT),
            "samples": [
                {
                    "asset_reference": {"id": "ref-1"},
                    "source_lineage": {**lineage, "revision_number": True},
                }
            ],
        },
        {
            "source_lineage_contract": dict(SOURCE_LINEAGE_CONTRACT),
            "samples": [
                {
                    "asset_reference": {"id": "ref-1"},
                    "source_lineage": {**lineage, "path_digest": "not-sha256"},
                }
            ],
        },
    ]
    for bad in bad_cases:
        with pytest.raises(DatasetContractError) as exc_info:
            validate_source_lineage(bad)
        assert isinstance(exc_info.value, InputMismatchError)


def test_ordered_names_manifest_core_samples_and_bbox_labels(tmp_path: Path) -> None:
    assert ordered_names(["cat", "dog"]) == ["cat", "dog"]
    assert ordered_names({0: "cat", 1: "dog"}) == ["cat", "dog"]
    assert ordered_names({"0": "cat", "1": "dog"}) == ["cat", "dog"]
    for bad_names in ([], [""], {1: "dog"}, {True: "cat"}, "cat"):
        with pytest.raises(DatasetContractError):
            ordered_names(bad_names)

    ds = tmp_path / "dataset"
    manifest, classes = _build_frozen_dataset(ds)
    classes_digest = canonical_json_digest(classes)

    loaded, loaded_classes = validate_frozen_manifest_core(
        ds,
        expected_classes_digest=classes_digest,
        expected_snapshot_digest=manifest["content_digest"],
        data_yaml_config={"train": "images/train", "val": "images/val", "names": {0: "cat", 1: "dog"}},
    )
    assert loaded_classes == classes
    rows = validate_sample_files(ds, loaded)
    assert len(rows) == 2
    for _sample, _img, lbl in rows:
        validate_yolo_bbox_label(lbl.read_text(encoding="utf-8"), len(loaded_classes))

    # Works without data_yaml_config (pure stdlib, no PyYAML needed).
    _, loaded_classes_no_yaml = validate_frozen_manifest_core(
        ds, expected_classes_digest=classes_digest
    )
    assert loaded_classes_no_yaml == classes

    # Rejects digest mismatch and bad data_yaml_config.
    with pytest.raises(DatasetContractError):
        validate_frozen_manifest_core(ds, expected_classes_digest="f" * 64)
    with pytest.raises(DatasetContractError):
        validate_frozen_manifest_core(
            ds,
            expected_classes_digest=classes_digest,
            data_yaml_config={"train": "images/train", "val": "images/val", "names": classes, "download": "http://bad"},
        )

    # Rejects missing val split.
    only_train = {**manifest, "samples": [manifest["samples"][0]]}
    with pytest.raises(DatasetContractError) as exc_info:
        validate_sample_files(ds, only_train)
    assert exc_info.value.code == "missing_splits"

    # Rejects .. traversal.
    bad_path_manifest = {
        **manifest,
        "samples": [{**manifest["samples"][0], "image_path": "images/train/../t0.jpg"}, manifest["samples"][1]],
    }
    with pytest.raises(DatasetContractError):
        validate_sample_files(ds, bad_path_manifest)

    # Rejects SHA-256 drift.
    drifted = {
        **manifest,
        "samples": [{**manifest["samples"][0], "sha256": "0" * 64}, manifest["samples"][1]],
    }
    with pytest.raises(DatasetContractError):
        validate_sample_files(ds, drifted)

    # Rejects symlinks BEFORE resolve().
    ds_sym = tmp_path / "dataset-sym"
    sym_manifest, _ = _build_frozen_dataset(ds_sym)
    real_img = ds_sym / "images/train/t0.jpg"
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(real_img.read_bytes())
    real_img.unlink()
    os.symlink(outside, real_img)
    with pytest.raises(DatasetContractError) as sym_exc:
        validate_sample_files(ds_sym, sym_manifest)
    assert sym_exc.value.code == "symlink_rejected"

    # YOLO bbox label checks.
    validate_yolo_bbox_label("", 2)
    validate_yolo_bbox_label("0 0.5 0.5 1.0 1.0\n", 2)
    for bad_label in (
        "0 0.5 0.5 0.2\n",
        "2 0.5 0.5 0.2 0.2\n",
        "-1 0.5 0.5 0.2 0.2\n",
        "0 0.5 0.5 0.0 0.2\n",
        "0 0.95 0.5 0.2 0.2\n",
        "0 nan 0.5 0.2 0.2\n",
    ):
        with pytest.raises(DatasetContractError):
            validate_yolo_bbox_label(bad_label, 2)


def test_validation_metrics_source_hash_and_package_version(tmp_path: Path) -> None:
    assert normalized({"p": PurePosixPath("a/b"), "t": (1, 2)}) == {"p": "a/b", "t": [1, 2]}

    validate_value(10, {"type": "integer", "minimum": 1, "maximum": 100})
    validate_value(640, {"type": "integer", "minimum": 128, "multipleOf": 32})
    validate_value(0.5, {"type": "number", "exclusiveMinimum": 0, "maximum": 1})
    validate_value(1, {"type": "number", "minimum": 0})
    validate_value("SGD", {"enum": ["SGD", "Adam"]})
    validate_value(None, {"anyOf": [{"type": "null"}, {"type": "integer", "minimum": 1}]})
    validate_value(
        [0.5, 1.5],
        {"type": "array", "items": {"type": "number", "exclusiveMinimum": 0}, "minItems": 2, "maxItems": 2, "ordered": True, "uniqueItems": True},
    )

    # Strict no-coercion rejections: bool is never int/number, NaN rejected, unordered rejected.
    for bad_val, rule in (
        (True, {"type": "integer", "minimum": 0}),
        (False, {"type": "number", "minimum": 0}),
        (1, {"enum": [True]}),
        (1.0, {"type": "integer"}),
        (float("nan"), {"type": "number"}),
        (float("inf"), {"type": "number"}),
        (0, {"type": "number", "exclusiveMinimum": 0}),
        (650, {"type": "integer", "multipleOf": 32}),
        ([1.5, 0.5], {"type": "array", "items": {"type": "number"}, "ordered": True}),
        ([1, 1], {"type": "array", "items": {"type": "integer"}, "uniqueItems": True}),
        ("bad", {"anyOf": [{"type": "null"}, {"type": "integer"}]}),
    ):
        with pytest.raises(ValueError):
            validate_value(bad_val, rule)

    raw_metrics = {
        "ok_int": 2,
        "ok_float": 0.75,
        "neg_zero": -0.0,
        "flag": True,
        "nan": float("nan"),
        "inf": float("inf"),
        "ninf": float("-inf"),
        "label": "cat",
        "empty": None,
    }
    cleaned = metrics_of(raw_metrics)
    assert cleaned == {"ok_int": 2.0, "ok_float": 0.75, "neg_zero": 0.0}
    assert str(cleaned["neg_zero"]) == "0.0"
    assert metrics_of(None) == {}

    f1 = tmp_path / "one.py"
    f2 = tmp_path / "two.py"
    f1.write_text("x = 1\n", encoding="utf-8")
    f2.write_text("y = 2\n", encoding="utf-8")
    digest_1 = hash_source_files([f1, f2])
    assert len(digest_1) == 64
    f2.write_text("y = 3\n", encoding="utf-8")
    assert hash_source_files([f1, f2]) != digest_1

    assert optional_package_version("pydantic") is not None
    assert optional_package_version("definitely-not-a-real-package-xyz", default="missing") == "missing"
