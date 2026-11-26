"""Pure-stdlib frozen dataset manifest, source lineage, and YOLO bbox validators."""

from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any
import uuid

from yoao_plugin_sdk.trainer.errors import DatasetContractError
from yoao_plugin_sdk.trainer.files import digest_file
from yoao_plugin_sdk.trainer.protocol import canonical_json_digest

SOURCE_LINEAGE_CONTRACT: dict[str, Any] = {"schema": 1, "scope": "accepted-reference"}
LINEAGE_FIELDS = frozenset(
    {
        "source_id",
        "location_id",
        "path_digest",
        "scan_run_id",
        "accepted_revision_id",
        "revision_number",
        "previous_revision_id",
        "accepted_at",
        "scan_completed_at",
        "content_verified_at",
        "source_config_revision",
        "location_config_revision",
    }
)
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


def ordered_names(names: Any) -> list[str]:
    if isinstance(names, dict):
        indexed: dict[int, Any] = {}
        for key, value in names.items():
            if isinstance(key, bool):
                raise DatasetContractError("invalid_classes", "Invalid dataset class indexes")
            try:
                idx = int(key)
            except (TypeError, ValueError) as exc:
                raise DatasetContractError(
                    "invalid_classes", "Invalid dataset class indexes"
                ) from exc
            if idx in indexed:
                raise DatasetContractError("invalid_classes", "Duplicate dataset class index")
            indexed[idx] = value
        if set(indexed) != set(range(len(indexed))):
            raise DatasetContractError("invalid_classes", "Invalid dataset class indexes")
        ordered = [indexed[i] for i in range(len(indexed))]
    elif isinstance(names, list):
        ordered = list(names)
    else:
        raise DatasetContractError(
            "invalid_classes", "Dataset class names must be a list or indexed mapping"
        )
    if not ordered or any(not isinstance(item, str) or not item for item in ordered):
        raise DatasetContractError(
            "invalid_classes", "Dataset class names must be non-empty strings"
        )
    return ordered


def _require_uuid(value: Any, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str):
        raise DatasetContractError(
            "invalid_source_lineage", "Lineage identifier must be a canonical UUID"
        )
    try:
        parsed = str(uuid.UUID(value))
    except ValueError as exc:
        raise DatasetContractError(
            "invalid_source_lineage", "Lineage identifier must be a canonical UUID"
        ) from exc
    if parsed != value:
        raise DatasetContractError(
            "invalid_source_lineage", "Lineage identifier must be a canonical UUID"
        )


def _require_timestamp(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, str) or len(value) > 64 or "T" not in value:
        raise DatasetContractError(
            "invalid_source_lineage", "Lineage timestamp must be tz-aware ISO-8601"
        )
    norm = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
    try:
        parsed = datetime.fromisoformat(norm)
    except ValueError as exc:
        raise DatasetContractError(
            "invalid_source_lineage", "Lineage timestamp must be tz-aware ISO-8601"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DatasetContractError(
            "invalid_source_lineage", "Lineage timestamp must be tz-aware ISO-8601"
        )


def validate_source_lineage(manifest: dict[str, Any]) -> None:
    if not isinstance(manifest, dict):
        raise DatasetContractError(
            "invalid_source_lineage", "Frozen dataset manifest must be a JSON object"
        )
    declared = False
    if "source_lineage_contract" in manifest:
        contract = manifest["source_lineage_contract"]
        if (
            not isinstance(contract, dict)
            or type(contract.get("schema")) is not int
            or contract != SOURCE_LINEAGE_CONTRACT
        ):
            raise DatasetContractError(
                "unknown_source_lineage_contract",
                "Unknown frozen source lineage contract",
            )
        declared = True
    samples = manifest.get("samples", [])
    if not isinstance(samples, list):
        raise DatasetContractError(
            "invalid_source_lineage", "Frozen dataset samples must be a list"
        )
    for member in samples:
        if not isinstance(member, dict):
            raise DatasetContractError(
                "invalid_source_lineage", "Each frozen sample must be a JSON object"
            )
        if "source_lineage" not in member:
            if declared and member.get("asset_reference") is not None:
                raise DatasetContractError(
                    "invalid_source_lineage",
                    "Referenced sample is missing declared source_lineage",
                )
            continue
        if not declared:
            raise DatasetContractError(
                "invalid_source_lineage",
                "source_lineage present without source_lineage_contract",
            )
        if member.get("asset_reference") is None:
            raise DatasetContractError(
                "invalid_source_lineage",
                "source_lineage must belong to a referenced asset",
            )
        value = member["source_lineage"]
        if not isinstance(value, dict) or set(value) != LINEAGE_FIELDS:
            raise DatasetContractError(
                "invalid_source_lineage",
                "Incomplete or unauthorized source_lineage fields",
            )
        for key in ("source_id", "location_id", "scan_run_id"):
            _require_uuid(value[key])
        for key in ("accepted_revision_id", "previous_revision_id"):
            _require_uuid(value[key], nullable=True)
        if not isinstance(value["path_digest"], str) or not _SHA256_RE.fullmatch(
            value["path_digest"]
        ):
            raise DatasetContractError(
                "invalid_source_lineage", "path_digest must be a SHA-256 hex digest"
            )
        for key in ("source_config_revision", "location_config_revision"):
            if type(value[key]) is not int or value[key] < 1:
                raise DatasetContractError(
                    "invalid_source_lineage", f"{key} must be a positive integer"
                )
        rev = value["revision_number"]
        if rev is not None and (type(rev) is not int or rev < 1):
            raise DatasetContractError(
                "invalid_source_lineage", "revision_number must be positive or None"
            )
        for key in ("accepted_at", "scan_completed_at", "content_verified_at"):
            _require_timestamp(value[key])


def _read_json_file(path: Path, label: str) -> Any:
    if path.is_symlink():
        raise DatasetContractError(
            "symlink_rejected", f"Frozen dataset {label} must not be a symlink"
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DatasetContractError(
            "invalid_dataset_json", f"Failed to parse frozen dataset {label}: {exc}"
        ) from exc


def validate_frozen_manifest_core(
    dataset_dir: str | Path,
    *,
    expected_classes_digest: str,
    expected_snapshot_digest: str | None = None,
    data_yaml_config: dict[str, Any] | None = None,
    require_classes_file: bool = True,
) -> tuple[dict[str, Any], list[str]]:
    """Validate frozen dataset manifest.json, classes.json, and optional parsed data.yaml."""
    root = Path(dataset_dir).resolve()
    manifest_path = root / "manifest.json"
    classes_path = root / "classes.json"
    if not manifest_path.is_file():
        raise DatasetContractError(
            "missing_dataset_files", "Frozen dataset is missing manifest.json"
        )
    if require_classes_file and not classes_path.is_file():
        raise DatasetContractError(
            "missing_dataset_files", "Frozen dataset is missing classes.json"
        )

    manifest = _read_json_file(manifest_path, "manifest.json")
    if not isinstance(manifest, dict):
        raise DatasetContractError(
            "invalid_manifest", "Frozen dataset manifest must be a JSON object"
        )
    validate_source_lineage(manifest)

    if data_yaml_config is not None:
        if (
            not isinstance(data_yaml_config, dict)
            or set(data_yaml_config) != {"train", "val", "names"}
            or data_yaml_config["train"] != "images/train"
            or data_yaml_config["val"] != "images/val"
        ):
            raise DatasetContractError(
                "invalid_data_yaml",
                "Unsupported frozen dataset paths or download configuration in data.yaml",
            )
        classes = ordered_names(data_yaml_config["names"])
        if manifest.get("classes") != classes:
            raise DatasetContractError(
                "class_mismatch", "Frozen manifest classes differ from data.yaml"
            )
    else:
        classes = ordered_names(manifest.get("classes"))

    if classes_path.is_file():
        classes_file = _read_json_file(classes_path, "classes.json")
        if classes_file != classes:
            raise DatasetContractError(
                "class_mismatch", "Frozen classes.json differs from dataset classes"
            )

    if canonical_json_digest(classes) != expected_classes_digest:
        raise DatasetContractError(
            "classes_digest_mismatch",
            "Frozen dataset classes digest does not match spec.classes_digest",
        )

    if expected_snapshot_digest is not None:
        content_digest = manifest.get("content_digest")
        if content_digest is not None and (
            not isinstance(content_digest, str)
            or content_digest != expected_snapshot_digest
        ):
            raise DatasetContractError(
                "snapshot_digest_mismatch",
                "Frozen dataset content_digest does not match spec.dataset_snapshot_digest",
            )

    return manifest, classes


def _has_symlink_component(root: Path, rel: PurePosixPath) -> bool:
    current = root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def validate_sample_files(
    dataset_dir: str | Path,
    manifest: dict[str, Any],
    *,
    require_splits: tuple[str, ...] = ("train", "val"),
) -> list[tuple[dict[str, Any], Path, Path]]:
    """Verify frozen sample paths, containment, symlink prohibition, and exact SHA-256 digests."""
    root = Path(dataset_dir).resolve()
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise DatasetContractError(
            "missing_samples", "Frozen dataset manifest must contain samples"
        )

    allowed_splits = {"train", "val"} | set(require_splits)
    seen: set[str] = set()
    splits: set[str] = set()
    validated: list[tuple[dict[str, Any], Path, Path]] = []

    for sample in samples:
        if not isinstance(sample, dict) or sample.get("split") not in allowed_splits:
            raise DatasetContractError(
                "invalid_sample_split", "Invalid frozen dataset sample split"
            )
        split = sample["split"]
        splits.add(split)
        raw_image = sample.get("image_path")
        raw_label = sample.get("label_path")
        parsed_rels: list[PurePosixPath] = []
        for index, raw in enumerate((raw_image, raw_label)):
            if (
                not isinstance(raw, str)
                or len(raw) > 1000
                or "\\" in raw
                or "\x00" in raw
            ):
                raise DatasetContractError("invalid_sample_path", f"Invalid sample path: {raw!r}")
            rel = PurePosixPath(raw)
            expected_root = "images" if index == 0 else "labels"
            if (
                rel.as_posix() != raw
                or rel.is_absolute()
                or ".." in rel.parts
                or len(rel.parts) != 3
                or rel.parts[:2] != (expected_root, split)
                or raw in seen
            ):
                raise DatasetContractError("invalid_sample_path", f"Invalid sample path: {raw!r}")
            seen.add(raw)
            parsed_rels.append(rel)

        image_rel, label_rel = parsed_rels
        image_path = root / image_rel
        label_path = root / label_rel

        if _has_symlink_component(root, image_rel) or _has_symlink_component(root, label_rel):
            raise DatasetContractError(
                "symlink_rejected", "Frozen dataset files must not be symlinks"
            )
        if (
            image_path.stem != label_path.stem
            or not image_path.is_file()
            or not label_path.is_file()
            or not image_path.resolve().is_relative_to(root)
            or not label_path.resolve().is_relative_to(root)
            or digest_file(image_path) != sample.get("sha256")
            or digest_file(label_path) != sample.get("label_sha256")
        ):
            raise DatasetContractError(
                "sample_digest_mismatch",
                "Frozen image or explicit label file missing or changed",
            )
        validated.append((sample, image_path, label_path))

    if not set(require_splits).issubset(splits):
        raise DatasetContractError(
            "missing_splits",
            f"Frozen dataset requires splits {sorted(require_splits)}; got {sorted(splits)}",
        )
    return validated


def validate_yolo_bbox_label(text: str, num_classes: int) -> None:
    if type(num_classes) is not int or num_classes < 1:
        raise DatasetContractError("invalid_classes", "num_classes must be a positive integer")
    for line in text.splitlines():
        values = line.split()
        if len(values) != 5:
            raise DatasetContractError("invalid_label", "Invalid detection label line")
        try:
            class_index = int(values[0])
            x, y, w, h = map(float, values[1:])
        except ValueError as exc:
            raise DatasetContractError(
                "invalid_label", "Invalid numeric detection label"
            ) from exc
        if (
            not 0 <= class_index < num_classes
            or not all(math.isfinite(v) for v in (x, y, w, h))
            or min(w, h) <= 0
            or x - w / 2 < -1e-8
            or y - h / 2 < -1e-8
            or x + w / 2 > 1 + 1e-8
            or y + h / 2 > 1 + 1e-8
        ):
            raise DatasetContractError(
                "invalid_label", "Invalid normalized detection bounding box"
            )


__all__ = [
    "LINEAGE_FIELDS",
    "SOURCE_LINEAGE_CONTRACT",
    "ordered_names",
    "validate_frozen_manifest_core",
    "validate_sample_files",
    "validate_source_lineage",
    "validate_yolo_bbox_label",
]
