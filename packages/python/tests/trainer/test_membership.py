from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import uuid

import pytest

from yoao_plugin_sdk.trainer import (
    MAX_MEMBERS,
    POLICY_DIGEST,
    InputMembership,
    InputMembershipResult,
    InputMismatchError,
    contract_for,
    has_member_identities,
    member_records,
)

EXPECTED_POLICY_DIGEST = (
    "1ad70c6b02157e1b4c33646027309a72e73aaa5eba693e7a5fe52ce8f9bb71b4"
)
DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
DIGEST_D = "d" * 64


def _sample(
    split: str,
    stem: str,
    *,
    task_id: str | None = None,
    img_sha: str = DIGEST_A,
    lbl_sha: str = DIGEST_B,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "split": split,
        "image_path": f"images/{split}/{stem}.jpg",
        "label_path": f"labels/{split}/{stem}.txt",
        "sha256": img_sha,
        "label_sha256": lbl_sha,
        "width": 64,
        "height": 64,
        "size_bytes": 123,
    }
    if task_id is not None:
        row["task_id"] = task_id
    return row


def _manifest_with_identities() -> dict[str, Any]:
    return {
        "dataset_id": str(uuid.uuid4()),
        "dataset_sha256": DIGEST_C,
        "samples": [
            _sample("train", "t0", task_id=str(uuid.uuid4())),
            _sample("train", "t1", task_id=str(uuid.uuid4())),
            _sample("val", "v0", task_id=str(uuid.uuid4())),
        ],
    }


def _create_files(root: Path, manifest: dict[str, Any]) -> None:
    for sample in manifest["samples"]:
        for key in ("image_path", "label_path"):
            path = root / sample[key]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"data")


def test_policy_digest_matches_platform_contract() -> None:
    assert POLICY_DIGEST == EXPECTED_POLICY_DIGEST


def test_input_membership_strict_happy_path_and_reordering(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    manifest = _manifest_with_identities()
    _create_files(dataset_dir, manifest)

    contract = contract_for(manifest)
    assert contract is not None
    assert contract["policy_digest"] == EXPECTED_POLICY_DIGEST
    assert contract["counts"] == {"train": 2, "val": 1}

    tracker = InputMembership(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        manifest=manifest,
        run_id=str(uuid.uuid4()),
        attempt_id=str(uuid.uuid4()),
        config_digest=DIGEST_D,
        dataset_id=manifest["dataset_id"],
        dataset_sha256=manifest["dataset_sha256"],
        require_identities=True,
    )
    # Reordered native paths must be accepted as a multiset.
    tracker.observe(
        "train",
        [dataset_dir / "images/train/t1.jpg", dataset_dir / "images/train/t0.jpg"],
    )
    tracker.observe("val", ["images/val/v0.jpg"])
    # Repeating identical observation is idempotent.
    tracker.observe("val", [dataset_dir / "images/val/v0.jpg"])

    result = tracker.complete()
    assert isinstance(result, InputMembershipResult)
    assert result.path == output_dir / "input-membership.json"
    assert result.path.is_file()
    written = json.loads(result.path.read_text(encoding="utf-8"))
    assert written == result.proof
    assert written["policy_digest"] == EXPECTED_POLICY_DIGEST
    assert written["members_digest"] == contract["members_digest"]
    assert written["counts"] == {"train": 2, "val": 1}


def test_input_membership_optional_identities_mode(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    manifest = {
        "samples": [
            _sample("train", "t0"),
            _sample("val", "v0"),
        ]
    }
    _create_files(dataset_dir, manifest)
    assert has_member_identities(manifest) is False

    # Strict mode rejects missing task_id UUIDs.
    with pytest.raises(InputMismatchError):
        InputMembership(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            manifest=manifest,
            run_id="run-1",
            attempt_id="attempt-1",
            config_digest=DIGEST_D,
            dataset_snapshot_digest=DIGEST_C,
            require_identities=True,
        )

    # Non-strict mode validates path multisets and returns None without writing proof file.
    tracker = InputMembership(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        manifest=manifest,
        run_id="run-1",
        attempt_id="attempt-1",
        config_digest=DIGEST_D,
        dataset_snapshot_digest=DIGEST_C,
        require_identities=False,
    )
    assert tracker.has_identities is False
    tracker.observe("train", ["images/train/t0.jpg"])
    tracker.observe("val", ["images/val/v0.jpg"])
    assert tracker.complete() is None
    assert not (output_dir / "input-membership.json").exists()

    # Non-strict mode WITH UUID task_ids writes input-membership.json using snapshot digest fallback.
    manifest_uuid = _manifest_with_identities()
    manifest_uuid.pop("dataset_id")
    manifest_uuid.pop("dataset_sha256")
    _create_files(dataset_dir, manifest_uuid)
    tracker_uuid = InputMembership(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        manifest=manifest_uuid,
        run_id="run-1",
        attempt_id="attempt-1",
        config_digest=DIGEST_D,
        dataset_snapshot_digest=DIGEST_C,
        require_identities=False,
    )
    assert tracker_uuid.has_identities is True
    tracker_uuid.observe("train", ["images/train/t0.jpg", "images/train/t1.jpg"])
    tracker_uuid.observe("val", ["images/val/v0.jpg"])
    res = tracker_uuid.complete()
    assert isinstance(res, InputMembershipResult)
    assert res.proof["dataset_id"] == DIGEST_C
    assert res.proof["dataset_sha256"] == DIGEST_C


def test_input_membership_rejects_traversal_missing_split_mutation_and_duplicates(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    manifest = _manifest_with_identities()
    _create_files(dataset_dir, manifest)

    tracker = InputMembership(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        manifest=manifest,
        run_id=str(uuid.uuid4()),
        attempt_id=str(uuid.uuid4()),
        config_digest=DIGEST_D,
        dataset_id=manifest["dataset_id"],
        dataset_sha256=manifest["dataset_sha256"],
    )

    # Path traversal rejected.
    with pytest.raises(InputMismatchError):
        tracker.observe("val", ["../dataset/images/val/v0.jpg"])
    with pytest.raises(InputMismatchError):
        tracker.observe("val", [tmp_path / "outside.jpg"])

    # Missing split observation rejected on complete().
    tracker.observe("train", ["images/train/t0.jpg", "images/train/t1.jpg"])
    with pytest.raises(InputMismatchError, match="incomplete"):
        tracker.complete()

    # Mid-run split mutation rejected even when mutated to different valid-looking list.
    tracker.observe("val", ["images/val/v0.jpg"])
    with pytest.raises(InputMismatchError):
        tracker.observe("train", ["images/train/t0.jpg"])
    # Mutating internal observed split state triggers mid-run change detection.
    tracker.members["val"] = [{"task_id": str(uuid.uuid4()), "member_digest": DIGEST_A}]
    with pytest.raises(InputMismatchError, match="changed during execution"):
        tracker.observe("val", ["images/val/v0.jpg"])

    # Duplicate label path rejected in member_records.
    dup_manifest = {
        "samples": [
            {
                **_sample("train", "a", task_id=str(uuid.uuid4())),
                "image_path": "images/train/a.jpg",
                "label_path": "labels/train/a.txt",
            },
            {
                **_sample("train", "a", task_id=str(uuid.uuid4())),
                "image_path": "images/train/a.png",
                "label_path": "labels/train/a.txt",
            },
        ]
    }
    with pytest.raises(ValueError, match="duplicate"):
        member_records(dup_manifest)


def test_input_membership_non_strict_rejects_missing_split_partial_task_id_and_oversized_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "output"

    # Missing val split in require_identities=False must raise InputMismatchError.
    only_train = {"samples": [_sample("train", "t0")]}
    with pytest.raises(InputMismatchError, match="training and validation"):
        InputMembership(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            manifest=only_train,
            run_id="run-1",
            attempt_id="attempt-1",
            config_digest=DIGEST_D,
            require_identities=False,
        )

    # Partially populated task_id must NOT silently downgrade to non-proof mode.
    partial_manifest = {
        "samples": [
            _sample("train", "t0", task_id=str(uuid.uuid4())),
            _sample("val", "v0"),
        ]
    }
    with pytest.raises(InputMismatchError):
        InputMembership(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            manifest=partial_manifest,
            run_id="run-1",
            attempt_id="attempt-1",
            config_digest=DIGEST_D,
            require_identities=False,
        )

    # Corrupted task_id (invalid UUID) must NOT silently downgrade to non-proof mode.
    corrupted_manifest = {
        "samples": [
            _sample("train", "t0", task_id="not-a-uuid"),
            _sample("val", "v0", task_id=str(uuid.uuid4())),
        ]
    }
    with pytest.raises(InputMismatchError):
        InputMembership(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            manifest=corrupted_manifest,
            run_id="run-1",
            attempt_id="attempt-1",
            config_digest=DIGEST_D,
            require_identities=False,
        )

    # Explicit null task_id key also triggers identity enforcement instead of silent downgrade.
    null_task_id_manifest = {
        "samples": [
            {**_sample("train", "t0"), "task_id": None},
            _sample("val", "v0"),
        ]
    }
    with pytest.raises(InputMismatchError):
        InputMembership(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            manifest=null_task_id_manifest,
            run_id="run-1",
            attempt_id="attempt-1",
            config_digest=DIGEST_D,
            require_identities=False,
        )

    # Oversized native_paths rejected before resolving any paths.
    valid_manifest = _manifest_with_identities()
    _create_files(dataset_dir, valid_manifest)
    tracker = InputMembership(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        manifest=valid_manifest,
        run_id=str(uuid.uuid4()),
        attempt_id=str(uuid.uuid4()),
        config_digest=DIGEST_D,
        dataset_id=valid_manifest["dataset_id"],
        dataset_sha256=valid_manifest["dataset_sha256"],
    )
    resolve_calls = 0
    real_resolve = Path.resolve

    def _counting_resolve(self: Path, strict: bool = False) -> Path:
        nonlocal resolve_calls
        resolve_calls += 1
        return real_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", _counting_resolve)
    with pytest.raises(InputMismatchError, match="limit"):
        tracker.observe("train", ["images/train/t0.jpg"] * (MAX_MEMBERS + 1))
    assert resolve_calls == 0
