"""Offline yoao-trainer/v1 contract verifier (`yoao-plugin trainer check`)."""

from __future__ import annotations

from collections.abc import Sequence
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import struct
import subprocess
import tempfile
import time
from typing import Any
import uuid
import zlib

from pydantic import BaseModel, ConfigDict, Field

from yoao_plugin_sdk.manifest import EXTENSION_MANIFEST_SCHEMA, load_manifest_file
from yoao_plugin_sdk.trainer.files import atomic_write_json, digest_file
from yoao_plugin_sdk.trainer.membership import POLICY_DIGEST, contract_for
from yoao_plugin_sdk.trainer.protocol import (
    TRAINER_PROTOCOL_VERSION,
    PreviewReportV1,
    PreviewSourceV1,
    PreviewSpecV1,
    TrainerCapabilitiesV1,
    TrainEventV1,
    TrainReceiptV1,
    TrainSpecV1,
    ValidateConfigResultV1,
    canonical_json_bytes,
    canonical_json_digest,
)
from yoao_plugin_sdk.trainer.runtime import SUPERVISOR_OWNED_ARTIFACTS

_CHECKPOINT_NAME_RE = re.compile(r"^checkpoint-([0-9]{1,4})\.[a-zA-Z0-9_-]{1,32}$")
_INPUT_MEMBERSHIP_FIELDS = frozenset(
    {
        "schema",
        "policy_digest",
        "run_id",
        "attempt_id",
        "config_digest",
        "dataset_id",
        "dataset_sha256",
        "members_digest",
        "members",
        "counts",
    }
)


class CheckResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    skipped: bool = False
    detail: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)


class TrainerCheckReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: str = TRAINER_PROTOCOL_VERSION
    entrypoint: list[str]
    passed: bool
    checks: dict[str, CheckResultV1]


def _ok(name: str, detail: str, **evidence: Any) -> CheckResultV1:
    return CheckResultV1(name=name, passed=True, detail=detail, evidence=evidence)


def _fail(name: str, detail: str) -> CheckResultV1:
    return CheckResultV1(name=name, passed=False, detail=detail)


def _skip(name: str, detail: str) -> CheckResultV1:
    return CheckResultV1(name=name, passed=True, skipped=True, detail=detail)


def _make_solid_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    raw = bytearray()
    row = bytes([0]) + bytes(rgb) * width
    for _ in range(height):
        raw.extend(row)

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(bytes(raw), level=1))
        + _chunk(b"IEND", b"")
    )


def _snapshot_tree(root: Path) -> list[tuple[str, str]]:
    return sorted(
        (str(path.relative_to(root)), digest_file(path))
        for path in root.rglob("*")
        if path.is_file()
    )


def _create_synthetic_dataset(root: Path) -> dict[str, Any]:
    classes = ["cat", "dog"]
    dataset_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "yoao-trainer-check-dataset"))
    samples: list[dict[str, Any]] = []
    specs = (
        ("train", "sample-000", (200, 40, 40), "0 0.5 0.5 0.4 0.4\n"),
        ("train", "sample-001", (40, 200, 40), "1 0.5 0.5 0.3 0.3\n"),
        ("val", "sample-002", (40, 40, 200), "0 0.5 0.5 0.35 0.35\n"),
    )
    for split, stem, rgb, label_text in specs:
        img_path = root / "images" / split / f"{stem}.png"
        lbl_path = root / "labels" / split / f"{stem}.txt"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        lbl_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(_make_solid_png(8, 8, rgb))
        lbl_path.write_text(label_text, encoding="utf-8")
        samples.append(
            {
                "task_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"yoao-check-{stem}")),
                "split": split,
                "image_path": f"images/{split}/{stem}.png",
                "label_path": f"labels/{split}/{stem}.txt",
                "sha256": digest_file(img_path),
                "label_sha256": digest_file(lbl_path),
                "width": 8,
                "height": 8,
                "size_bytes": img_path.stat().st_size,
            }
        )

    content_digest = canonical_json_digest({"classes": classes, "samples": samples})
    validation_digest = canonical_json_digest([s for s in samples if s["split"] == "val"])
    manifest: dict[str, Any] = {
        "schema": 3,
        "task": "detect",
        "media_type": "image",
        "format": "ultralytics-yolo",
        "geometry": "normalized-bbox",
        "bridge": "ultralytics-bbox-1",
        "dataset_id": dataset_id,
        "dataset_sha256": content_digest,
        "content_digest": content_digest,
        "validation_digest": validation_digest,
        "classes": classes,
        "samples": samples,
    }
    (root / "classes.json").write_text(
        canonical_json_bytes(classes).decode("utf-8") + "\n", encoding="utf-8"
    )
    (root / "manifest.json").write_text(
        canonical_json_bytes(manifest).decode("utf-8") + "\n", encoding="utf-8"
    )
    (root / "data.yaml").write_text(
        "train: images/train\nval: images/val\nnames:\n  - cat\n  - dog\n",
        encoding="utf-8",
    )
    return {
        "root": root,
        "classes": classes,
        "classes_digest": canonical_json_digest(classes),
        "dataset_id": dataset_id,
        "dataset_sha256": content_digest,
        "content_digest": content_digest,
        "manifest": manifest,
        "membership_contract": contract_for(manifest),
        "snapshot": _snapshot_tree(root),
    }


def _set_tree_readonly(root: Path) -> None:
    files = [p for p in root.rglob("*") if not p.is_dir()]
    dirs = [p for p in root.rglob("*") if p.is_dir()]
    for path in files:
        os.chmod(path, 0o444)
    for path in sorted(dirs, key=lambda p: len(p.parts), reverse=True):
        os.chmod(path, 0o555)
    os.chmod(root, 0o555)


def _restore_tree_writable(root: Path) -> None:
    if root.is_symlink() or not root.exists():
        return
    stack = [root]
    while stack:
        current = stack.pop()
        if current.is_symlink():
            continue
        try:
            is_dir = current.is_dir()
            os.chmod(current, 0o755 if is_dir else 0o644)
        except OSError:
            continue
        if is_dir:
            try:
                stack.extend(current.iterdir())
            except OSError:
                pass


def _cpu_env() -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["NVIDIA_VISIBLE_DEVICES"] = "void"
    return env


def _normalize_entrypoint(entrypoint: str | Sequence[str]) -> list[str]:
    cmd = shlex.split(entrypoint) if isinstance(entrypoint, str) else [str(p) for p in entrypoint]
    if not cmd:
        raise ValueError("entrypoint must not be empty")
    return cmd


def _select_probe_target(caps: TrainerCapabilitiesV1) -> tuple[str, str, list[str]]:
    chosen_arch = caps.architectures[0]
    chosen_task = chosen_arch.tasks[0]
    for arch in caps.architectures:
        if "detect" in arch.tasks:
            chosen_arch, chosen_task = arch, "detect"
            break

    def _rank(contract: str) -> int:
        lower = contract.lower()
        if chosen_task == "detect":
            if lower == "ultralytics-bbox-1":
                return 0
            if "bbox" in lower:
                return 1
            if not any(tok in lower for tok in ("polygon", "rle", "mask", "seg")):
                return 2
            return 3
        return 0 if any(tok in lower for tok in ("polygon", "rle", "mask", "seg")) else 1

    ranked = sorted(
        enumerate(caps.supported_dataset_contracts),
        key=lambda pair: (_rank(pair[1]), pair[0]),
    )
    return chosen_arch.id, chosen_task, [item for _, item in ranked]


def _terminate_process_group(proc: subprocess.Popen[str], sig: int) -> None:
    if os.name == "posix":
        try:
            os.killpg(proc.pid, sig)
            return
        except ProcessLookupError:
            return
    try:
        proc.send_signal(sig)
    except ProcessLookupError:
        pass


def _kill_process_group(proc: subprocess.Popen[str]) -> None:
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            return
        except ProcessLookupError:
            return
    try:
        proc.kill()
    except ProcessLookupError:
        pass


def _reap_killed_process(proc: subprocess.Popen[str]) -> None:
    try:
        proc.communicate(timeout=5.0)
    except subprocess.TimeoutExpired:
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()
        proc.wait()


def _run_subprocess(
    cmd: list[str], *, timeout_seconds: float, env: dict[str, str]
) -> tuple[subprocess.CompletedProcess[str] | None, str | None]:
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=(os.name == "posix"),
        )
    except OSError as exc:
        return None, f"failed to launch entrypoint: {exc}"

    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        _reap_killed_process(proc)
        return None, f"command timed out after {timeout_seconds}s"
    except BaseException:
        _kill_process_group(proc)
        _reap_killed_process(proc)
        raise
    finally:
        _kill_process_group(proc)

    return (
        subprocess.CompletedProcess(
            args=cmd,
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
        ),
        None,
    )


def _probe_validate_config(
    cmd: list[str],
    payload: dict[str, Any],
    spec_path: Path,
    *,
    timeout_seconds: float,
    env: dict[str, str],
) -> tuple[ValidateConfigResultV1 | None, str | None]:
    atomic_write_json(spec_path, payload)
    proc, err = _run_subprocess(
        [*cmd, "validate-config", "--spec", str(spec_path)],
        timeout_seconds=timeout_seconds,
        env=env,
    )
    if err is not None or proc is None:
        return None, err or "validate-config failed"
    if proc.returncode != 0:
        return None, f"validate-config exited {proc.returncode}: {proc.stderr.strip()}"
    try:
        return ValidateConfigResultV1.model_validate_json(proc.stdout), None
    except Exception as exc:
        return None, f"validate-config output is not valid ValidateConfigResultV1: {exc}"


def _check_manifest(manifest_path: Path | None) -> tuple[CheckResultV1, Any | None]:
    if manifest_path is None:
        return _skip("manifest", "manifest_path not provided"), None
    try:
        manifest = load_manifest_file(manifest_path)
    except Exception as exc:
        return _fail("manifest", f"manifest validation failed: {exc}"), None
    if manifest.manifest_schema != EXTENSION_MANIFEST_SCHEMA:
        return _fail("manifest", f"unexpected manifest_schema {manifest.manifest_schema!r}"), manifest
    if manifest.extension_point != "trainer":
        return (
            _fail("manifest", f"expected extension_point='trainer', got {manifest.extension_point!r}"),
            manifest,
        )
    if manifest.protocol_version != TRAINER_PROTOCOL_VERSION:
        return (
            _fail(
                "manifest",
                f"expected protocol_version={TRAINER_PROTOCOL_VERSION!r}, got {manifest.protocol_version!r}",
            ),
            manifest,
        )
    return (
        _ok(
            "manifest",
            "valid yoao-extension-manifest/v1 trainer manifest",
            plugin_id=manifest.plugin_id,
            protocol_version=manifest.protocol_version,
        ),
        manifest,
    )


def _check_describe(
    cmd: list[str],
    manifest: Any | None,
    *,
    timeout_seconds: float,
    env: dict[str, str],
) -> tuple[CheckResultV1, TrainerCapabilitiesV1 | None]:
    proc, err = _run_subprocess([*cmd, "describe"], timeout_seconds=timeout_seconds, env=env)
    if err is not None or proc is None:
        return _fail("describe", err or "describe failed"), None
    if proc.returncode != 0:
        return _fail("describe", f"describe exited {proc.returncode}: {proc.stderr.strip()}"), None
    try:
        caps = TrainerCapabilitiesV1.model_validate_json(proc.stdout)
    except Exception as exc:
        return _fail("describe", f"describe output failed TrainerCapabilitiesV1 validation: {exc}"), None
    if proc.stdout.rstrip("\r\n") != canonical_json_bytes(caps).decode("utf-8"):
        return _fail("describe", "describe stdout is not canonical JSON (sort_keys, compact separators)"), caps
    if manifest is not None:
        manifest_fw = manifest.extension.get("framework_id")
        if isinstance(manifest_fw, str) and manifest_fw != caps.framework_id:
            return (
                _fail(
                    "describe",
                    f"manifest extension.framework_id {manifest_fw!r} does not match describe framework_id {caps.framework_id!r}",
                ),
                caps,
            )
    return (
        _ok(
            "describe",
            "valid canonical TrainerCapabilitiesV1",
            framework_id=caps.framework_id,
            framework_version=caps.framework_version,
            supported_tasks=list(caps.supported_tasks),
            architecture_count=len(caps.architectures),
        ),
        caps,
    )


def _check_validate_config(
    cmd: list[str],
    caps: TrainerCapabilitiesV1,
    work_dir: Path,
    *,
    timeout_seconds: float,
    env: dict[str, str],
) -> tuple[CheckResultV1, dict[str, Any] | None]:
    arch_id, task, ranked_contracts = _select_probe_target(caps)
    invalid_res, invalid_err = _probe_validate_config(
        cmd,
        {
            "framework_id": caps.framework_id,
            "task": "__yoao_invalid_task__",
            "architecture_id": "__yoao_invalid_architecture__",
            "dataset_contract": "__yoao_invalid_contract__",
            "parameters": {"__yoao_unknown_param__": True},
        },
        work_dir / "validate-invalid.json",
        timeout_seconds=timeout_seconds,
        env=env,
    )
    if invalid_err is not None or invalid_res is None:
        return _fail("validate_config", f"invalid payload probe failed: {invalid_err}"), None
    if invalid_res.valid or not invalid_res.issues:
        return (
            _fail(
                "validate_config",
                "invalid payload probe unexpectedly returned valid=True or empty issues",
            ),
            None,
        )

    valid_res: ValidateConfigResultV1 | None = None
    chosen_contract: str | None = None
    last_error = "no supported_dataset_contracts accepted"
    for contract in ranked_contracts:
        res, err = _probe_validate_config(
            cmd,
            {
                "framework_id": caps.framework_id,
                "task": task,
                "architecture_id": arch_id,
                "dataset_contract": contract,
                "parameters": {},
            },
            work_dir / "validate-valid.json",
            timeout_seconds=timeout_seconds,
            env=env,
        )
        if err is not None or res is None:
            last_error = err or "validate-config failed"
            continue
        if res.valid and res.schema_id and res.schema_digest:
            valid_res, chosen_contract = res, contract
            break
        last_error = "; ".join(f"{i.field or i.code}: {i.message}" for i in res.issues)

    if valid_res is None or chosen_contract is None:
        return _fail("validate_config", f"valid payload probe rejected: {last_error}"), None

    return (
        _ok(
            "validate_config",
            "valid and invalid config probes behaved per yoao-trainer/v1",
            architecture_id=arch_id,
            task=task,
            dataset_contract=chosen_contract,
            schema_id=valid_res.schema_id,
            schema_digest=valid_res.schema_digest,
            invalid_issue_count=len(invalid_res.issues),
        ),
        {
            "architecture_id": arch_id,
            "task": task,
            "dataset_contract": chosen_contract,
            "schema_id": valid_res.schema_id,
            "schema_digest": valid_res.schema_digest,
        },
    )


def _prepare_train_spec(
    cmd: list[str],
    caps: TrainerCapabilitiesV1,
    target: dict[str, Any],
    dataset_info: dict[str, Any],
    run_dir: Path,
    *,
    desired_epochs: int,
    extra_digest_tag: str,
    timeout_seconds: float,
    env: dict[str, str],
) -> tuple[TrainSpecV1, Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    res, err = _probe_validate_config(
        cmd,
        {
            "framework_id": caps.framework_id,
            "task": target["task"],
            "architecture_id": target["architecture_id"],
            "dataset_contract": target["dataset_contract"],
            "parameters": {"epochs": desired_epochs},
        },
        run_dir / "probe-epochs.json",
        timeout_seconds=timeout_seconds,
        env=env,
    )
    params = {"epochs": desired_epochs} if (err is None and res is not None and res.valid) else {}
    config_digest = canonical_json_digest(
        {
            "framework_id": caps.framework_id,
            "task": target["task"],
            "architecture_id": target["architecture_id"],
            "dataset_contract": target["dataset_contract"],
            "parameters": params,
            "tag": extra_digest_tag,
        }
    )
    spec = TrainSpecV1(
        protocol_version=TRAINER_PROTOCOL_VERSION,
        run_id=str(uuid.uuid4()),
        attempt_id=str(uuid.uuid4()),
        framework_id=caps.framework_id,
        task=target["task"],
        architecture_id=target["architecture_id"],
        dataset_contract=target["dataset_contract"],
        dataset_dir=str(dataset_info["root"]),
        dataset_snapshot_digest=dataset_info["content_digest"],
        config_digest=config_digest,
        classes_digest=dataset_info["classes_digest"],
        schema_id=target["schema_id"],
        schema_digest=target["schema_digest"],
        parameters=params,
    )
    spec_path = run_dir / "spec.json"
    atomic_write_json(spec_path, spec)
    atomic_write_json(
        run_dir / "dataset-identity.json",
        {
            "dataset_id": dataset_info["dataset_id"],
            "dataset_sha256": dataset_info["dataset_sha256"],
        },
    )
    return spec, spec_path, run_dir / "output"


def _verify_events_jsonl(events_path: Path) -> tuple[bool, str, int]:
    if events_path.is_symlink():
        return False, "events.jsonl must not be a symlink", 0
    if not events_path.is_file():
        return False, "events.jsonl was not written", 0
    lines = events_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return False, "events.jsonl is empty", 0
    prev_seq = 0
    for idx, line in enumerate(lines, start=1):
        try:
            event = TrainEventV1.model_validate_json(line)
        except Exception as exc:
            return False, f"events.jsonl line {idx} invalid TrainEventV1: {exc}", idx - 1
        if event.seq <= prev_seq:
            return (
                False,
                f"events.jsonl seq is not strictly monotonic at line {idx}: {event.seq} <= {prev_seq}",
                idx,
            )
        prev_seq = event.seq
    return True, "", len(lines)


def _verify_file_entry(
    directory: Path, filename: str, expected_sha: Any, expected_size: Any, label: str
) -> str | None:
    target = directory / filename
    if target.is_symlink() or not target.is_file():
        return f"{label} {filename!r} is missing or is a symlink"
    if type(expected_size) is not int or expected_size < 0:
        return f"{label} {filename!r} size_bytes must be a non-negative integer"
    if digest_file(target) != expected_sha:
        return f"{label} {filename!r} sha256 mismatch"
    if target.stat().st_size != expected_size:
        return f"{label} {filename!r} size_bytes mismatch"
    return None


def _verify_checkpoint_metadata(output_dir: Path) -> str | None:
    meta_path = output_dir / "checkpoint.json"
    if meta_path.is_symlink():
        return "checkpoint.json must not be a symlink"
    if not meta_path.exists():
        return None
    if not meta_path.is_file():
        return "checkpoint.json must be a regular file"
    try:
        doc = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"checkpoint.json is not valid JSON: {exc}"
    if not isinstance(doc, dict):
        return "checkpoint.json must be a JSON object"
    name = doc.get("name")
    match = _CHECKPOINT_NAME_RE.fullmatch(name) if isinstance(name, str) else None
    if match is None:
        return f"checkpoint.json name {name!r} does not match checkpoint-NNNN.<ext>"
    if doc.get("role") != "checkpoint":
        return f"checkpoint.json role must be 'checkpoint', got {doc.get('role')!r}"
    epoch = doc.get("epoch")
    if type(epoch) is not int or not 1 <= epoch <= 9999:
        return f"checkpoint.json epoch must be int in 1..9999, got {epoch!r}"
    if int(match.group(1)) != epoch:
        return f"checkpoint.json filename epoch {int(match.group(1))} does not match epoch {epoch}"
    size_bytes = doc.get("size_bytes")
    if type(size_bytes) is not int or size_bytes <= 0:
        return f"checkpoint.json size_bytes must be a positive integer, got {size_bytes!r}"
    return _verify_file_entry(
        output_dir, name, doc.get("sha256"), size_bytes, "checkpoint file"
    )


def _verify_input_membership(
    output_dir: Path,
    receipt: TrainReceiptV1,
    spec: TrainSpecV1,
    dataset_info: dict[str, Any],
    caps: TrainerCapabilitiesV1,
) -> CheckResultV1:
    role_declared = [art for art in receipt.artifacts if art.role == "input_membership"]
    name_declared = [art for art in receipt.artifacts if art.filename == "input-membership.json"]
    path = output_dir / "input-membership.json"
    file_present = path.exists() or path.is_symlink()

    if len(role_declared) > 1:
        return _fail(
            "input_membership",
            "TrainReceiptV1 declared multiple input_membership artifacts",
        )
    if role_declared and role_declared[0].filename != "input-membership.json":
        return _fail(
            "input_membership",
            f"input_membership artifact must be named 'input-membership.json', got {role_declared[0].filename!r}",
        )
    if name_declared and name_declared[0].role != "input_membership":
        return _fail(
            "input_membership",
            f"artifact 'input-membership.json' must have role='input_membership', got {name_declared[0].role!r}",
        )

    if not role_declared:
        if file_present:
            return _fail(
                "input_membership",
                "input-membership.json exists on disk but is not declared in TrainReceiptV1.artifacts with role='input_membership'",
            )
        if "input_membership" in caps.policy_digests:
            return _fail(
                "input_membership",
                "capabilities declare policy_digests['input_membership'] but TrainReceiptV1 omitted input-membership.json",
            )
        return _skip("input_membership", "input-membership.json not declared by plugin")

    if path.is_symlink() or not path.is_file():
        return _fail("input_membership", f"declared input membership file {path.name!r} missing or symlink")
    try:
        proof = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return _fail("input_membership", f"input-membership.json is invalid JSON: {exc}")
    expected_contract = dataset_info["membership_contract"]
    if not isinstance(proof, dict):
        return _fail("input_membership", "input-membership.json must be a JSON object")
    if set(proof) != _INPUT_MEMBERSHIP_FIELDS:
        return _fail(
            "input_membership",
            f"input-membership.json keys {sorted(proof)!r} do not match expected schema keys",
        )
    if type(proof["schema"]) is not int or proof["schema"] != 1:
        return _fail("input_membership", "input-membership.json schema must be integer 1")
    if proof["policy_digest"] != POLICY_DIGEST:
        return _fail(
            "input_membership",
            f"input-membership.json policy_digest {proof['policy_digest']!r} != expected {POLICY_DIGEST!r}",
        )
    if proof["members_digest"] != expected_contract["members_digest"]:
        return _fail(
            "input_membership",
            "input-membership.json members_digest does not match frozen dataset contract",
        )
    counts = proof["counts"]
    if (
        not isinstance(counts, dict)
        or not all(type(v) is int for v in counts.values())
        or counts != expected_contract["counts"]
    ):
        return _fail(
            "input_membership",
            f"input-membership.json counts {counts!r} != {expected_contract['counts']!r}",
        )
    if proof["run_id"] != spec.run_id or proof["attempt_id"] != spec.attempt_id:
        return _fail("input_membership", "input-membership.json run_id/attempt_id mismatch")
    if proof["config_digest"] != spec.config_digest:
        return _fail("input_membership", "input-membership.json config_digest mismatch")
    if (
        proof["dataset_id"] != dataset_info["dataset_id"]
        or proof["dataset_sha256"] != dataset_info["dataset_sha256"]
    ):
        return _fail("input_membership", "input-membership.json dataset_id/dataset_sha256 mismatch")
    members = proof["members"]
    if not isinstance(members, dict):
        return _fail("input_membership", "input-membership.json members must be a JSON object")
    try:
        members_digest = canonical_json_digest(members)
    except ValueError as exc:
        return _fail("input_membership", f"input-membership.json members invalid JSON values: {exc}")
    if members_digest != proof["members_digest"]:
        return _fail("input_membership", "input-membership.json members digest does not match members body")
    return _ok(
        "input_membership",
        "input-membership.json matches POLICY_DIGEST and frozen members_digest",
        policy_digest=POLICY_DIGEST,
        members_digest=proof["members_digest"],
        counts=proof["counts"],
    )


def _check_train_and_membership(
    cmd: list[str],
    caps: TrainerCapabilitiesV1,
    target: dict[str, Any],
    dataset_info: dict[str, Any],
    work_dir: Path,
    *,
    timeout_seconds: float,
    env: dict[str, str],
) -> tuple[CheckResultV1, CheckResultV1]:
    spec, spec_path, train_out = _prepare_train_spec(
        cmd,
        caps,
        target,
        dataset_info,
        work_dir / "train-run",
        desired_epochs=2,
        extra_digest_tag="train",
        timeout_seconds=timeout_seconds,
        env=env,
    )
    im_blocked = _fail("input_membership", "skipped because train check failed")
    proc, err = _run_subprocess(
        [*cmd, "train", "--spec", str(spec_path), "--output", str(train_out)],
        timeout_seconds=timeout_seconds,
        env=env,
    )
    if err is not None or proc is None:
        return _fail("train", err or "train failed"), im_blocked
    if proc.returncode != 0:
        return _fail("train", f"train exited {proc.returncode}: {proc.stderr.strip()}"), im_blocked

    result_path = train_out / "result.json"
    if result_path.is_symlink():
        return _fail("train", "result.json must not be a symlink"), im_blocked
    if not result_path.is_file():
        return _fail("train", "result.json was not written"), im_blocked
    try:
        receipt = TrainReceiptV1.model_validate_json(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return _fail("train", f"result.json failed TrainReceiptV1 validation: {exc}"), im_blocked
    if receipt.status != "succeeded":
        return _fail("train", f"expected status='succeeded', got {receipt.status!r}"), im_blocked

    events_ok, events_err, event_count = _verify_events_jsonl(train_out / "events.jsonl")
    if not events_ok:
        return _fail("train", events_err), im_blocked

    for artifact in receipt.artifacts:
        if artifact.filename in SUPERVISOR_OWNED_ARTIFACTS:
            return (
                _fail("train", f"TrainReceiptV1 declared supervisor-owned artifact {artifact.filename!r}"),
                im_blocked,
            )
        file_err = _verify_file_entry(
            train_out, artifact.filename, artifact.sha256, artifact.size_bytes, "declared artifact"
        )
        if file_err is not None:
            return _fail("train", file_err), im_blocked

    ckpt_err = _verify_checkpoint_metadata(train_out)
    if ckpt_err is not None:
        return _fail("train", ckpt_err), im_blocked

    leftover_tmp = sorted(p.name for p in train_out.glob(".tmp-*"))
    if leftover_tmp:
        return _fail("train", f"leftover temporary files in output: {leftover_tmp}"), im_blocked
    if _snapshot_tree(dataset_info["root"]) != dataset_info["snapshot"]:
        return _fail("train", "train mutated read-only frozen dataset directory"), im_blocked

    return (
        _ok(
            "train",
            "train succeeded with monotonic events and verified artifacts",
            event_count=event_count,
            artifact_count=len(receipt.artifacts),
            artifacts=[a.filename for a in receipt.artifacts],
        ),
        _verify_input_membership(train_out, receipt, spec, dataset_info, caps),
    )


def _check_cancel(
    cmd: list[str],
    caps: TrainerCapabilitiesV1,
    target: dict[str, Any],
    dataset_info: dict[str, Any],
    work_dir: Path,
    *,
    timeout_seconds: float,
    env: dict[str, str],
) -> CheckResultV1:
    _spec, spec_path, cancel_out = _prepare_train_spec(
        cmd,
        caps,
        target,
        dataset_info,
        work_dir / "cancel-run",
        desired_epochs=25,
        extra_digest_tag="cancel",
        timeout_seconds=timeout_seconds,
        env=env,
    )
    try:
        proc = subprocess.Popen(
            [*cmd, "train", "--spec", str(spec_path), "--output", str(cancel_out)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=(os.name == "posix"),
        )
    except OSError as exc:
        return _fail("cancel", f"failed to launch entrypoint for cancel check: {exc}")

    events_path = cancel_out / "events.jsonl"
    wait_start = time.monotonic()
    signal_wait_limit = min(max(timeout_seconds * 0.5, 2.0), 15.0)
    sent_signal = False
    try:
        while proc.poll() is None:
            if (
                not events_path.is_symlink()
                and events_path.is_file()
                and events_path.stat().st_size > 0
            ) or (time.monotonic() - wait_start >= signal_wait_limit):
                _terminate_process_group(proc, signal.SIGTERM)
                sent_signal = True
                break
            time.sleep(0.005)
        cancel_grace = min(timeout_seconds, 5.0)
        try:
            _stdout, stderr = proc.communicate(timeout=cancel_grace)
        except subprocess.TimeoutExpired:
            _kill_process_group(proc)
            _reap_killed_process(proc)
            return _fail("cancel", f"process ignored SIGTERM and did not exit within {cancel_grace:.1f}s")
    finally:
        _kill_process_group(proc)
        if proc.poll() is None:
            _reap_killed_process(proc)

    if not sent_signal:
        return _fail("cancel", f"process exited ({proc.returncode}) before SIGTERM could be delivered")
    if proc.returncode != 130:
        suffix = f": {stderr.strip()}" if stderr.strip() else ""
        return _fail("cancel", f"expected exit code 130 after SIGTERM, got {proc.returncode}{suffix}")
    result_path = cancel_out / "result.json"
    if result_path.is_symlink():
        return _fail("cancel", "cancel result.json must not be a symlink")
    if not result_path.is_file():
        return _fail("cancel", "SIGTERM exited 130 but omitted result.json")
    try:
        receipt = TrainReceiptV1.model_validate_json(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return _fail("cancel", f"cancel result.json invalid TrainReceiptV1: {exc}")
    if receipt.status != "cancelled":
        return _fail("cancel", f"expected result.json status='cancelled', got {receipt.status!r}")
    return _ok(
        "cancel",
        "SIGTERM exited 130 and persisted status='cancelled' result.json",
        exit_code=proc.returncode,
        status=receipt.status,
    )


def _check_preview(
    cmd: list[str],
    caps: TrainerCapabilitiesV1,
    target: dict[str, Any],
    dataset_info: dict[str, Any],
    work_dir: Path,
    *,
    timeout_seconds: float,
    env: dict[str, str],
) -> CheckResultV1:
    preview_dir = work_dir / "preview-run"
    preview_out = preview_dir / "output"
    preview_dir.mkdir(parents=True, exist_ok=True)
    sample = dataset_info["manifest"]["samples"][0]
    for rel_key in ("image_path", "label_path"):
        dst = preview_dir / sample[rel_key]
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((dataset_info["root"] / sample[rel_key]).read_bytes())

    spec = PreviewSpecV1(
        protocol_version=TRAINER_PROTOCOL_VERSION,
        preview_id="check-preview-1",
        framework_id=caps.framework_id,
        task=target["task"],
        architecture_id=target["architecture_id"],
        input_digest=canonical_json_digest({"sample": sample["sha256"], "seed": 42}),
        config_digest=canonical_json_digest(
            {
                "framework_id": caps.framework_id,
                "task": target["task"],
                "architecture_id": target["architecture_id"],
            }
        ),
        dataset_snapshot_digest=dataset_info["content_digest"],
        policy_digest=caps.policy_digests.get("preview", canonical_json_digest({"preview_check": 1})),
        seed=42,
        epoch=1,
        count=1,
        sources=[
            PreviewSourceV1(
                task_id=sample["task_id"],
                filename=sample["image_path"],
                sha256=sample["sha256"],
                width=sample["width"],
                height=sample["height"],
                size_bytes=sample["size_bytes"],
                box_count=1,
                label_path=sample["label_path"],
                label_sha256=sample["label_sha256"],
            )
        ],
        parameters={},
        native_values={"device": "cpu", "workers": 0},
        pool_summary={"candidate_count": 1},
    )
    spec_path = preview_dir / "preview-spec.json"
    atomic_write_json(spec_path, spec)

    proc, err = _run_subprocess(
        [*cmd, "preview", "--spec", str(spec_path), "--output", str(preview_out)],
        timeout_seconds=timeout_seconds,
        env=env,
    )
    if err is not None or proc is None:
        return _fail("preview", err or "preview failed")
    if proc.returncode != 0:
        return _fail("preview", f"preview exited {proc.returncode}: {proc.stderr.strip()}")
    report_path = preview_out / "report.json"
    if report_path.is_symlink():
        return _fail("preview", "report.json must not be a symlink")
    if not report_path.is_file():
        return _fail("preview", "preview did not write report.json")
    try:
        raw_report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return _fail("preview", f"report.json invalid JSON: {exc}")
    if isinstance(raw_report, dict) and raw_report.get("cuda_initialized") is not False:
        return _fail(
            "preview",
            f"preview reported cuda_initialized={raw_report.get('cuda_initialized')!r}; must be False",
        )
    try:
        report = PreviewReportV1.model_validate(raw_report)
    except Exception as exc:
        return _fail("preview", f"report.json failed PreviewReportV1 validation: {exc}")
    if report.status != "succeeded" or report.cuda_initialized:
        return _fail(
            "preview",
            f"preview status={report.status!r}, cuda_initialized={report.cuda_initialized!r}",
        )
    if report.input_digest != spec.input_digest or report.policy_digest != spec.policy_digest:
        return _fail("preview", "preview report input_digest or policy_digest mismatch")
    for item in report.outputs:
        file_err = _verify_file_entry(
            preview_out, item.filename, item.sha256, item.size_bytes, "preview output"
        )
        if file_err is not None:
            return _fail("preview", file_err)
    return _ok(
        "preview",
        "CPU-only preview succeeded with verified output digests",
        output_count=len(report.outputs),
        cuda_initialized=False,
    )


def check_trainer_plugin(
    entrypoint: str | Sequence[str],
    *,
    manifest_path: str | Path | None = None,
    skip_train: bool = False,
    check_preview: bool = False,
    timeout_seconds: float = 60.0,
) -> TrainerCheckReport:
    """Verify a yoao-trainer/v1 plugin offline against synthetic read-only data."""
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be a finite positive number")
    cmd = _normalize_entrypoint(entrypoint)
    env = _cpu_env()
    checks: dict[str, CheckResultV1] = {}

    manifest_res, manifest_obj = _check_manifest(
        Path(manifest_path) if manifest_path is not None else None
    )
    checks["manifest"] = manifest_res
    describe_res, caps = _check_describe(cmd, manifest_obj, timeout_seconds=timeout_seconds, env=env)
    checks["describe"] = describe_res

    if caps is None:
        checks["validate_config"] = _fail("validate_config", "skipped because describe failed")
        if not skip_train:
            for name in ("train", "input_membership", "cancel"):
                checks[name] = _fail(name, "skipped because describe failed")
        if check_preview:
            checks["preview"] = _fail("preview", "skipped because describe failed")
        return TrainerCheckReport(
            entrypoint=cmd, passed=all(c.passed for c in checks.values()), checks=checks
        )

    with tempfile.TemporaryDirectory(prefix="yoao-trainer-check-") as temp_root:
        work_dir = Path(temp_root)
        dataset_root = work_dir / "dataset"
        try:
            validate_res, target = _check_validate_config(
                cmd, caps, work_dir, timeout_seconds=timeout_seconds, env=env
            )
            checks["validate_config"] = validate_res

            dataset_info: dict[str, Any] | None = None
            if not skip_train or check_preview:
                dataset_info = _create_synthetic_dataset(dataset_root)
                _set_tree_readonly(dataset_root)

            if not skip_train:
                if target is None or dataset_info is None:
                    for name in ("train", "input_membership", "cancel"):
                        checks[name] = _fail(name, "skipped because validate_config failed")
                else:
                    train_res, membership_res = _check_train_and_membership(
                        cmd, caps, target, dataset_info, work_dir,
                        timeout_seconds=timeout_seconds, env=env,
                    )
                    checks["train"] = train_res
                    checks["input_membership"] = membership_res
                    checks["cancel"] = _check_cancel(
                        cmd, caps, target, dataset_info, work_dir,
                        timeout_seconds=timeout_seconds, env=env,
                    )

            if check_preview:
                if target is None or dataset_info is None:
                    checks["preview"] = _fail("preview", "skipped because validate_config failed")
                else:
                    checks["preview"] = _check_preview(
                        cmd, caps, target, dataset_info, work_dir,
                        timeout_seconds=timeout_seconds, env=env,
                    )
        finally:
            _restore_tree_writable(work_dir)

    return TrainerCheckReport(
        entrypoint=cmd, passed=all(c.passed for c in checks.values()), checks=checks
    )


__all__ = [
    "CheckResultV1",
    "TrainerCheckReport",
    "check_trainer_plugin",
]
