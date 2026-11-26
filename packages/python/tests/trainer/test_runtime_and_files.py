from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import stat
import sys
import types
from typing import Any

import pytest

from yoao_plugin_sdk.trainer import (
    SUPERVISOR_OWNED_ARTIFACTS,
    TRAINER_PROTOCOL_VERSION,
    CheckpointSlotStore,
    DatasetContractError,
    EventEmitter,
    InputMismatchError,
    NativeConfigurationMismatch,
    NativeRuntimeMismatch,
    TrainArtifactV1,
    TrainerError,
    TrainEventV1,
    TrainReceiptV1,
    TrainSpecV1,
    atomic_write_bytes,
    build_artifact_entry,
    classify_failure,
    digest_file,
    immutable_copy,
    load_dataset_identity,
    run_with_signal_receipt,
    sanitize_failure_message,
)
import yoao_plugin_sdk.trainer.runtime as runtime_mod

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64


def _sample_spec(dataset_dir: Path) -> dict[str, Any]:
    return TrainSpecV1(
        protocol_version=TRAINER_PROTOCOL_VERSION,
        run_id="run-1",
        attempt_id="attempt-1",
        framework_id="demo",
        task="detect",
        architecture_id="demo-n",
        dataset_contract="ultralytics-bbox-1",
        dataset_dir=str(dataset_dir),
        dataset_snapshot_digest=DIGEST_A,
        config_digest=DIGEST_B,
        classes_digest=DIGEST_C,
        parameters={"epochs": 2},
    ).model_dump(mode="json")


def test_atomic_write_bytes_and_cleanup_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "nested" / "weights.bin"
    atomic_write_bytes(target, b"hello-weights", mode=0o600)
    assert target.read_bytes() == b"hello-weights"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600

    def _boom(_src: str, _dst: Path) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", _boom)
    fail_target = tmp_path / "fail" / "weights.bin"
    with pytest.raises(OSError, match="disk full"):
        atomic_write_bytes(fail_target, b"abc")
    assert not fail_target.exists()
    assert not list(fail_target.parent.glob(".tmp-*"))


def test_immutable_copy_idempotency_conflict_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src_a = tmp_path / "a.bin"
    src_a.write_bytes(b"payload-a")
    src_b = tmp_path / "b.bin"
    src_b.write_bytes(b"payload-b")
    dst = tmp_path / "out" / "best.pt"

    first = immutable_copy(src_a, dst)
    assert first["name"] == "best.pt"
    assert first["sha256"] == digest_file(src_a)
    assert first["size_bytes"] == len(b"payload-a")

    second = immutable_copy(src_a, dst)
    assert second == first

    with pytest.raises(ValueError, match="different content"):
        immutable_copy(src_b, dst)
    assert dst.read_bytes() == b"payload-a"

    def _boom(_src: str, _dst: Path) -> None:
        raise RuntimeError("replace failed")

    monkeypatch.setattr(os, "replace", _boom)
    dst_new = tmp_path / "out" / "other.pt"
    with pytest.raises(RuntimeError, match="replace failed"):
        immutable_copy(src_a, dst_new)
    assert not dst_new.exists()
    assert not list(dst_new.parent.glob(".tmp-*"))


def test_checkpoint_slot_store_rotation_validation_and_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for bad_ext in ("", "pt", "..pt", ".pt/a", f".{'a' * 33}", ".pt;rm"):
        with pytest.raises(ValueError):
            CheckpointSlotStore(tmp_path, extension=bad_ext)
    for bad_meta in ("", "manifest.json", "result.json", "../checkpoint.json"):
        with pytest.raises(ValueError):
            CheckpointSlotStore(tmp_path, metadata_filename=bad_meta)

    store = CheckpointSlotStore(tmp_path)
    for bad_epoch in (0, -1, 10000, True, False, 1.0):
        with pytest.raises(ValueError):
            store.slot_filename(bad_epoch)  # type: ignore[arg-type]

    ep1 = store.commit_via_writer(1, lambda stream: stream.write(b"ckpt-epoch-1"))
    slot1 = tmp_path / "checkpoint-0001.pt"
    meta_path = tmp_path / "checkpoint.json"
    assert ep1["name"] == "checkpoint-0001.pt"
    assert ep1["role"] == "checkpoint"
    assert ep1["epoch"] == 1
    assert ep1["sha256"] == digest_file(slot1)
    assert ep1["size_bytes"] == len(b"ckpt-epoch-1")
    assert json.loads(meta_path.read_text(encoding="utf-8")) == ep1
    assert not (tmp_path / "checkpoint-staging.pt").exists()

    # Failing writer must retain epoch 1 and clean up staging.
    def _failing_writer(stream: Any) -> None:
        stream.write(b"partial")
        raise RuntimeError("writer crash")

    with pytest.raises(RuntimeError, match="writer crash"):
        store.commit_via_writer(2, _failing_writer)
    assert slot1.is_file()
    assert not (tmp_path / "checkpoint-0002.pt").exists()
    assert not (tmp_path / "checkpoint-staging.pt").exists()
    assert json.loads(meta_path.read_text(encoding="utf-8"))["name"] == "checkpoint-0001.pt"

    # Failing checkpoint.json update must keep epoch 1 intact and not unlink it.
    real_atomic_write_json = runtime_mod.atomic_write_json

    def _fail_meta(path: Any, payload: Any, **kwargs: Any) -> None:
        raise OSError("meta write failed")

    monkeypatch.setattr(runtime_mod, "atomic_write_json", _fail_meta)
    with pytest.raises(OSError, match="meta write failed"):
        store.commit_via_writer(2, lambda stream: stream.write(b"ckpt-epoch-2"))
    assert slot1.is_file()
    assert json.loads(meta_path.read_text(encoding="utf-8"))["name"] == "checkpoint-0001.pt"
    assert store.current is not None and store.current["name"] == "checkpoint-0001.pt"
    assert not (tmp_path / "checkpoint-staging.pt").exists()
    (tmp_path / "checkpoint-0002.pt").unlink()

    # Successful rotation updates checkpoint.json first, then unlinks slot 1.
    monkeypatch.setattr(runtime_mod, "atomic_write_json", real_atomic_write_json)
    ep2 = store.commit_via_writer(2, lambda stream: stream.write(b"ckpt-epoch-2"))
    slot2 = tmp_path / "checkpoint-0002.pt"
    assert slot2.is_file()
    assert not slot1.exists()
    assert json.loads(meta_path.read_text(encoding="utf-8")) == ep2

    # Same-epoch idempotent commit_file does not unlink its own slot.
    resume_copy = tmp_path / "resume-src.pt"
    resume_copy.write_bytes(b"ckpt-epoch-2")
    ep2_again = store.commit_file(resume_copy, 2)
    assert ep2_again == ep2
    assert slot2.is_file()


def test_event_emitter_monotonic_seq_and_helpers(tmp_path: Path) -> None:
    emitter = EventEmitter(tmp_path)
    assert emitter.path == tmp_path / "events.jsonl"

    e1 = emitter.phase("preparing")
    e2 = emitter.epoch(epoch=1, total_epochs=4, metrics={"loss": -0.0, "map50": 0.5})
    e3 = emitter.progress(progress=0.5, epoch=2, total_epochs=4, detail={"step": 10})
    e4 = emitter.checkpoint(
        epoch=1, total_epochs=4, filename="checkpoint-0001.pt", sha256=DIGEST_A
    )
    e5 = emitter.warning("slow disk", phase="training")

    with pytest.raises(ValueError):
        emitter.epoch(epoch=5, total_epochs=4, metrics={})
    e6 = emitter.phase("evaluating")

    assert [e.seq for e in (e1, e2, e3, e4, e5, e6)] == [1, 2, 3, 4, 5, 6]
    assert e2.progress == 0.25
    assert e4.detail == {"filename": "checkpoint-0001.pt", "sha256": DIGEST_A}
    assert e5.detail == {"message": "slow disk"}

    lines = emitter.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6
    parsed = [TrainEventV1.model_validate_json(line) for line in lines]
    assert [row.seq for row in parsed] == [1, 2, 3, 4, 5, 6]

    explicit = EventEmitter(tmp_path / "custom.jsonl")
    assert explicit.path == tmp_path / "custom.jsonl"
    explicit.phase("init")
    assert (tmp_path / "custom.jsonl").is_file()


def test_build_artifact_entry_rejects_supervisor_owned(tmp_path: Path) -> None:
    for owned in sorted(SUPERVISOR_OWNED_ARTIFACTS):
        path = tmp_path / owned
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(NativeConfigurationMismatch) as exc_info:
            build_artifact_entry("config", path)
        assert exc_info.value.failure_code == "config_mismatch"

    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights-bytes")
    entry = build_artifact_entry("weights", weights)
    assert isinstance(entry, TrainArtifactV1)
    assert entry.filename == "best.pt"
    assert entry.sha256 == digest_file(weights)
    assert entry.size_bytes == len(b"weights-bytes")


def test_error_hierarchy_classification_and_sanitization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ds_err = DatasetContractError("invalid_label", "bad bbox")
    assert isinstance(ds_err, InputMismatchError)
    assert isinstance(ds_err, TrainerError)
    assert isinstance(ds_err, ValueError)
    assert ds_err.code == "invalid_label"
    assert classify_failure(ds_err) == "input_mismatch"

    rt_err = NativeRuntimeMismatch("wrong version")
    assert isinstance(rt_err, NativeConfigurationMismatch)
    assert classify_failure(rt_err) == "runtime_mismatch"
    assert classify_failure(TrainerError("custom", failure_code="my_code")) == "my_code"
    assert classify_failure(RuntimeError("boom")) == "native_exception"

    fake_cuda = types.SimpleNamespace()

    class FakeOutOfMemoryError(RuntimeError):
        pass

    fake_cuda.OutOfMemoryError = FakeOutOfMemoryError
    fake_torch = types.SimpleNamespace(cuda=fake_cuda)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert classify_failure(FakeOutOfMemoryError("OOM")) == "cuda_oom"

    noisy = RuntimeError("\x00bad\n\tmessage\x7f  " + ("x" * 2500))
    sanitized = sanitize_failure_message(noisy)
    assert len(sanitized) <= 2048
    assert not any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in sanitized)
    assert sanitized.startswith("bad message x")
    assert sanitize_failure_message(RuntimeError("  \n\t\x00 "), fallback="fallback msg") == "fallback msg"
    assert sanitize_failure_message(RuntimeError("")) == "RuntimeError"


def test_load_dataset_identity_sidecar(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text("{}", encoding="utf-8")
    assert load_dataset_identity(None) == {}
    assert load_dataset_identity(spec_path) == {}

    sidecar = tmp_path / "dataset-identity.json"
    sidecar.write_text(json.dumps({"dataset_id": "d-1"}), encoding="utf-8")
    assert load_dataset_identity(spec_path) == {"dataset_id": "d-1"}

    sidecar.write_text('["not-an-object"]', encoding="utf-8")
    with pytest.raises(InputMismatchError):
        load_dataset_identity(spec_path)

    sidecar.write_text("{broken json", encoding="utf-8")
    with pytest.raises(InputMismatchError):
        load_dataset_identity(spec_path)


def test_run_with_signal_receipt_success_cancel_and_cuda_oom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_sample_spec(tmp_path / "dataset")), encoding="utf-8")

    # Happy path writes result.json when run_fn returns TrainReceiptV1.
    out_ok = tmp_path / "out-ok"
    before_sigterm = signal.getsignal(signal.SIGTERM)

    def _run_ok(spec: TrainSpecV1, output_dir: Path, sp: Path) -> TrainReceiptV1:
        assert sp == spec_path
        weights = output_dir / "best.pt"
        weights.write_bytes(b"ok")
        return TrainReceiptV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            status="succeeded",
            artifacts=[build_artifact_entry("weights", weights)],
        )

    assert run_with_signal_receipt(spec_path, out_ok, _run_ok) == 0
    assert signal.getsignal(signal.SIGTERM) is before_sigterm
    receipt_ok = TrainReceiptV1.model_validate_json(
        (out_ok / "result.json").read_text(encoding="utf-8")
    )
    assert receipt_ok.status == "succeeded"

    # SIGTERM cancels cleanly with exit code 130 and status="cancelled".
    out_cancel = tmp_path / "out-cancel"

    def _run_sigterm(spec: TrainSpecV1, output_dir: Path, sp: Path) -> TrainReceiptV1:
        os.kill(os.getpid(), signal.SIGTERM)
        raise AssertionError("unreachable")

    assert run_with_signal_receipt(spec_path, out_cancel, _run_sigterm) == 130
    assert signal.getsignal(signal.SIGTERM) is before_sigterm
    receipt_cancel = TrainReceiptV1.model_validate_json(
        (out_cancel / "result.json").read_text(encoding="utf-8")
    )
    assert receipt_cancel.status == "cancelled"

    # Simulated torch.cuda.OutOfMemoryError via sys.modules yields failure_code="cuda_oom".
    class FakeOOM(RuntimeError):
        pass

    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(cuda=types.SimpleNamespace(OutOfMemoryError=FakeOOM)),
    )
    out_oom = tmp_path / "out-oom"

    def _run_oom(spec: TrainSpecV1, output_dir: Path, sp: Path) -> TrainReceiptV1:
        raise FakeOOM("CUDA out of memory\nallocated 24GiB")

    assert run_with_signal_receipt(spec_path, out_oom, _run_oom) == 1
    assert signal.getsignal(signal.SIGTERM) is before_sigterm
    receipt_oom = TrainReceiptV1.model_validate_json(
        (out_oom / "result.json").read_text(encoding="utf-8")
    )
    assert receipt_oom.status == "failed"
    assert receipt_oom.failure_code == "cuda_oom"
    assert receipt_oom.failure_message == "CUDA out of memory allocated 24GiB"
