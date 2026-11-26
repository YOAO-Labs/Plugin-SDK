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
from pydantic import ValidationError

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
    atomic_write_json,
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


def test_immutable_copy_rejects_symlinks_non_files_and_missing_source(tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")

    # Live symlink destination rejected even when target content matches.
    target_real = tmp_path / "real.bin"
    target_real.write_bytes(b"payload")
    live_link = tmp_path / "out" / "live.pt"
    live_link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target_real, live_link)
    with pytest.raises(ValueError, match="symlink"):
        immutable_copy(src, live_link)

    # Dangling symlink destination rejected before exists().
    dangling_link = tmp_path / "out" / "dangling.pt"
    os.symlink(tmp_path / "does-not-exist.bin", dangling_link)
    with pytest.raises(ValueError, match="symlink"):
        immutable_copy(src, dangling_link)

    # Non-regular file (directory) destination rejected.
    dir_dst = tmp_path / "out" / "dir.pt"
    dir_dst.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        immutable_copy(src, dir_dst)

    # Missing source fails before creating a temp file in destination directory.
    clean_dir = tmp_path / "clean-out"
    with pytest.raises(FileNotFoundError):
        immutable_copy(tmp_path / "missing.bin", clean_dir / "best.pt")
    assert not list(clean_dir.glob(".tmp-*"))


def test_atomic_writers_handle_ebadf_after_fdopen_and_unlink_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_fdopen = os.fdopen

    def _fdopen_close_then_interrupt(fd: int, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        # Simulate ownership transfer closing fd right before an async KeyboardInterrupt.
        stream = real_fdopen(fd, mode, *args, **kwargs)
        stream.close()
        raise KeyboardInterrupt("async cancel right after fdopen")

    monkeypatch.setattr(os, "fdopen", _fdopen_close_then_interrupt)

    bytes_dir = tmp_path / "bytes-case"
    with pytest.raises(KeyboardInterrupt, match="async cancel"):
        atomic_write_bytes(bytes_dir / "a.bin", b"abc")
    assert not list(bytes_dir.glob(".tmp-*"))

    json_dir = tmp_path / "json-case"
    with pytest.raises(KeyboardInterrupt, match="async cancel"):
        atomic_write_json(json_dir / "a.json", {"ok": True})
    assert not list(json_dir.glob(".tmp-*"))

    src = tmp_path / "copy-src.bin"
    src.write_bytes(b"copy-data")
    copy_dir = tmp_path / "copy-case"
    with pytest.raises(KeyboardInterrupt, match="async cancel"):
        immutable_copy(src, copy_dir / "a.bin")
    assert not list(copy_dir.glob(".tmp-*"))


def test_event_emitter_epoch_zero_total_raises_validation_error(tmp_path: Path) -> None:
    emitter = EventEmitter(tmp_path)
    with pytest.raises(ValidationError):
        emitter.epoch(epoch=0, total_epochs=0, metrics={})
    with pytest.raises(ValidationError):
        emitter.epoch(epoch=1, total_epochs=-2, metrics={})
    assert emitter.seq == 0
    assert not emitter.path.exists()


def test_classify_failure_sanitizes_untrusted_failure_code(tmp_path: Path) -> None:
    dirty = TrainerError("boom", failure_code="  \x00bad\ncode\t\x7f" + ("x" * 100))
    code = classify_failure(dirty)
    assert len(code) <= 64
    assert code == code.strip()
    assert not any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in code)

    # Trailing space at index 64 must be stripped cleanly.
    boundary = TrainerError("boom", failure_code=("a" * 63) + " bcd")
    assert classify_failure(boundary) == "a" * 63

    # Control-only or whitespace-only falls back to native_exception.
    assert classify_failure(TrainerError("boom", failure_code=" \x00\n\t\x7f ")) == "native_exception"

    # End-to-end: run_with_signal_receipt persists valid TrainReceiptV1 without ValidationError.
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_sample_spec(tmp_path / "dataset")), encoding="utf-8")
    out_dir = tmp_path / "out-dirty-code"

    def _run_dirty(spec: TrainSpecV1, output_dir: Path, sp: Path) -> TrainReceiptV1:
        raise dirty

    assert run_with_signal_receipt(spec_path, out_dir, _run_dirty) == 1
    receipt = TrainReceiptV1.model_validate_json((out_dir / "result.json").read_text(encoding="utf-8"))
    assert receipt.status == "failed"
    assert receipt.failure_code == code


def test_run_with_signal_receipt_reentrant_sigterm_stale_receipt_and_partial_signal_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_sample_spec(tmp_path / "dataset")), encoding="utf-8")
    before_sigterm = signal.getsignal(signal.SIGTERM)
    before_sigint = signal.getsignal(signal.SIGINT)

    # 1. Stale result.json is overwritten when run_fn returns a fresh TrainReceiptV1.
    out_stale = tmp_path / "out-stale"
    out_stale.mkdir(parents=True, exist_ok=True)
    stale_receipt = TrainReceiptV1(
        protocol_version=TRAINER_PROTOCOL_VERSION,
        status="failed",
        failure_code="stale_error",
        failure_message="from earlier run",
    )
    atomic_write_json(out_stale / "result.json", stale_receipt)

    def _run_fresh(spec: TrainSpecV1, output_dir: Path, sp: Path) -> TrainReceiptV1:
        return TrainReceiptV1(
            protocol_version=TRAINER_PROTOCOL_VERSION,
            status="succeeded",
            artifacts=[],
        )

    assert run_with_signal_receipt(spec_path, out_stale, _run_fresh) == 0
    fresh_written = TrainReceiptV1.model_validate_json(
        (out_stale / "result.json").read_text(encoding="utf-8")
    )
    assert fresh_written.status == "succeeded"

    # 2. Re-entrant SIGTERM during terminal receipt write does not abort persistence.
    real_atomic_write_json = runtime_mod.atomic_write_json
    sent_reentrant = {"count": 0}

    def _reentrant_write(path: Any, payload: Any, **kwargs: Any) -> None:
        sent_reentrant["count"] += 1
        os.kill(os.getpid(), signal.SIGTERM)
        os.kill(os.getpid(), signal.SIGINT)
        real_atomic_write_json(path, payload, **kwargs)

    monkeypatch.setattr(runtime_mod, "atomic_write_json", _reentrant_write)

    out_cancel = tmp_path / "out-reentrant-cancel"

    def _run_cancel(spec: TrainSpecV1, output_dir: Path, sp: Path) -> TrainReceiptV1:
        os.kill(os.getpid(), signal.SIGTERM)
        raise AssertionError("unreachable")

    assert run_with_signal_receipt(spec_path, out_cancel, _run_cancel) == 130
    assert sent_reentrant["count"] == 1
    assert signal.getsignal(signal.SIGTERM) is before_sigterm
    assert signal.getsignal(signal.SIGINT) is before_sigint
    assert (
        TrainReceiptV1.model_validate_json(
            (out_cancel / "result.json").read_text(encoding="utf-8")
        ).status
        == "cancelled"
    )

    out_fail = tmp_path / "out-reentrant-fail"

    def _run_fail(spec: TrainSpecV1, output_dir: Path, sp: Path) -> TrainReceiptV1:
        raise RuntimeError("boom during train")

    assert run_with_signal_receipt(spec_path, out_fail, _run_fail) == 1
    assert sent_reentrant["count"] == 2
    assert signal.getsignal(signal.SIGTERM) is before_sigterm
    assert signal.getsignal(signal.SIGINT) is before_sigint
    assert (
        TrainReceiptV1.model_validate_json(
            (out_fail / "result.json").read_text(encoding="utf-8")
        ).status
        == "failed"
    )

    monkeypatch.setattr(runtime_mod, "atomic_write_json", real_atomic_write_json)

    # 3. Failed signal.signal install does not record previous handler for cleanup.
    real_signal = signal.signal
    calls: list[int] = []

    def _reject_sigint(signum: int, handler: Any) -> Any:
        calls.append(signum)
        if signum == signal.SIGINT:
            raise ValueError("not main thread")
        return real_signal(signum, handler)

    monkeypatch.setattr(signal, "signal", _reject_sigint)
    out_partial = tmp_path / "out-partial-sig"
    assert run_with_signal_receipt(spec_path, out_partial, _run_fresh) == 0
    # SIGINT was attempted once during install and never touched again during finally cleanup.
    assert calls.count(signal.SIGINT) == 1
    assert calls.count(signal.SIGTERM) == 2
    assert signal.getsignal(signal.SIGTERM) is before_sigterm
