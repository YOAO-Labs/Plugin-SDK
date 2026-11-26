"""Atomic, fsynced file writes, SHA-256 helpers, and immutable artifact copies."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from typing import TypedDict

from yoao_plugin_sdk.trainer.protocol import atomic_write_json


class ImmutableArtifactInfo(TypedDict):
    name: str
    sha256: str
    size_bytes: int


def digest_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sync_directory(path: str | Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_bytes(
    path: str | Path,
    data: bytes,
    *,
    mode: int = 0o644,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=str(target.parent))
    fd_open = True
    try:
        stream = os.fdopen(fd, "wb")
        fd_open = False
        with stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, target)
        sync_directory(target.parent)
    except BaseException:
        if fd_open:
            try:
                os.close(fd)
            except OSError:
                pass
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def immutable_copy(
    source: str | Path,
    destination: str | Path,
    *,
    mode: int = 0o644,
) -> ImmutableArtifactInfo:
    src = Path(source)
    dst = Path(destination)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        raise ValueError("Immutable artifact destination must not be a symlink")
    if dst.exists():
        if not dst.is_file():
            raise ValueError("Immutable artifact destination must be a regular file")
        dst_digest = digest_file(dst)
        if digest_file(src) != dst_digest:
            raise ValueError("Immutable artifact already exists with different content")
        return {
            "name": dst.name,
            "sha256": dst_digest,
            "size_bytes": dst.stat().st_size,
        }
    hasher = hashlib.sha256()
    size_bytes = 0
    with src.open("rb") as original:
        fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=str(dst.parent))
        fd_open = True
        try:
            copy_stream = os.fdopen(fd, "wb")
            fd_open = False
            with copy_stream:
                for chunk in iter(lambda: original.read(1024 * 1024), b""):
                    hasher.update(chunk)
                    copy_stream.write(chunk)
                    size_bytes += len(chunk)
                copy_stream.flush()
                os.fsync(copy_stream.fileno())
            os.chmod(tmp_name, mode)
            os.replace(tmp_name, dst)
            sync_directory(dst.parent)
        except BaseException:
            if fd_open:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise
    return {
        "name": dst.name,
        "sha256": hasher.hexdigest(),
        "size_bytes": size_bytes,
    }


__all__ = [
    "ImmutableArtifactInfo",
    "atomic_write_bytes",
    "atomic_write_json",
    "digest_file",
    "immutable_copy",
    "sync_directory",
]
