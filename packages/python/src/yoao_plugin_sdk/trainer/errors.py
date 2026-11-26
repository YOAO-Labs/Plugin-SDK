"""Typed trainer domain exceptions and failure classification without ML imports."""

from __future__ import annotations

import sys


class TrainerError(ValueError):
    failure_code: str = "native_exception"

    def __init__(self, message: str, *, failure_code: str | None = None) -> None:
        super().__init__(message)
        if failure_code is not None:
            self.failure_code = failure_code


class InputMismatchError(TrainerError):
    failure_code = "input_mismatch"


class NativeConfigurationMismatch(TrainerError):
    failure_code = "config_mismatch"


class NativeRuntimeMismatch(NativeConfigurationMismatch):
    failure_code = "runtime_mismatch"


class DatasetContractError(InputMismatchError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message, failure_code="input_mismatch")
        self.code = code


def _clean_text(text: str, *, max_length: int = 2048) -> str:
    cleaned = "".join(" " if ord(ch) < 0x20 or ord(ch) == 0x7F else ch for ch in text)
    collapsed = " ".join(cleaned.split())
    if len(collapsed) > max_length:
        collapsed = collapsed[:max_length].rstrip()
    return collapsed


def classify_failure(exc: BaseException) -> str:
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None:
        cuda_mod = getattr(torch_mod, "cuda", None)
        oom_cls = getattr(cuda_mod, "OutOfMemoryError", None)
        if isinstance(oom_cls, type) and isinstance(exc, oom_cls):
            return "cuda_oom"
    code = getattr(exc, "failure_code", None)
    if isinstance(code, str):
        cleaned = _clean_text(code, max_length=64)
        if cleaned:
            return cleaned
    return "native_exception"


def sanitize_failure_message(
    exc: BaseException,
    fallback: str | None = None,
) -> str:
    cleaned_raw = _clean_text(str(exc))
    if cleaned_raw:
        return cleaned_raw
    if fallback is not None:
        cleaned_fallback = _clean_text(fallback)
        if cleaned_fallback:
            return cleaned_fallback
    return _clean_text(type(exc).__name__) or "error"


__all__ = [
    "DatasetContractError",
    "InputMismatchError",
    "NativeConfigurationMismatch",
    "NativeRuntimeMismatch",
    "TrainerError",
    "classify_failure",
    "sanitize_failure_message",
]
