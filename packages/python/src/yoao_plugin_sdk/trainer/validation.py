"""Strict no-coercion schema validation, metric sanitization, and source hashing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import importlib.metadata
import math
from pathlib import Path, PurePath
from typing import Any

from yoao_plugin_sdk.trainer.protocol import canonical_json_digest


def normalized(value: Any) -> Any:
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, tuple):
        return [normalized(item) for item in value]
    if isinstance(value, list):
        return [normalized(item) for item in value]
    if isinstance(value, dict):
        return {str(key): normalized(item) for key, item in value.items()}
    return value


def validate_value(value: Any, rule: Mapping[str, Any]) -> None:
    if "anyOf" in rule:
        for option in rule["anyOf"]:
            try:
                validate_value(value, option)
                return
            except ValueError:
                pass
        raise ValueError("Value does not match any allowed type or range")
    if "enum" in rule:
        if not any(
            type(value) is type(item) and value == item for item in rule["enum"]
        ):
            raise ValueError(f"Must be one of {rule['enum']}")
        return
    kind = rule["type"]
    types = {
        "null": type(None),
        "boolean": bool,
        "integer": int,
        "string": str,
        "array": list,
    }
    if kind == "number":
        valid = type(value) is int or (type(value) is float and math.isfinite(value))
    else:
        valid = type(value) is types[kind]
    if not valid:
        raise ValueError(f"Must be {kind} without implicit type conversion")
    if kind in {"number", "integer"}:
        if type(value) is float and not math.isfinite(value):
            raise ValueError("Must be a finite number")
        for key, predicate in (
            ("minimum", lambda bound: value >= bound),
            ("maximum", lambda bound: value <= bound),
            ("exclusiveMinimum", lambda bound: value > bound),
        ):
            if key in rule and not predicate(rule[key]):
                raise ValueError(f"Violates {key}={rule[key]}")
        if "multipleOf" in rule and value % rule["multipleOf"] != 0:
            raise ValueError(f"Must be a multiple of {rule['multipleOf']}")
    if kind == "array":
        if not rule.get("minItems", 0) <= len(value) <= rule.get("maxItems", 100):
            raise ValueError("List length is out of allowed bounds")
        for item in value:
            validate_value(item, rule["items"])
        if rule.get("uniqueItems") and len(set(value)) != len(value):
            raise ValueError("List items must be unique")
        if rule.get("ordered") and value != sorted(value):
            raise ValueError("Lower bound must not exceed upper bound")


def metrics_of(raw: Mapping[str, Any] | None) -> dict[str, float]:
    if not raw:
        return {}
    result: dict[str, float] = {}
    for key, value in raw.items():
        if (
            not isinstance(value, bool)
            and isinstance(value, (float, int))
            and math.isfinite(float(value))
        ):
            as_float = float(value)
            result[str(key)] = 0.0 if as_float == 0.0 else as_float
    return result


def hash_source_files(paths: Sequence[str | Path]) -> str:
    return canonical_json_digest(
        [hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths]
    )


def optional_package_version(
    package: str,
    default: str | None = None,
) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return default


__all__ = [
    "hash_source_files",
    "metrics_of",
    "normalized",
    "optional_package_version",
    "validate_value",
]
