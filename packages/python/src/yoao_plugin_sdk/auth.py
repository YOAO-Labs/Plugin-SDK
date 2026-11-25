from __future__ import annotations

import hmac
import os
from collections.abc import Mapping
from pathlib import Path


def read_secret(
    env: Mapping[str, str] | None,
    value_key: str,
    file_key: str,
) -> str:
    source = os.environ if env is None else env
    direct = (source.get(value_key) or "").strip()
    file_path = (source.get(file_key) or "").strip()
    if direct and file_path:
        raise ValueError(f"configure only one of {value_key} or {file_key}")
    if file_path:
        content = Path(file_path).read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(f"{file_key} is empty")
        return content
    if not direct:
        raise ValueError(f"{value_key} or {file_key} is required")
    return direct


def verify_bearer_token(authorization_header: str | None, expected_token: str) -> bool:
    if not expected_token or not isinstance(authorization_header, str):
        return False
    prefix = "Bearer "
    if not authorization_header.startswith(prefix):
        return False
    supplied = authorization_header[len(prefix) :]
    return hmac.compare_digest(supplied.encode("utf-8"), expected_token.encode("utf-8"))
