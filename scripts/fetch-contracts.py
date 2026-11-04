#!/usr/bin/env python3
"""Copy portable contract wheels from an explicitly pinned manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen

CONTRACTS = {
    "yoao-inference-contract", "yoao-lifecycle-contract", "yoao-protocol-v2",
    "yoao-task-contract", "yoao-training-contract",
}


def read_location(location):
    parsed = urlparse(str(location))
    if parsed.scheme in ("https", "http", "file"):
        with urlopen(str(location), timeout=120) as response:
            return response.read()
    if parsed.scheme:
        raise ValueError(f"unsupported artifact location: {location}")
    return Path(location).read_bytes()


def verified(data, expected, label):
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected) or hashlib.sha256(data).hexdigest() != expected.lower():
        raise ValueError(f"artifact hash mismatch: {label}")
    return data


def fetch(manifest_location, manifest_sha256, output, packages=None):
    manifest = json.loads(verified(read_location(manifest_location), manifest_sha256, "manifest"))
    selected = set(packages) if packages else CONTRACTS
    if not selected or not selected <= CONTRACTS:
        raise ValueError("unexpected contract package filter")
    inventory = [item for item in manifest["files"] if item["file"].endswith(".whl")]
    if not packages and (len(inventory) != len(CONTRACTS) or {item["name"] for item in inventory} != CONTRACTS):
        raise ValueError("unexpected contract artifact inventory")
    wheels = [item for item in inventory if item["name"] in selected]
    if len(wheels) != len(selected) or {item["name"] for item in wheels} != selected:
        raise ValueError("unexpected contract artifact inventory")
    parsed = urlparse(str(manifest_location))
    output.mkdir(parents=True, exist_ok=True)
    for item in wheels:
        name = item["file"]
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("invalid contract artifact filename")
        location = urljoin(str(manifest_location), name) if parsed.scheme else Path(manifest_location).parent / name
        (output / name).write_bytes(verified(read_location(location), item["sha256"], name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="manifest path or URL; wheel files must be alongside it")
    parser.add_argument("--sha256", required=True, help="expected SHA-256 of manifest bytes")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--package", action="append", choices=sorted(CONTRACTS), help="contract distribution to fetch; repeat to select multiple")
    args = parser.parse_args()
    fetch(args.manifest, args.sha256, args.output, args.package)
