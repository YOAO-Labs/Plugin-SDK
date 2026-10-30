#!/usr/bin/env python3
"""Download the five platform-owned portable contract wheels with pinned hashes."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

BASE = "https://github.com/YOAO-Labs/Plugin-SDK/releases/download/contracts-3d6393e64630/"
MANIFEST_SHA256 = "d3749d2a7429da60c02e222c65e9c4a5ec545b050b797422263d5a3bcb02ddec"
REVISION = "3d6393e64630fdeb6ff699b13c2d1f9e198a1a06"
CONTRACTS = {"aap-inference-contract", "aap-lifecycle-contract", "aap-protocol-v2", "aap-task-contract", "aap-training-contract"}


def get(name, expected):
    if Path(name).name != name:
        raise ValueError("invalid contract artifact filename")
    with urlopen(BASE + name, timeout=120) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"contract artifact hash mismatch: {name}")
    return data


def fetch(output):
    manifest = json.loads(get("manifest.json", MANIFEST_SHA256))
    if manifest["source_revision"] != REVISION:
        raise ValueError("contract source revision mismatch")
    wheels = [item for item in manifest["files"] if item["file"].endswith(".whl")]
    if len(wheels) != 5 or {item["name"] for item in wheels} != CONTRACTS:
        raise ValueError("unexpected contract artifact inventory")
    output.mkdir(parents=True, exist_ok=True)
    for item in wheels:
        (output / item["file"]).write_bytes(get(item["file"], item["sha256"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    fetch(parser.parse_args().output)
