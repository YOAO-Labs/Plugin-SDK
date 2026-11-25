"""Contract fetcher integrity and inventory behavior."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("fetch_contracts", Path(__file__).with_name("fetch-contracts.py"))
fetch_contracts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_contracts)


def manifest_with_wheel(tmp_path, name="yoao-sdk", filename="contract.whl"):
    wheel = b"verified contract artifact"
    if Path(filename).name == filename:
        (tmp_path / filename).write_bytes(wheel)
    manifest = {"files": [{"name": name, "file": filename, "sha256": hashlib.sha256(wheel).hexdigest()}]}
    data = json.dumps(manifest).encode()
    path = tmp_path / "manifest.json"
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest(), wheel


def test_fetch_filtered_contract_from_local_manifest(tmp_path):
    manifest, digest, wheel = manifest_with_wheel(tmp_path)
    output = tmp_path / "out"
    fetch_contracts.fetch(manifest.as_uri(), digest, output, ["yoao-sdk"])
    assert (output / "contract.whl").read_bytes() == wheel
    unfiltered_output = tmp_path / "out-all"
    fetch_contracts.fetch(manifest.as_uri(), digest, unfiltered_output)
    assert (unfiltered_output / "contract.whl").read_bytes() == wheel
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    bad_manifest, bad_digest, _ = manifest_with_wheel(bad_dir, name="unexpected-pkg")
    with pytest.raises(ValueError, match="inventory"):
        fetch_contracts.fetch(bad_manifest, bad_digest, output)


def test_fetch_rejects_tampering_and_unsafe_filename(tmp_path):
    manifest, digest, _ = manifest_with_wheel(tmp_path)
    with pytest.raises(ValueError, match="manifest"):
        fetch_contracts.fetch(manifest, "0" * 64, tmp_path / "out", ["yoao-sdk"])
    (tmp_path / "contract.whl").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="contract.whl"):
        fetch_contracts.fetch(manifest, digest, tmp_path / "out", ["yoao-sdk"])
    manifest, digest, _ = manifest_with_wheel(tmp_path, filename="../escape.whl")
    with pytest.raises(ValueError, match="filename"):
        fetch_contracts.fetch(manifest, digest, tmp_path / "out", ["yoao-sdk"])
