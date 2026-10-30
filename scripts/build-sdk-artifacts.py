#!/usr/bin/env python3
"""Build SDK release artifacts from an explicit committed SDK revision."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("packages/inference/sdk", "packages/inference/runtime", "packages/inference/mask-utils")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="committed SDK revision to publish")
    parser.add_argument("--output", required=True, type=Path, help="new output directory")
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "--verify", args.revision + "^{commit}"], cwd=ROOT, text=True).strip()
    epoch = int(subprocess.check_output(["git", "show", "-s", "--format=%ct", revision], cwd=ROOT, text=True).strip())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "SOURCE_DATE_EPOCH": str(epoch), "PYTHONHASHSEED": "0"}
    archive = output / "sdk-source.tar"
    # Restrict source to the inference packages and their top-level license/readme.
    subprocess.run(["git", "archive", "--format=tar", "--output", str(archive), revision, "README.md", "LICENSE", *PACKAGES], cwd=ROOT, check=True)
    with tempfile.TemporaryDirectory(prefix="sdk-artifacts-") as temp:
        source = Path(temp) / "source"
        source.mkdir()
        with tarfile.open(archive) as tar:
            tar.extractall(source, filter="data")
        for package in PACKAGES:
            subprocess.run([sys.executable, "-m", "build", "--wheel", "--sdist", "--no-isolation", "--outdir", str(output), str(source / package)], env=env, check=True)
    files = [{"file": path.name, "sha256": sha(path), "size": path.stat().st_size} for path in sorted(output.iterdir()) if path.is_file()]
    manifest = {
        "schema_version": 1,
        "source_repository": "https://github.com/YOAO-Labs/Plugin-SDK",
        "source_revision": revision,
        "source_date_epoch": epoch,
        "source_archive": archive.name,
        "packages": list(PACKAGES),
        "license": "MIT",
        "license_scope": "First-party source only; third-party dependencies and assets retain their own terms.",
        "license_files": ["LICENSE", *(f"{package}/LICENSE" for package in PACKAGES)],
        "builders": {name: importlib.metadata.version(name) for name in ("build", "setuptools", "wheel")},
        "files": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "SHA256SUMS").write_text("".join(f"{sha(path)}  {path.name}\n" for path in sorted(output.iterdir()) if path.is_file() and path.name != "SHA256SUMS"), encoding="ascii")
    print(f"SDK publisher inputs: {output} (source {revision})")


if __name__ == "__main__":
    main()
