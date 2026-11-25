"""Standalone development commands for detection plugins."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import httpx

from yoao_plugin_sdk.inference._scaffold import init_project
from yoao_plugin_sdk.inference.runtime.environment import migrated_value

from .child_environment import isolated_provider_environment


def dev_server(args) -> None:
    """Warm an installed detection provider on loopback without host authority."""
    token = args.token_file.read_text(encoding="utf-8").strip()
    if len(token) < 32:
        raise ValueError("token file must contain at least 32 characters")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.entrypoint):
        raise ValueError("--entrypoint must name an installed console script")
    executable = shutil.which(args.entrypoint)
    if executable is None:
        raise ValueError(f"installed console script {args.entrypoint!r} was not found")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    extra = args.serve_args
    if extra[:1] == ["--"]:
        extra = extra[1:]
    reserved = {"--host", "--port", "--token-file", "--device"}
    if any(value.split("=", 1)[0] in reserved for value in extra):
        raise ValueError("provider arguments cannot override development host, port, token or device")
    command = [executable, "serve", "--device", args.device,
               "--token-file", str(args.token_file.resolve()),
               "--host", "127.0.0.1", "--port", str(args.port), *extra]
    with tempfile.TemporaryDirectory(prefix="yoao-inference-dev-") as state:
        environment = isolated_provider_environment(os.environ)
        environment["YOAO_PLUGIN_DATA_DIR"] = state
        process = subprocess.Popen(command, env=environment)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=120,
                              trust_env=False,
                              headers={"Authorization": "Bearer " + token}) as client:
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise ValueError(f"developer service exited before readiness: {process.returncode}")
                    try:
                        if client.get("/health", timeout=2).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.1)
                else:
                    raise ValueError("developer service did not become ready")
                response = client.post("/warmup", json={"task": "detect"})
                response.raise_for_status()
                health = client.get("/health")
                health.raise_for_status()
                info = health.json()
                if (info.get("compute", {}).get("effective_device") != args.device
                        or info.get("pool", {}).get("current_size", 0) < 1):
                    raise ValueError("preflight did not load the provider on the requested device")
                if (args.device.startswith("cuda:")
                        and info.get("residency", {}).get("gpu_loaded") is not True):
                    raise ValueError("preflight did not establish CUDA residency")
            print("Developer provider ready; this is not platform-managed qualification", flush=True)
            code = process.wait()
            if code:
                raise ValueError(f"developer service exited with status {code}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="yoao-plugin")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("directory", type=Path)
    dev = commands.add_parser("dev")
    dev.add_argument("--entrypoint", required=True)
    dev.add_argument("--device", required=True)
    dev.add_argument("--token-file", type=Path, required=True)
    dev.add_argument("--port", type=int, default=8000)
    dev.add_argument("serve_args", nargs=argparse.REMAINDER,
                     help="Provider-specific serve arguments after --")
    verify = commands.add_parser("check")
    mode = verify.add_mutually_exclusive_group(required=True)
    mode.add_argument("--url", help="Check a running HTTP endpoint only")
    mode.add_argument(
        "--qualify-local",
        action="store_true",
        help="Run full local-exclusive GPU qualification",
    )
    verify.add_argument("--fixtures", required=True, type=Path)
    verify.add_argument(
        "--entrypoint",
        help="Installed author console script for local qualification",
    )
    verify.add_argument("--device")
    verify.add_argument("--gpu-uuid")
    verify.add_argument("--asset-dir", type=Path)
    verify.add_argument("--exclusive-gpu", action="store_true")
    verify.add_argument("--report", type=Path)
    build = commands.add_parser("build")
    build.add_argument("directory", nargs="?", type=Path, default=Path.cwd())
    build.add_argument(
        "--base-image", help="Digest-pinned base image; pip installs offline, APT uses its configured repositories"
    )
    build.add_argument("--wheelhouse", type=Path)
    build.add_argument("--requirements-lock", type=Path)
    build.add_argument(
        "--entrypoint",
        help="Installed author console script (required for image builds)",
    )
    build.add_argument("--tag")
    build.add_argument("--push", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            init_project(args.directory)
        elif args.command == "dev":
            dev_server(args)
        elif args.command == "check":
            if args.qualify_local:
                if not (
                    args.entrypoint
                    and args.device
                    and args.gpu_uuid
                    and args.asset_dir
                    and args.report
                    and args.exclusive_gpu
                ):
                    raise ValueError(
                        "--qualify-local requires --entrypoint, --device, --gpu-uuid, --asset-dir, --exclusive-gpu and --report"
                    )
                from .qualification import qualify_local

                report = qualify_local(
                    args.fixtures,
                    args.device,
                    args.gpu_uuid,
                    args.asset_dir,
                    args.report,
                    entrypoint=args.entrypoint,
                )
                print(
                    json.dumps(
                        {
                            "mode": report["mode"],
                            "report": str(args.report),
                            "platform_managed_qualified": False,
                        }
                    )
                )
            else:
                if args.entrypoint or args.device or args.gpu_uuid or args.asset_dir or args.exclusive_gpu:
                    raise ValueError(
                        "GPU qualification options require --qualify-local"
                    )
                from .check import check_http

                plugin_token = (
                    migrated_value(os.environ, "YOAO_PLUGIN_TOKEN", "AAP_PLUGIN_TOKEN")
                    or ""
                )
                report = check_http(args.url, plugin_token, args.fixtures)
                if args.report:
                    args.report.write_text(
                        json.dumps(report, indent=2) + "\n", encoding="utf-8"
                    )
                print(json.dumps(report))
        else:
            image_options = (
                args.base_image,
                args.wheelhouse,
                args.requirements_lock,
                args.tag,
            )
            if any(image_options) or args.entrypoint or args.push:
                if not all(image_options) or not args.entrypoint:
                    raise ValueError(
                        "image build requires --entrypoint, --base-image, --wheelhouse, --requirements-lock and --tag"
                    )
                from .image_build import build_image

                build_image(
                    base_image=args.base_image,
                    wheelhouse=args.wheelhouse,
                    requirements_lock=args.requirements_lock,
                    entrypoint=args.entrypoint,
                    tag=args.tag,
                    push=args.push,
                )
            else:
                subprocess.run(
                    [sys.executable, "-m", "build", "--wheel", "--no-isolation", str(args.directory)],
                    check=True,
                )
                for artifact in sorted((args.directory / "dist").glob("*.whl")):
                    print(
                        f"{hashlib.sha256(artifact.read_bytes()).hexdigest()}  {artifact}"
                    )
    except (OSError, RuntimeError, ValueError, httpx.HTTPError, subprocess.SubprocessError) as exc:
        print(f"yoao-plugin: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
