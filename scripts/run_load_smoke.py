"""Start an isolated authenticated API+BFF pair, run CI load smoke, and clean up."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

from scripts.load_test import run_profile

ROOT = Path(__file__).resolve().parents[1]


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait(url: str, process: subprocess.Popen, *, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("a local service exited before becoming healthy")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.25)
    raise RuntimeError("a local service did not become healthy before the startup timeout")


def _verify_recovery(mode: str, api_url: str, web_url: str) -> None:
    path = "/api/v1/ready" if mode == "direct" else "/api/ready"
    base = api_url if mode == "direct" else web_url
    request = urllib.request.Request(f"{base}{path}")
    with urllib.request.urlopen(request, timeout=5) as response:
        payload = json.loads(response.read())
        if response.status != 200 or payload.get("status") != "ready":
            raise RuntimeError("readiness did not recover after load")


def _descendants(pid: int) -> set[int]:
    try:
        import psutil

        return {process.pid for process in psutil.Process(pid).children(recursive=True)} | {pid}
    except (ImportError, OSError):
        return {pid}


def _stop(process: subprocess.Popen | None) -> bool:
    if process is None:
        return True
    pids = _descendants(process.pid)
    if process.poll() is None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                return False
    try:
        import psutil

        tracked = []
        for pid in pids:
            try:
                tracked.append(psutil.Process(pid))
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(tracked, timeout=5)
        if alive:
            return False
    except ImportError:
        pass
    return process.poll() is not None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--requests", type=int, default=50)
    profile.add_argument("--duration-seconds", type=int)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--target", choices=("direct", "bff"), default="bff")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        raise SystemExit("Operational smoke requires the locked Python 3.11 environment.")
    node = shutil.which("node")
    if not node:
        raise SystemExit("Node was not found on PATH.")
    node_version = subprocess.run([node, "--version"], capture_output=True, text=True, check=True).stdout.strip()
    if node_version != "v20.19.0":
        raise SystemExit(f"Operational smoke requires Node v20.19.0; received {node_version}.")

    api_key = secrets.token_urlsafe(48)
    api_port, web_port = _port(), _port()
    api_url = f"http://127.0.0.1:{api_port}"
    web_url = f"http://127.0.0.1:{web_port}"
    processes: list[subprocess.Popen] = []
    process_pids: list[int] = []
    temp_logs = tempfile.TemporaryDirectory(prefix="medical-ai-load-smoke-")
    api_log_path = Path(temp_logs.name) / "api.log"
    web_log_path = Path(temp_logs.name) / "web.log"
    log_handles = []
    cleanup_ok = True
    profile_error: str | None = None
    report: dict | None = None
    try:
        api_log = api_log_path.open("w+b")
        web_log = web_log_path.open("w+b")
        log_handles.extend((api_log, web_log))
        api_env = {
            **os.environ, "APP_ENV": "production", "API_KEY": api_key,
            "API_HOST": "127.0.0.1", "API_PORT": str(api_port),
            "ALLOWED_ORIGINS": web_url, "RATE_LIMIT_PER_MINUTE": "10000",
            "LOG_LEVEL": "WARNING", "NEXT_TELEMETRY_DISABLED": "1",
        }
        api = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1",
             "--port", str(api_port), "--log-level", "warning"],
            cwd=ROOT, env=api_env, stdout=api_log, stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        processes.append(api)
        _wait(f"{api_url}/api/v1/health", api)

        web_env = {
            **os.environ, "NODE_ENV": "production", "NEXT_TELEMETRY_DISABLED": "1",
            "BACKEND_INTERNAL_URL": api_url, "BACKEND_API_KEY": api_key,
            "BACKEND_TIMEOUT_MS": "15000", "BFF_MAX_REQUEST_BYTES": "16384",
        }
        next_cli = ROOT / "frontend/node_modules/next/dist/bin/next"
        web = subprocess.Popen(
            [node, str(next_cli), "start", "-H", "127.0.0.1", "-p", str(web_port)],
            cwd=ROOT / "frontend", env=web_env, stdout=web_log, stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        processes.append(web)
        _wait(f"{web_url}/api/health", web)
        process_pids = [api.pid, web.pid]
        report = asyncio.run(run_profile(
            mode=args.target, base_url=api_url if args.target == "direct" else web_url,
            concurrency=args.concurrency,
            request_count=args.requests if args.duration_seconds is None else None,
            duration_seconds=args.duration_seconds, api_key=api_key,
            metrics_url=f"{api_url}/internal/metrics", process_pids=process_pids,
        ))
        time.sleep(0.25)
        _verify_recovery(args.target, api_url, web_url)
        report["recovery"] = {"readiness_after_load": "ready", "confirmed": True}
        if report["results"]["errors"]:
            profile_error = "load smoke reported request or response-contract errors"
    except Exception as exc:
        # Exception text is deliberately omitted: subprocess/network errors can
        # carry URLs, arguments, credentials, or request data.
        profile_error = type(exc).__name__
    finally:
        for process in reversed(processes):
            cleanup_ok = _stop(process) and cleanup_ok
        for handle in log_handles:
            handle.flush()
            handle.close()

        # Inspect captured service logs only after processes stop; never print them.
        api_text = api_log_path.read_text(encoding="utf-8", errors="replace") if api_log_path.exists() else ""
        web_text = web_log_path.read_text(encoding="utf-8", errors="replace") if web_log_path.exists() else ""
        private_markers = [api_key, api_url, "%PDF", "Input measurements:"]
        if any(marker and (marker in api_text or marker in web_text) for marker in private_markers):
            profile_error = "privacy marker found in captured service logs"
        temp_logs.cleanup()

    if not cleanup_ok:
        profile_error = "temporary service process did not terminate"
    if report is not None:
        report["environment"] = {
            "node": node_version,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": platform.platform(),
            "target": f"isolated local production {args.target.upper()} (API+BFF services started)",
        }
    if report is not None and args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report is not None and args.markdown_output:
        from scripts.load_test import _markdown

        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(_markdown(report), encoding="utf-8")
    if report is None:
        print(f"LOAD_SMOKE FAIL category={profile_error or 'startup'} cleanup_ok={cleanup_ok}")
        return 1
    print(
        "LOAD_SMOKE " + ("PASS" if profile_error is None and cleanup_ok else "FAIL")
        + f" node={node_version} python={sys.version_info.major}.{sys.version_info.minor}"
        + f" requests={report['results']['successful_requests']} errors={report['results']['errors']}"
        + f" throughput_rps={report['results']['throughput_requests_per_second']}"
        + f" peak_in_flight={report['results']['peak_in_flight']}"
        + f" metrics_verified={report['metrics_verification']['all_diseases_observed']}"
        + f" readiness_recovered={report['recovery']['confirmed']}"
        + f" cleanup_ok={cleanup_ok}"
    )
    return 0 if profile_error is None and cleanup_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
