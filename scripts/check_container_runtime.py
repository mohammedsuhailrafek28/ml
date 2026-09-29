"""Assert Compose container hardening without printing runtime secrets."""
from __future__ import annotations

import json
import subprocess
import sys


def output(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def inspect(service: str) -> dict:
    container_id = output("docker", "compose", "ps", "-q", service)
    if not container_id:
        raise RuntimeError(f"{service} container is not running")
    return json.loads(output("docker", "inspect", container_id))[0]


def main() -> int:
    try:
        api, web = inspect("api"), inspect("web")
        for name, item in (("api", api), ("web", web)):
            config, host = item["Config"], item["HostConfig"]
            assert config["User"] == "10001:10001", f"{name} is not the dedicated non-root user"
            assert host["ReadonlyRootfs"] is True, f"{name} root filesystem is writable"
            assert "ALL" in host["CapDrop"], f"{name} has Linux capabilities"
            assert any("no-new-privileges:true" in option for option in host["SecurityOpt"]), f"{name} lacks no-new-privileges"
        assert not any(api["NetworkSettings"]["Ports"].get(port) for port in ("8000/tcp",)), "API port is published to the host"
        api_networks = api["NetworkSettings"]["Networks"]
        assert len(api_networks) == 1 and any(name.endswith("_backend") for name in api_networks), "API is not isolated to backend network"
        web_ports = web["NetworkSettings"]["Ports"].get("3000/tcp") or []
        assert web_ports and all(item["HostIp"] in {"127.0.0.1", "::1"} for item in web_ports), "frontend is not bound only to loopback"
        networks = json.loads(output("docker", "network", "inspect", *api_networks.keys()))
        assert all(item["Internal"] for item in networks), "API network is not marked internal"
        print("CONTAINER_RUNTIME PASS: non-root, read-only roots, no capabilities, no-new-privileges, API private, frontend loopback-only")
        return 0
    except (AssertionError, KeyError, RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError):
        print("CONTAINER_RUNTIME FAIL: hardening assertion failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
