"""Check protected internal metrics with the runtime-mounted Compose key."""
from __future__ import annotations

import subprocess
import sys

CHECK = r'''import pathlib, urllib.error, urllib.request
url = "http://127.0.0.1:8000/internal/metrics"
def status(headers=None):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=3) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code
assert status() == 401
assert status({"X-API-Key": "invalid-container-check"}) == 401
key = pathlib.Path("/run/secrets/service_api_key").read_text(encoding="utf-8").strip()
assert len(key) >= 32
assert status({"X-API-Key": key}) == 200
print("CONTAINER_METRICS_AUTH PASS: missing=401 invalid=401 valid=200; credential value withheld")'''


def main() -> int:
    result = subprocess.run(
        ["docker", "compose", "exec", "-T", "api", "python", "-c", CHECK],
        capture_output=True, text=True,
    )
    if result.returncode:
        print("CONTAINER_METRICS_AUTH FAIL: auth status assertion failed")
        return 1
    print(result.stdout.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
