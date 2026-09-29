"""Scan emitted browser assets from inside the frontend container safely."""
from __future__ import annotations

import subprocess
import sys

CHECK = r'''const fs=require("node:fs"), path=require("node:path");
const key=fs.readFileSync("/run/secrets/service_api_key","utf8").trim();
const root="/app/.next/static";
const forbidden=[key,"http://api:8000","BACKEND_INTERNAL_URL","BACKEND_API_KEY","/run/secrets/service_api_key"];
function walk(dir){for(const ent of fs.readdirSync(dir,{withFileTypes:true})){const p=path.join(dir,ent.name);if(ent.isDirectory())walk(p);else{const text=fs.readFileSync(p,"utf8");if(forbidden.some(value=>value&&text.includes(value)))throw new Error("browser asset boundary violation")}}}
walk(root); console.log("BROWSER_BOUNDARY PASS: no service key, internal URL, or server configuration in static assets")'''


def main() -> int:
    result = subprocess.run(
        ["docker", "compose", "exec", "-T", "web", "node", "-e", CHECK],
        capture_output=True, text=True,
    )
    if result.returncode:
        print("BROWSER_BOUNDARY FAIL: static asset boundary assertion failed")
        return 1
    print(result.stdout.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
