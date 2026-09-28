# Deployment runbook (local release-candidate topology)

This runbook describes a reproducible local Compose topology, not a cloud
deployment. The candidate is educational/research only; it is not a diagnosis or
screening service, is not clinically validated, and must not be used to direct
care. Parkinson's remains Experimental; all scores are uncalibrated outputs.
No deployed availability SLO has been measured, and local load tests are not
availability evidence.

## Prerequisites and secret setup

Use Docker Engine with Compose v2 and Buildx, Python 3.11.15, Node 20.19.0,
and a supported Linux/Windows environment. Python and Node base image multiarch
index digests are pinned in both Dockerfiles; the Debian/Alpine package feeds
used during the backend build remain time-varying inputs. Do not commit `.env`,
`.secrets/`, or any credential.

PowerShell example; it writes a fresh local service key directly to an ignored
file and does not display the generated value:

```powershell
Copy-Item .env.example .env
New-Item -ItemType Directory -Force .secrets | Out-Null
python -c "import secrets; print(secrets.token_urlsafe(48), end='')" | Set-Content -NoNewline .secrets/service_api_key
docker compose --env-file .env config --quiet
docker compose --env-file .env up --build --detach --wait
```

The key is service-to-service authentication between the Next.js BFF and
FastAPI, not user authentication. The API has no host-published port. Only the
frontend binds to `127.0.0.1:3000` by default; metrics are protected and internal.
FastAPI and the frontend run as UID/GID 10001 with dropped capabilities,
`no-new-privileges`, read-only roots and narrowly scoped temporary filesystems.

File-backed Compose secrets are read-only mounts, not a production secret
manager. The host must make the file readable to container UID/GID 10001 without
making it broadly readable to other host users; verify with the included
metrics-auth check. For a deployed system, replace this local-file mechanism
with an approved runtime secret provider.

## Verify and operate

```powershell
Invoke-RestMethod http://127.0.0.1:3000/api/health
Invoke-RestMethod http://127.0.0.1:3000/api/ready
node scripts/container_smoke.mjs
python scripts/check_container_runtime.py
python scripts/check_container_metrics_auth.py
docker compose ps
docker compose down
```

The metrics-auth check tests missing, invalid, and mounted valid credentials
without printing the key. The metrics endpoint is `GET /internal/metrics` on the
private API network and is intentionally not routed through the browser BFF.
The smoke tool validates all five golden prediction results and five in-memory
PDF reports; it does not persist reports. Do not expose the API port or metrics
route through a reverse proxy.

Keep logs access-controlled and short-lived. Request IDs can correlate bounded
operational metadata; logs must not contain measurements, scores, PDF contents,
credentials, authorization headers, cookies, internal URLs, or exception
traces. The API key authenticates the service, not an individual.

## Release compatibility

Treat the app image, backend image, Python/Node runtime, frontend contracts,
model files, preprocessing, thresholds, and model/application manifests as one
compatible release unit. A hash mismatch or degraded readiness means stop
traffic and follow the rollback runbook. Operational readiness is not clinical
validation or an availability guarantee.
