# Rollback runbook

Rollback is a coordinated return of **both** application images and the release
configuration to a known compatible version. Never roll only the web or API
image, and never replace model files or thresholds in a running image. Model
artifacts, preprocessing, thresholds, feature contracts, and the model release
manifest form one compatibility unit.

This educational/research candidate is not a diagnosis or screening service and
is not clinically validated. Parkinson's remains Experimental and scores are
uncalibrated. No deployed availability SLO is measured; a local load result is
not availability evidence.

## Indicators

Initiate rollback review if readiness is degraded, a model/application hash
mismatches, model loading fails, protected API auth unexpectedly rejects the
BFF, 5xx errors remain elevated, the service fails to recover after bounded
load, or read-only/security boundaries are lost. Investigate through request IDs
and aggregate logs only; do not collect measurements, scores, reports, or user
identities (none exist).

## Procedure

1. Stop routing new requests at the operator's local ingress.
2. Select the previously reviewed API image, web image, `.env` configuration,
   and matching mounted service-key file from a protected backup. Confirm both
   OCI revisions/versions and release-manifest identifiers before startup.
3. Set `API_IMAGE`, `WEB_IMAGE`, `APP_VERSION`, and `APP_SOURCE_REVISION` to the
   same recorded release unit in the protected Compose environment; do not put
   secrets in image arguments or source control.
4. Start both images together and verify health/readiness and golden vectors:

```powershell
docker compose --env-file .env up --detach --wait
Invoke-RestMethod http://127.0.0.1:3000/api/health
Invoke-RestMethod http://127.0.0.1:3000/api/ready
node scripts/container_smoke.mjs
```

5. If readiness, hash checks, or golden predictions fail, stop the stack and
   preserve only redacted operational diagnostics. Do not edit models in place.
6. Record the selected compatible image IDs and config revision in the local
   incident record, without credentials, payloads, scores, or PDF contents.

When registry images are unavailable, recovery depends on previously exported,
verified local images and protected config backups. Do not pull an unverified
tag or rebuild from mutable dependencies during an incident.
