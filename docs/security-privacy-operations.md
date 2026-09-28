# Security and privacy operating notes

## Intended use and limits

Educational/research use only. This application is not a diagnosis or screening
service and is not clinically validated. Parkinson's remains Experimental.
Scores are uncalibrated estimator outputs, not probabilities or medical advice.
Operational health, CI, image scans, and local load tests do not establish
clinical validity, real-world safety, or deployed availability. No deployed
availability SLO has been measured.

## Boundaries

Only the Next.js frontend is published by the example Compose topology. The API
is on an internal network and `/internal/metrics` requires the service
credential. Runtime secrets are mounted as files and are not Docker build args,
frontend public variables, or image layers. Both images use a dedicated
non-root UID, read-only root filesystems, tmpfs-only scratch paths, dropped
capabilities, `no-new-privileges`, resource bounds, and rotated logs.

The shared service key authenticates BFF-to-API service traffic only; there is
no end-user identity, account, per-user authorization, or persistent patient
store. Rate limiting is process-local, not distributed. Distributed rate
limiting and centralized monitoring are future work. Do not expose API or
metrics ports publicly.

## Safe operations

Use bounded request IDs, normalized routes, status classes, and controlled error
categories for investigation. Never log or upload measurements, request or
response bodies, model scores, API keys, authorization headers, cookies, IPs,
full filesystem paths, internal URLs, stack traces, or PDF bytes/report text.
Review access and retention of operational logs. On a privacy incident, restrict
access and preserve only necessary redacted metadata.

Treat model load failures, readiness degradation, release hash mismatches,
authentication anomalies, sustained rate-limit rejection, and unrecovered
resource saturation as operational signals. Restore both application images
and compatible release configuration together. Models, preprocessing,
thresholds, contracts, and release manifest are inseparable. Consult the
[rollback runbook](rollback-runbook.md).

## Source and image supply-chain gates

The release-candidate workflow uses Gitleaks v8.30.1, pinned by an immutable
container digest, for two separate checks: a Git-aware snapshot of tracked and
unignored untracked candidate source files, and all reachable Git history. The
source snapshot omits only generated/vendor/cache, raw dataset, and binary
artifact locations; documentation, scripts, CI, release/configuration files,
lockfiles, and `.env.example` remain in scope. Common lockfile basenames that
Gitleaks' own default path allowlist skips are copied under a scanner-only
suffix, preserving their contents while making them visible to the detector.
No finding allowlist is added. Both reports use Gitleaks redaction.

Trivy is reserved for the two final container images: it checks embedded
secrets and fails on Critical or High vulnerabilities with a published fix
(`ignore-unfixed`), matching the existing image-scan policy. Trivy also
generates CycloneDX SBOMs for both images. Trivy JSON reports are scrubbed of
secret match text before artifact
upload; SBOMs and redacted reports are CI artifacts, not published images.
Hosted CI remains unobserved until the workflow is run on the pushed branch.
