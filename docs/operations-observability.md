# Operations: privacy-safe observability and load evidence

## Signals and access

FastAPI emits one JSON request record in production with a UUID request ID, bounded route, method, status, duration, known disease (when applicable), and—after inference—model identifier and release status. It never logs measurement bodies, auth headers, response bodies, raw exception messages, client IPs, or query values. The Next.js BFF emits the corresponding request ID and bounded route; it does not log backend URLs or credentials. Keep application logs access-controlled, set retention to the minimum needed for incident response, and do not attach request bodies or generated reports to log platforms.

`GET /internal/metrics` is not in OpenAPI, is protected by the configured API key, returns `Cache-Control: no-store`, and should additionally be restricted to the private service network / metrics collector. It exports bounded-cardinality request/error/duration, prediction/report, threshold, model-load, readiness, auth/rate-limit rejection, and in-flight metrics. Do not expose this endpoint publicly. Label values must remain from the fixed method, route, disease, outcome, result, and status-class domains in `src/api/observability.py`.

Metric families: `medical_ai_http_requests_total{method,route,status_class}`, `medical_ai_http_errors_total{method,route,status_class,error_category}`, `medical_ai_http_request_duration_seconds{method,route}`, `medical_ai_predictions_total{disease,outcome}`, `medical_ai_prediction_duration_seconds{disease}`, `medical_ai_reports_total{disease,outcome}`, `medical_ai_report_duration_seconds{disease}`, `medical_ai_threshold_results_total{disease,result}`, `medical_ai_model_loads_total{disease,result}`, `medical_ai_readiness{state}`, `medical_ai_auth_rejections_total`, `medical_ai_rate_limit_rejections_total`, and `medical_ai_in_flight_requests`.

## Provisional engineering objectives

These are initial operational targets, not measured guarantees, medical validation, or release claims. Establish production SLOs only after representative deployment/load evidence:

| Signal | Provisional target | Response |
| --- | --- | --- |
| Availability | 99.5% monthly, once deployment monitoring exists | Page/triage sustained unavailability; use verified rollback if readiness or request failures indicate release regression |
| Prediction latency | p95 under 2 seconds after warm-up | Inspect API latency, saturation, model loading and dependency health |
| PDF report latency | p95 under 5 seconds after warm-up | Inspect report generation and memory/CPU pressure |
| Unexpected request failures | under 1% over a 5-minute window | Triage by bounded error category and request ID; never request user measurements in logs |
| Readiness | continuously ready during normal operation | Inspect artifact integrity and restart/rollback using deployment controls |
| In-flight requests | no sustained growth after traffic subsides | Check stalled downstream calls and resource exhaustion |

Treat authentication and rate-limit rejections separately from service failures. These initial targets are intentionally provisional; review them against deployment capacity and privacy requirements before operational adoption.

Operational response: a model-load failure increments its bounded failure counter and causes readiness/integrity checks to fail; keep the instance out of service and restore the last hash-verified release rather than replacing a model in place. A release/hash mismatch is a stop/rollback indicator; verify the manifest and persisted-model evaluator before re-entry. A spike in authentication failures calls for service-key review/rotation and network-boundary checks; this service credential authenticates the BFF/service, not an end user. There is currently no user identity or per-user authorization. Rate-limit counters should be reviewed for abuse or accidental bursts; the existing in-memory limiter is a single-process brake, not a distributed quota. For privacy incidents, correlate using request ID and bounded metadata, restrict log access, preserve only necessary evidence, and never collect submitted measurements or report contents for triage. Roll back when readiness is degraded, release hashes disagree, errors stay above the provisional threshold, or latency/resource saturation does not recover after traffic subsides. Operational health and these service objectives do not establish clinical validity or safety.

## Local load profiles

Use Node `20.19.0`, Python `3.11`, the locked Python dependencies, and an installed frontend dependency tree. `scripts/run_load_smoke.py` starts a fresh authenticated production-mode FastAPI process and the already-built Next.js production server, runs a deterministic fixture workload through the BFF, verifies protected metrics and model-release golden responses, inspects temporary logs for private markers, and terminates both process trees. It elevates the in-memory per-IP rate limit for the load window; independent tests cover rate-limit behavior. It does not save PDF contents or patient values.

```powershell
$env:PATH = "$Node20Directory;$env:PATH"
$env:E2E_PYTHON = "$LockedPython311"
& $env:E2E_PYTHON -m scripts.run_load_smoke --requests 50 --concurrency 5
& $env:E2E_PYTHON -m scripts.run_load_smoke --target direct --requests 50 --concurrency 5
& $env:E2E_PYTHON -m scripts.run_load_smoke --requests 500 --concurrency 10 `
  --json-output reports/operations/bff-baseline-500.json `
  --markdown-output reports/operations/bff-baseline-500.md
& $env:E2E_PYTHON -m scripts.run_load_smoke --duration-seconds 60 --concurrency 25 `
  --json-output reports/operations/bff-stress-60s.json `
  --markdown-output reports/operations/bff-stress-60s.md
```

The load runner warms all five persisted pipelines and cycles a fixed workload mix: 70% predictions, 10% PDF reports, 5% catalog, 5% disease metadata, 5% health and 5% readiness. Reports contain only aggregate latency percentiles, throughput, error categories, in-flight and process memory observations. Results describe only the recorded machine/configuration and must not be interpreted as cross-environment capacity claims.
