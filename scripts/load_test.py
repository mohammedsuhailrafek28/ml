"""Privacy-safe deterministic HTTP load profiles for the private API or BFF.

This reports engineering measurements for one local environment; it is not a
capacity promise, a clinical validation, or a service guarantee.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
import uuid

import httpx

ROOT = Path(__file__).resolve().parents[1]
DISEASES = ("liver", "diabetes", "heart", "kidney", "parkinsons")
MIX = ("prediction",) * 14 + ("report",) * 2 + ("catalog", "metadata", "health", "readiness")
TIMEOUT_SECONDS = 20.0


def _fixtures() -> tuple[dict, dict]:
    golden = json.loads((ROOT / "tests/fixtures/model_release_golden.json").read_text(encoding="utf-8"))
    release = json.loads((ROOT / "models/release_manifest.json").read_text(encoding="utf-8"))
    return {row["disease_identifier"]: row for row in golden["vectors"]}, release["releases"]


def _path(mode: str, kind: str, disease: str | None = None) -> str:
    prefix = "/api/v1" if mode == "direct" else "/api"
    if kind == "health":
        return f"{prefix}/health"
    if kind == "readiness":
        return f"{prefix}/ready"
    if kind == "catalog":
        return f"{prefix}/diseases"
    if kind == "metadata":
        return f"{prefix}/diseases/{disease}"
    if kind == "prediction":
        return f"{prefix}/predictions/{disease}"
    if kind == "report":
        return f"{prefix}/reports/{disease}"
    raise ValueError("unsupported request kind")


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index] * 1000, 3)


def _memory_snapshot(process_pids: list[int]) -> dict[str, float | None]:
    if not process_pids:
        return {}
    try:
        import psutil
    except ImportError:
        return {f"process_{index + 1}": None for index, _ in enumerate(process_pids)}
    result: dict[str, float | None] = {}
    for index, pid in enumerate(process_pids):
        try:
            result[f"process_{index + 1}"] = round(psutil.Process(pid).memory_info().rss / (1024 * 1024), 2)
        except (psutil.Error, OSError):
            result[f"process_{index + 1}"] = None
    return result


async def _metrics_check(url: str, api_key: str) -> dict:
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        response = await client.get(url, headers={"X-API-Key": api_key, "X-Request-ID": str(uuid.uuid4())})
    if response.status_code != 200 or "medical_ai_http_requests_total" not in response.text:
        raise RuntimeError("protected metrics endpoint did not return Prometheus metrics")
    present = {disease: f'disease="{disease}"' in response.text for disease in DISEASES}
    if not all(present.values()):
        raise RuntimeError("metrics did not include every disease label after warm-up/load")
    in_flight_zero = "medical_ai_in_flight_requests 0.0" in response.text
    if not in_flight_zero:
        raise RuntimeError("in-flight request gauge did not recover to zero after the profile")
    return {"protected_endpoint_verified": True, "all_diseases_observed": True,
            "in_flight_zero_after_profile": True}


async def run_profile(
    *, mode: str, base_url: str, concurrency: int, request_count: int | None = None,
    duration_seconds: int | None = None, api_key: str = "", metrics_url: str | None = None,
    process_pids: list[int] | None = None,
) -> dict:
    if mode not in {"direct", "bff"}:
        raise ValueError("mode must be direct or bff")
    if concurrency < 1 or (request_count is not None and request_count < 1) or (
        duration_seconds is not None and duration_seconds < 1
    ):
        raise ValueError("concurrency, requests, and duration must be positive")
    if (request_count is None) == (duration_seconds is None):
        raise ValueError("select exactly one of request_count or duration_seconds")

    vectors, releases = _fixtures()
    selected_models = {
        disease: json.loads((ROOT / "models" / disease / "metrics.json").read_text(encoding="utf-8"))["selected_model"]
        for disease in DISEASES
    }
    disease_cursor = 0
    active = 0
    peak_active = 0
    sequence = 0
    latencies: dict[str, list[float]] = defaultdict(list)
    outcomes: Counter[str] = Counter()
    failure_categories: Counter[str] = Counter()
    client_headers = {"X-Request-ID": str(uuid.uuid4())}
    if mode == "direct":
        if not api_key:
            raise ValueError("direct API mode requires an API key")
        client_headers["X-API-Key"] = api_key

    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/"), timeout=TIMEOUT_SECONDS,
        limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
        follow_redirects=False,
    ) as client:
        # Warm all five pipelines via the selected HTTP boundary before timing.
        async def warm(disease: str) -> None:
            fixture = vectors[disease]
            request_id = str(uuid.uuid4())
            response = await client.post(
                _path(mode, "prediction", disease),
                headers={**client_headers, "X-Request-ID": request_id},
                json={"measurements": fixture["measurements"]},
            )
            _validate_prediction(response, disease, fixture, releases[disease])
            if response.headers.get("X-Request-ID") != request_id:
                raise RuntimeError("request ID was not propagated through the selected HTTP boundary")

        await asyncio.gather(*(warm(disease) for disease in DISEASES))

        def next_job() -> tuple[int, str, str | None]:
            nonlocal sequence, disease_cursor
            current = sequence
            sequence += 1
            kind = MIX[current % len(MIX)]
            disease = None
            if kind in {"prediction", "report", "metadata"}:
                disease = DISEASES[disease_cursor % len(DISEASES)]
                disease_cursor += 1
            return current, kind, disease

        async def one(current: int, kind: str, disease: str | None) -> None:
            nonlocal active, peak_active
            target = _path(mode, kind, disease)
            headers = {**client_headers, "X-Request-ID": str(uuid.uuid4())}
            json_body = None
            if kind in {"prediction", "report"}:
                json_body = {"measurements": vectors[disease]["measurements"]}
            key = f"{kind}_{disease}" if disease else kind
            start = time.perf_counter()
            active += 1
            peak_active = max(peak_active, active)
            try:
                response = await client.request(
                    "POST" if kind in {"prediction", "report"} else "GET",
                    target, json=json_body, headers=headers,
                )
                if response.status_code != 200:
                    raise ProfileFailure("unexpected_http_status")
                if response.headers.get("X-Request-ID") != headers["X-Request-ID"]:
                    raise ProfileFailure("request_id_mismatch")
                if kind == "prediction":
                    _validate_prediction(response, disease, vectors[disease], releases[disease])
                elif kind == "report":
                    if not response.headers.get("content-type", "").lower().startswith("application/pdf"):
                        raise ProfileFailure("invalid_report_content_type")
                    if len(response.content) <= 4 or not response.content.startswith(b"%PDF"):
                        raise ProfileFailure("invalid_pdf_signature")
                elif kind == "metadata":
                    body = response.json()
                    if body.get("slug") != disease or body.get("selected_model") != selected_models[disease]:
                        raise ProfileFailure("invalid_metadata_contract")
                elif kind == "catalog":
                    body = response.json()
                    if {item.get("slug") for item in body} != set(DISEASES):
                        raise ProfileFailure("invalid_catalog_contract")
                elif kind == "health":
                    if response.json().get("status") != "alive":
                        raise ProfileFailure("invalid_health_contract")
                elif kind == "readiness":
                    if response.json().get("status") != "ready":
                        raise ProfileFailure("readiness_not_ready")
                latencies[key].append(time.perf_counter() - start)
                outcomes["success"] += 1
            except ProfileFailure as exc:
                outcomes["error"] += 1
                failure_categories[exc.category] += 1
            except httpx.TimeoutException:
                outcomes["error"] += 1
                failure_categories["timeout"] += 1
            except httpx.HTTPError:
                outcomes["error"] += 1
                failure_categories["network"] += 1
            except (ValueError, KeyError, TypeError):
                outcomes["error"] += 1
                failure_categories["response_contract"] += 1
            finally:
                active -= 1

        started_at = time.perf_counter()
        memory_before = _memory_snapshot(process_pids or [])
        peak_memory = dict(memory_before)
        memory_stop = asyncio.Event()

        async def monitor_memory() -> None:
            if not process_pids:
                return
            while not memory_stop.is_set():
                sample = _memory_snapshot(process_pids)
                for name, value in sample.items():
                    if value is not None and (peak_memory.get(name) is None or value > peak_memory[name]):
                        peak_memory[name] = value
                try:
                    await asyncio.wait_for(memory_stop.wait(), timeout=0.1)
                except TimeoutError:
                    pass

        monitor = asyncio.create_task(monitor_memory())
        try:
            async def worker() -> None:
                while True:
                    if duration_seconds is not None and time.perf_counter() - started_at >= duration_seconds:
                        return
                    if request_count is not None and sequence >= request_count:
                        return
                    job = next_job()
                    await one(*job)

            await asyncio.gather(*(worker() for _ in range(concurrency)))
        finally:
            memory_stop.set()
            await monitor
        elapsed = time.perf_counter() - started_at
        memory_after = _memory_snapshot(process_pids or [])

    if metrics_url:
        if not api_key:
            raise ValueError("metrics verification requires an API key")
        metrics = await _metrics_check(metrics_url, api_key)
    else:
        metrics = {"protected_endpoint_verified": False, "all_diseases_observed": False}

    def summary(values: list[float]) -> dict:
        return {"count": len(values), "p50_ms": _percentile(values, .50),
                "p95_ms": _percentile(values, .95), "p99_ms": _percentile(values, .99)}

    categories = {key: summary(value) for key, value in sorted(latencies.items())}
    prediction_times = [v for key, values in latencies.items() if key.startswith("prediction_") for v in values]
    report_times = [v for key, values in latencies.items() if key.startswith("report_") for v in values]
    errors = outcomes["error"]
    total = outcomes["success"] + errors
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile": {"mode": mode, "concurrency": concurrency,
                    "request_count": total, "duration_seconds": round(elapsed, 3),
                    "timeout_seconds": TIMEOUT_SECONDS},
        "results": {"successful_requests": outcomes["success"], "errors": errors,
                    "error_rate": round(errors / total, 6) if total else 0.0,
                    "throughput_requests_per_second": round(total / elapsed, 3) if elapsed else 0,
                    "peak_in_flight": peak_active, "failure_categories": dict(sorted(failure_categories.items())),
                    "latency": {"overall": summary([v for values in latencies.values() for v in values]),
                                "predictions": summary(prediction_times), "reports": summary(report_times),
                                "by_category": categories}},
        "memory_mb": {"before": memory_before, "peak": peak_memory, "after": memory_after,
                      "measurable": bool(process_pids)},
        "metrics_verification": metrics,
        "acceptance": {"unexpected_error_rate_zero": errors == 0,
                       "prediction_p95_under_2s": _percentile(prediction_times, .95) is not None and _percentile(prediction_times, .95) < 2000,
                       "report_p95_under_5s": _percentile(report_times, .95) is not None and _percentile(report_times, .95) < 5000},
        "clinical_validity": "not assessed; this is local engineering evidence only",
    }


class ProfileFailure(Exception):
    def __init__(self, category: str):
        super().__init__(category)
        self.category = category


def _validate_prediction(response: httpx.Response, disease: str, fixture: dict, release: dict) -> None:
    if response.status_code != 200:
        raise ProfileFailure("unexpected_http_status")
    payload = response.json()
    score = payload.get("model_score")
    if (payload.get("disease") != disease or payload.get("prediction") != fixture.get("expected_prediction")
            or payload.get("model_identifier") != release["model"]["identifier"]
            or payload.get("release_status") != release["release_status"]
            or payload.get("decision_threshold") != fixture.get("threshold")
            or not isinstance(score, (int, float))
            or abs(float(score) - float(fixture["expected_model_score"])) > 1e-8):
        raise ProfileFailure("prediction_contract_or_golden_mismatch")
    expected_result = "at_or_above" if fixture["expected_prediction"] else "below"
    if payload.get("threshold_result") != expected_result:
        raise ProfileFailure("threshold_result_mismatch")


def _markdown(report: dict) -> str:
    profile, results = report["profile"], report["results"]
    latency = results["latency"]
    lines = [
        "# Local load-test result",
        "",
        f"Mode: `{profile['mode']}`; workers: {profile['concurrency']}; requests: {profile['request_count']}; elapsed: {profile['duration_seconds']} s.",
        f"Runtime: `{json.dumps(report.get('environment', {}), sort_keys=True)}`.",
        "",
        "This run measures only this machine/configuration. It is not a universal capacity promise or clinical validation.",
        "",
        "| Scope | Count | p50 ms | p95 ms | p99 ms |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label in ("overall", "predictions", "reports"):
        item = latency[label]
        lines.append(f"| {label} | {item['count']} | {item['p50_ms']} | {item['p95_ms']} | {item['p99_ms']} |")
    lines.extend([
        "", f"Errors: {results['errors']} ({results['error_rate']:.2%}); throughput: {results['throughput_requests_per_second']} requests/s; peak in-flight: {results['peak_in_flight']}.",
        f"Memory observations (MB): `{json.dumps(report['memory_mb'], sort_keys=True)}`.",
        f"Provisional engineering thresholds: `{json.dumps(report['acceptance'], sort_keys=True)}`.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("direct", "bff"), required=True)
    parser.add_argument("--base-url", required=True, help="Private FastAPI URL or same-origin Next.js URL")
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--requests", type=int, default=500)
    size.add_argument("--duration-seconds", type=int)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--metrics-url", help="Optional protected direct metrics URL")
    parser.add_argument("--memory-pid", action="append", type=int, default=[], help="Service PID to sample; repeat for API/BFF")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    api_key = os.environ.get("API_KEY", "")
    report = asyncio.run(run_profile(
        mode=args.target, base_url=args.base_url, concurrency=args.concurrency,
        request_count=args.requests if args.duration_seconds is None else None,
        duration_seconds=args.duration_seconds, api_key=api_key,
        metrics_url=args.metrics_url, process_pids=args.memory_pid,
    ))
    json_output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    markdown_output = _markdown(report)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json_output, encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown_output, encoding="utf-8")
    if args.json_output or args.markdown_output:
        print(f"Load profile complete: {report['results']['successful_requests']} successful, {report['results']['errors']} errors; output files written.")
    else:
        print(json_output, end="")
        print(markdown_output)
    return 0 if report["results"]["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
