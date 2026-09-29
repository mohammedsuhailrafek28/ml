# Local load-test result

Mode: `bff`; workers: 10; requests: 500; elapsed: 13.003 s.
Runtime: `{"node": "v20.19.0", "platform": "Windows-10-10.0.26200-SP0", "python": "3.11.15", "target": "isolated local production BFF (API+BFF services started)"}`.

This run measures only this machine/configuration. It is not a universal capacity promise or clinical validation.

| Scope | Count | p50 ms | p95 ms | p99 ms |
| --- | ---: | ---: | ---: | ---: |
| overall | 500 | 232.263 | 467.826 | 835.784 |
| predictions | 350 | 233.571 | 369.489 | 462.632 |
| reports | 50 | 238.607 | 367.829 | 411.291 |

Errors: 0 (0.00%); throughput: 38.453 requests/s; peak in-flight: 10.
Memory observations (MB): `{"after": {"process_1": 4.23, "process_2": 116.09}, "before": {"process_1": 4.23, "process_2": 74.18}, "measurable": true, "peak": {"process_1": 4.23, "process_2": 116.68}}`.
Provisional engineering thresholds: `{"prediction_p95_under_2s": true, "report_p95_under_5s": true, "unexpected_error_rate_zero": true}`.
