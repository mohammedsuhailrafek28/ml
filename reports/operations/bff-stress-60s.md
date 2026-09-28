# Local load-test result

Mode: `bff`; workers: 25; requests: 2651; elapsed: 60.294 s.
Runtime: `{"node": "v20.19.0", "platform": "Windows-10-10.0.26200-SP0", "python": "3.11.15", "target": "isolated local production BFF (API+BFF services started)"}`.

This run measures only this machine/configuration. It is not a universal capacity promise or clinical validation.

| Scope | Count | p50 ms | p95 ms | p99 ms |
| --- | ---: | ---: | ---: | ---: |
| overall | 2651 | 547.188 | 919.606 | 1193.35 |
| predictions | 1859 | 559.914 | 736.563 | 864.81 |
| reports | 264 | 538.784 | 691.344 | 820.598 |

Errors: 0 (0.00%); throughput: 43.968 requests/s; peak in-flight: 25.
Memory observations (MB): `{"after": {"process_1": 4.18, "process_2": 110.94}, "before": {"process_1": 4.2, "process_2": 73.93}, "measurable": true, "peak": {"process_1": 4.2, "process_2": 120.22}}`.
Provisional engineering thresholds: `{"prediction_p95_under_2s": true, "report_p95_under_5s": true, "unexpected_error_rate_zero": true}`.
