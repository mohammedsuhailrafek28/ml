# Generated report artifact audit

The active report directory intentionally contains no checked-in PDFs. The seven PDFs listed below were removed during the Phase 7 privacy/provenance audit. Each was a one-page report containing raw health measurements and prediction output, with no direct patient identifier detected. Their synthetic/public-data provenance could not be established from repository evidence, and the files predated the current model release. They are therefore not authoritative examples and should not be redistributed.

| Removed artifact | PDF creation date | Evidence / disposition |
| --- | --- | --- |
| `diabetes-api-report.pdf` | 2026-07-24 | Measurements present; provenance unverified; removed |
| `diabetes_sample_report.pdf` | 2026-08-27 | Measurements present; provenance unverified; removed |
| `heart_sample_report.pdf` | 2026-08-27 | Measurements present; provenance unverified; removed |
| `kidney_sample_report.pdf` | 2026-08-27 | Measurements present; provenance unverified; removed |
| `liver-api-report.pdf` | 2026-07-24 | Measurements present; provenance unverified; removed |
| `liver_sample_report.pdf` | 2026-07-24 | Measurements present; provenance unverified; removed |
| `parkinsons_sample_report.pdf` | 2026-08-27 | Measurements present; provenance unverified; removed |

The historical PDF under `reports/evaluation/superseded/` is preserved unchanged as evaluation evidence. It is not an active sample report. Runtime-generated PDFs remain available to the application; do not check user-generated reports into source control. To create a shareable example in future, use explicitly synthetic values, label it non-clinical, and record its provenance and model-release identifier.
