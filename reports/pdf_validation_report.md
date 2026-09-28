# PDF validation and privacy audit

The seven active-directory PDF artifacts were inspected with a temporary `pypdf` installation on 2026-09-27. All were parseable one-page PDFs and contained prediction output plus raw input measurements. No direct name, date-of-birth, email, or phone field was found, but health measurements remain sensitive data. The repository did not establish their synthetic/public-data provenance, and their creation dates predate the current model release. They were removed; see [the artifact audit](generated_reports/README.md).

The historical evaluation PDF in `reports/evaluation/superseded/` is preserved as historical evidence, not a current report example. Runtime PDF generation remains covered by the API and browser tests, which verify the `%PDF` signature and non-empty response without retaining generated reports on disk.
