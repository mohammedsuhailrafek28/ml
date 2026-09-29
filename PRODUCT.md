# Medical AI Suite

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primarily researchers and technical reviewers, as confirmed for Phase 9. Forms
must remain approachable to ordinary users exploring the research demonstrator.

## Product Purpose

Inspect five persisted public-dataset models through structured measurements,
uncalibrated scores, fixed threshold classifications, and downloadable reports.
This is an educational/research release candidate, not a diagnostic or clinical
screening product.

## Capabilities and Constraints

- Liver, Diabetes, Heart, Kidney, and Parkinson's share one assessment wizard.
- Model provenance, intended use, limitations, and threshold semantics must stay visible.
- Parkinson's remains Experimental, with its existing evidence and acknowledgement gate.
- No raw-audio processing; Parkinson's accepts pre-computed biomarkers only.
- Browser requests use the authenticated server-side boundary; measurements are
  not intentionally persisted in browser storage and reports are created in memory.
- Phase 9 cannot change model artifacts, datasets, training/preprocessing,
  thresholds, metrics, golden vectors, contracts, or evaluation evidence.
- Existing URLs and functional journeys remain intact.

## Brand Commitments

Medical AI Suite: calm, precise, contemporary Clinical Intelligence presentation.
Warm off-white canvas, white surfaces, deep ink/navy, restrained medical teal;
muted amber exclusively for Experimental content. No decorative AI imagery,
stock medical photography, neon, glassmorphism, or excessive gradients. Evidence
and limitations carry trust, not claims of clinical validity.

## Evidence on Hand

`models/release_manifest.json`, `reports/evaluation/current/`, generated input
contracts, and `tests/fixtures/model_release_golden.json` are authoritative.
The baseline is `021cdf6a3cca0d6e878b76ab4bcd61bf283ca6a9`; the published
`v0.9.0-rc.1` tag must not move. No new clinical claims may be invented.

## Product Principles

1. Distinguish dataset-associated output from medical inference.
2. Make provenance and limitations as inspectable as scores.
3. Preserve validated prediction and privacy boundaries.
4. Treat incomplete evidence candidly, especially for Parkinson's.

## Accessibility & Inclusion

WCAG AA contrast, keyboard completion, visible focus, reduced-motion support,
200% zoom, and layouts at 320, 390, 768, 1024, and 1440 pixels. Use local font
resources without runtime remote-font requests.
