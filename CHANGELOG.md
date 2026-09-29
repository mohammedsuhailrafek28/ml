# Changelog

## 0.9.0-rc.1 — educational/research release candidate

- Establishes a public-safe application version and release provenance.
- Adds separate non-root, multi-stage backend and standalone Next.js images.
- Adds a private Compose backend network, runtime-mounted service credential,
  read-only container roots, restricted capabilities, and bounded logs/resources.
- Adds container golden-vector, PDF, readiness, metrics-auth, and browser checks;
  scans the candidate source tree and Git history with Gitleaks, scans built
  images with Trivy, and produces CI-only CycloneDX SBOMs and redacted reports.
- Does not change persisted models, datasets, preprocessing, thresholds, metrics,
  or golden prediction vectors.

This is not a clinical release and is not validated for diagnosis or screening.
