# 0.9.0-rc.1 release verification checklist

This checklist is for review of an educational/research release candidate. It
does not authorize publishing or deployment. The application is not a diagnosis
or screening service and is not clinically validated; Parkinson's remains
Experimental and scores are uncalibrated. Local load tests do not prove
availability, and no deployed availability SLO has been measured.

- [ ] Confirm branch, source revision, and uncommitted diff review.
- [ ] `python -m scripts.build_application_release_manifest --check`
- [ ] `python -m scripts.build_model_release_manifest --check`
- [ ] `python -m scripts.generate_frontend_contracts --check`
- [ ] `python scripts/evaluate_persisted_models.py --check`
- [ ] Full hash-locked Python checks and all backend tests.
- [ ] Under Node 20.19.0: `npm ci`, both audits, typecheck, lint, build,
      security build/test, and the full Playwright suite.
- [ ] `docker compose config --quiet`; build both images with Buildx.
- [ ] Verify Gitleaks current-source and full reachable Git-history reports.
- [ ] Verify Trivy Critical/High vulnerability and embedded-secret reports for
      both final images, plus both CycloneDX SBOMs.
- [ ] Confirm scan reports are redacted and inspect any findings or exceptions.
- [ ] Start Compose and verify only loopback frontend publishing, internal API,
      service-key protection for metrics, non-root users, read-only roots,
      capability drop, and `no-new-privileges`.
- [ ] Validate all five golden predictions, unchanged thresholds, and all five
      PDF `%PDF` downloads; verify no report file persists in either container.
- [ ] Run browser journeys, bounded load smoke, restart/readiness recovery, and
      cleanup; ensure diagnostics contain no private payloads or credentials.
- [ ] Review backup and coordinated rollback procedure for a compatible image
      pair, configuration, model manifest, and secret.
- [ ] Record hosted CI results only after observing the actual workflow run.
- [ ] Confirm no push, tag, publication, or deployment occurred without separate
      approval.

The deterministic application release-input manifest binds application
identity, model-release manifest, dependency lockfiles, Dockerfiles, Node/Python
runtime identifiers, generated frontend contracts, and Next configuration.
Source revision is injected at runtime and is not a future commit hash baked
into the deterministic artifact.
