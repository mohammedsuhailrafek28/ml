# Current persisted-model release verification

Release: `medical-ai-suite-model-release-2026-09-24`

This is the current report. It verifies file identity and executes one locked
golden prediction through the production registry for each disease. It does not
retrain models or recreate historical holdout splits. Evaluation metrics below
come from the metadata stored with each released artifact.

| Disease | Model | SHA-256 | Golden class | Golden model score | Status |
|---|---|---|---:|---:|---|
| liver | LogisticRegression | `ab2eb1b92956119b8814a1ec4b462d8316edbf5543445bcfa340d62fad35a960` | 1 | 0.641961451315 | educational/research release |
| diabetes | LogisticRegression | `07af879e4bb77d9bf3bc283bb0c4b40f9b4f7f006ef7aa9dee918d8cdb7f3b2f` | 1 | 0.800530460666 | educational/research release |
| heart | LogisticRegression | `5220edd64dcae73f30d611bc4ed90614c17576ae66f911fb4c5e23fe8281052f` | 0 | 0.459835914318 | educational/research release |
| kidney | LogisticRegression | `8ef6da06f2f70cdb448ff73abd31e914a6b759fdda1bf063923f3cec6be4ad58` | 1 | 0.868915885678 | educational/research release |
| parkinsons | LogisticRegression | `43bfa333fd5f3f13d9db5556d667408cefa7a4e6d795ba0036f16dcbe27d7585` | 1 | 0.969005143388 | experimental |

Parkinson's remains experimental: its subject-disjoint holdout contains only
seven subjects and reports ROC-AUC 0.5860 with specificity 0.0 at threshold 0.5.
The stronger development group-CV result is not a substitute for external validation.
