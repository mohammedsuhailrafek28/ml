
## Liver

The UCI Indian Liver Patient Dataset (ILPD) contains 583 records, 9 numeric predictors, one
categorical predictor (`Gender`), and the `Selector` target (1 = liver-disease group,
2 = control), remapped to {1, 0}. Cleaning for the liver trainer (`src/training/train_liver.py`):

* Rows with a missing target: 0 removed.
* Exact duplicate rows: **13 removed before splitting** (ILPD has no patient identifier, so
  identical rows must not be allowed to straddle the train/test boundary). 570 rows remain.
* `A/G Ratio` has 4 missing values; these are imputed with the training-fold median inside the
  pipeline (never fitted on validation or holdout data). `Gender` missing values would be
  imputed with the training-fold mode.
* Extreme enzyme values (e.g. SGOT up to ~5000 IU/L) are retained; no statistical outlier
  deletion. Negative values are out of range and rejected at the API boundary.
* Class balance after cleaning: 406 positive / 164 negative (~71 % positive). Handled with a
  CV-tuned `class_weight` and an unchanged 0.5 decision threshold, not resampling.

## Heart

The UCI **Cleveland** Heart Disease database contains 303 records, 5 continuous/integer
predictors (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) and 8 integer-coded
*categorical* predictors (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`).
The `num` target (0 = <50 % vessel narrowing, 1–4 = >50 %) is remapped to {0, 1}. Cleaning
for the heart trainer (`src/training/train_heart.py`):

* Missing values are the literal `?` in the raw file, read as `NaN`: **4 in `ca`, 2 in
  `thal`**, nothing else. They are imputed with the training-fold mode inside the pipeline
  (numeric features would use the training-fold median). No rows are dropped for missingness.
* Rows with a missing target: 0. Exact duplicate rows: **0** (checked before splitting;
  Cleveland has no patient identifier, so the guard is kept regardless).
* Categorical codes are normalised to canonical string tokens
  (`cp` → `"1".."4"`, `thal` → `"3"/"6"/"7"`, `ca` → `"0".."3"`, …) so training and
  inference feed the one-hot encoder the *identical* representation. This closes the
  previous `ca`/`thal` train/serve mismatch where JSON ints were silently encoded as
  all-zeros. `restecg`, `slope`, etc. are one-hot encoded, never scaled as if continuous.
* Extreme but physiologically possible values (`chol` up to 564 mg/dL) are retained; no
  statistical outlier deletion. The API rejects values outside generous sanity ranges.
* Class balance after cleaning: 139 positive / 164 negative (~46 % positive) — near
  balanced. `class_weight` (None vs `balanced`) is a CV-tuned hyper-parameter; the
  decision threshold is left at 0.5 (a dev-only sweep is published, not used).

## Diabetes

The Pima Indians Diabetes Database contains 768 records (Pima women aged 21+) and 8 numeric
predictors. Cleaning for the diabetes trainer (`src/training/train_diabetes.py`):

* Rows with a missing target: 0. Exact duplicate rows: **0** (checked before splitting;
  Pima has no patient identifier).
* **Zero-as-missing:** a value of exactly `0` in `glucose` (5 rows), `blood_pressure` (35),
  `skin_thickness` (227), `insulin` (374) or `bmi` (11) is physiologically impossible in a
  living adult and is the dataset's missing-value placeholder. A `ZeroToNaN` transformer —
  the **first step inside the persisted pipeline** — converts those to `NaN`, then
  `SimpleImputer(median)` fills them using training-fold medians only. `pregnancies == 0`
  (nulliparous) and `age` are genuine values and are left untouched. The previous build did
  this zero→NaN step *outside* the pipeline, so the shipped model never applied it at
  inference — a submitted `0` was scaled as a real value. That is now fixed.
* Heavily missing columns (`insulin` ~49 %, `skin_thickness` ~30 %) are imputed rather than
  dropped; dropping them would discard roughly half the dataset.
* Extreme but possible values are retained; no statistical outlier deletion. The API rejects
  values outside generous sanity ranges and rejects negatives.
* Class balance: 268 positive / 500 negative (~35 % positive). `class_weight` (None vs
  `balanced`) is a CV-tuned hyper-parameter; the decision threshold is left at 0.5 (a
  dev-only sweep is published, not used). No SMOTE. A dev-only CV experiment with engineered
  interaction terms (`glucose×bmi`, `bmi×age`, `glucose/insulin`) was run and **rejected**
  (mean CV ROC-AUC change −0.005).

## Chronic Kidney Disease

The UCI CKD dataset contains 400 records and 24 predictors + `class`. Cleaning for the
kidney trainer (`src/training/train_kidney.py`):

* **`id` column dropped before training.** The rows are sorted by class (id 0–249 are all
  `ckd`, id 250–399 are all `notckd`), so `id` is a perfect target proxy. It was already
  absent from the feature list; the trainer drops it explicitly and a test asserts the
  proxy relationship and its exclusion.
* **Target normalisation:** 2 rows carry a trailing tab (`ckd\t`). The target is stripped
  of whitespace, lower-cased, then mapped `ckd → 1`, `notckd → 0` (250 / 150).
* **Categorical token cleaning:** `dm` and `cad` contain stray tab/space variants
  (`"\tno"`, `"\tyes"`, `" yes"`). Every categorical is normalised by lower-casing and
  removing all non `a–z` characters, so `"\tno" → "no"`. The old persisted OneHotEncoder
  had learned `dm_\tno`, `dm_ yes`, `cad_\tno` as distinct categories — a train/serve
  hazard now removed.
* **Missing values:** the file is heavily incomplete — `rbc` 38 %, `rc` 33 %, `wc` 27 %,
  `pot`/`sod` ~22 %, `pcv` 18 %, `pc` 16 %, `hemo` 13 %, and 15 more columns below 13 %.
  Numeric → `SimpleImputer(median)`, categorical → `SimpleImputer(most_frequent)`, both
  fitted inside CV folds only. **No rows dropped for missingness** (that would lose a large
  fraction of the data).
* **Missingness is itself target-correlated** (a collection artefact): CKD rows average 3.8
  missing labs vs 0.8 for non-CKD; the count of missing features has AUC ≈ 0.82 vs the
  target. We deliberately **do not** add missing-indicator features; imputation means the
  model sees imputed values, not raw missingness. Recorded as a limitation.
* Exact duplicate rows: **0** (checked before splitting; 0 conflicting-label feature
  vectors). Class balance 250 / 150 (~63 % positive); `class_weight` is CV-tuned; threshold
  left at 0.5.
* **Feature-selection experiment:** a fold-safe comparison (LR + RF, DEV CV) of the full
  24-feature set vs a compact 14-feature clinical set found no loss of ROC-AUC and lower
  variance, so the **14-feature set was adopted** for production
  (`hemo, sc, sg, al, pcv, rc, bgr, bu, age, bp, htn, dm, appet, ane`).

## Parkinson's Disease

The UCI Parkinson's voice dataset contains 195 sustained-vowel recordings from **32 subjects**
(~6 recordings each), 22 acoustic voice biomarkers, a `name` sample identifier and the
subject-level `status` label. Handling for the Parkinson's trainer
(`src/training/train_parkinsons.py`):

* **Subject identity** is extracted from `name` with `^(phon_R\d+_S\d+)` (`phon_R01_S07_4` →
  `phon_R01_S07`). Verified: every S-number maps to exactly one R and one `status`; 32
  groups, no null/ambiguous groups. `name` and `status` are **never** model features.
* **Subject leakage is the central risk.** All splitting is subject-aware:
  `GroupShuffleSplit(test_size=0.2)` for a subject-disjoint DEV/HOLDOUT (25 vs 7 subjects),
  `StratifiedGroupKFold(5)` with `groups=subject` for all tuning/selection/threshold work.
  A diagnostic naive row-level `train_test_split` scores **ROC-AUC 0.924** vs the
  subject-aware **0.573** for the same probe model — a **0.35 inflation** that is pure
  leakage. Only the subject-aware numbers are reported as production performance.
* No missing values; a defensive `SimpleImputer(median)` precedes `StandardScaler`. 0 exact
  duplicate rows (multiple legitimate recordings per subject are expected and kept).
* **Feature-redundancy experiment:** the jitter family (`RAP`, `PPQ`, `DDP`) and shimmer
  family (`Shimmer`, `APQ3`, `APQ5`, `DDA`) are pairwise collinear (|r| > 0.95). A
  group-aware DEV-CV comparison of the full 22 vs a reduced 15-feature set showed no ROC-AUC
  loss and better stability, so the **15-feature set was adopted**.
* Class balance: 147/48 by row, **24/8 by subject** (~75 % positive). `class_weight` is
  CV-tuned; threshold left at 0.5.
* **Honest performance:** group-CV ROC-AUC 0.88 ± 0.12 (seed 42), multi-seed group-CV mean
  ≈ 0.78; the 7-subject holdout (ROC-AUC 0.59) is too small for a reliable point estimate
  and is reported with that caveat. The previous metadata's claim that "no group split
  [was] implemented" was **false** (the old shared trainer did use GroupShuffleSplit +
  GroupKFold) and is corrected.
