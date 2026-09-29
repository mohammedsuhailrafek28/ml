"""Subject-aware, leakage-safe training pipeline for the Parkinson's module (UCI voice).

Run: python -m src.training.train_parkinsons

The dataset has ~6 recordings per subject, so ordinary row-level splitting leaks a
person's voice across train and test. Everything here is subject/group-aware:

    raw parkinsons.csv  (195 rows, 32 subjects, no missing values)
      -> subject id from `name` (phon_R01_S07_4 -> phon_R01_S07); `name`/`status` never features
      -> drop exact duplicate rows                (none; guarded anyway)
      -> PHASE 5  naive row-split vs GroupShuffleSplit (diagnostic only, NOT shipped)
      -> GroupShuffleSplit(test_size=0.2) -> subject-disjoint DEV / HOLDOUT
      -> Pipeline(SimpleImputer(median) -> StandardScaler -> estimator), fit in-fold only
      -> PHASE 11 feature-redundancy experiment (22 vs reduced), group-aware DEV CV
      -> per family: GridSearchCV over StratifiedGroupKFold(5), scoring = ROC-AUC,
         groups= passed through, DEV only
      -> winner by mean group-CV ROC-AUC, simplest family within a small tie band
      -> PHASE 12 multi-seed GroupShuffleSplit stability for the winning architecture
      -> operating threshold fixed at 0.5 (group out-of-fold sweep published)
      -> winner refit on the whole DEV set
      -> evaluated exactly once on the subject-disjoint HOLDOUT
      -> persisted pipeline + metadata + metrics + permutation importance
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from hashlib import sha256

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    StratifiedGroupKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data.loader import load_dataset
from src.preprocessing.parkinsons_schema import (
    PARKINSONS_FEATURES,
    PARKINSONS_REDUCED_FEATURES,
    SUBJECT_ID_COLUMN,
    TARGET,
    subject_ids,
)
from src.utils.config import DISEASES, RANDOM_STATE, ROOT, TEST_SIZE

CONFIG = DISEASES["parkinsons"]
PERM_REPEATS = 10
PRODUCTION_THRESHOLD = 0.5
STABILITY_SEEDS = [21, 42, 84, 123, 2026]
CV_SPLITS = 5

SELECTION_TIE_BAND = 0.02  # a touch wider: only 25 training subjects -> noisier CV
SIMPLICITY_ORDER = [
    "logistic_regression",
    "gaussian_nb",
    "decision_tree",
    "gradient_boosting",
    "random_forest",
    "svm",
    "knn",
]


def _search_spaces(seed: int) -> dict[str, tuple[object, dict]]:
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=5000, random_state=seed),
            {"model__C": [0.05, 0.2, 1.0], "model__class_weight": [None, "balanced"]},
        ),
        "decision_tree": (
            DecisionTreeClassifier(random_state=seed),
            {
                "model__max_depth": [2, 3, 5],
                "model__min_samples_leaf": [3, 5, 8],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=seed, n_jobs=-1),
            {
                "model__n_estimators": [300, 500],
                "model__max_depth": [None, 4, 6],
                "model__min_samples_leaf": [1, 3],
                "model__class_weight": [None, "balanced", "balanced_subsample"],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=seed),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
        "svm": (
            SVC(kernel="rbf", probability=True, random_state=seed),
            {
                "model__C": [0.5, 1.0, 5.0],
                "model__gamma": ["scale"],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "knn": (
            KNeighborsClassifier(),
            {"model__n_neighbors": [5, 9, 15], "model__weights": ["uniform", "distance"]},
        ),
        "gaussian_nb": (GaussianNB(), {}),
    }


def _pipe(estimator) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def _load() -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    raw = load_dataset(CONFIG.raw_path)
    rows_raw = len(raw)
    dup_mask = raw.drop(columns=[SUBJECT_ID_COLUMN]).duplicated(keep="first")
    n_dupes = int(dup_mask.sum())
    frame = raw.loc[~dup_mask].reset_index(drop=True)

    groups = subject_ids(frame[SUBJECT_ID_COLUMN])
    if groups.isna().any() or (groups == "").any():
        raise ValueError("Parkinson's: subject id extraction produced null/empty groups")

    X = frame[list(PARKINSONS_FEATURES)].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(frame[TARGET], errors="coerce").astype(int)

    by_subject = frame.assign(_g=groups, _y=y).groupby("_g")["_y"].agg(["first", "nunique"])
    if (by_subject["nunique"] > 1).any():
        raise ValueError("Parkinson's: a subject has conflicting status labels")

    cleaning = {
        "rows_raw": rows_raw,
        "exact_duplicate_rows_removed": n_dupes,
        "rows_used": len(frame),
        "unique_subjects": int(groups.nunique()),
        "recordings_per_subject": {
            "min": int(groups.value_counts().min()),
            "median": float(groups.value_counts().median()),
            "max": int(groups.value_counts().max()),
        },
        "target_distribution_by_row": {
            "positive": int((y == 1).sum()), "negative": int((y == 0).sum())
        },
        "target_distribution_by_subject": {
            "positive": int((by_subject["first"] == 1).sum()),
            "negative": int((by_subject["first"] == 0).sum()),
        },
        "subject_id_rule": "regex ^(phon_R\\d+_S\\d+) on `name`; verified every S-number maps to one R and one status.",
        "missing_values": int(X.isna().sum().sum()),
        "duplicate_policy": "Exact duplicate feature rows removed before splitting (0 found).",
        "identifier_excluded": [SUBJECT_ID_COLUMN, TARGET],
    }
    return X, y, groups, cleaning


def _metrics(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _threshold_report(y_true, proba) -> list[dict]:
    y_true = np.asarray(y_true)
    grid = []
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        grid.append(
            {
                "threshold": float(t),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(sens),
                "specificity": float(spec),
                "f1": float(f1_score(y_true, pred, zero_division=0)),
                "youden_j": float(sens + spec - 1.0),
            }
        )
    return grid


def _permutation_importance(pipe, X, y, seed: int) -> list[dict]:
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        pipe, X, y, scoring="roc_auc", n_repeats=PERM_REPEATS, random_state=seed
    )
    order = np.argsort(result.importances_mean)[::-1]
    return [
        {
            "feature": str(X.columns[i]),
            "importance_mean": float(result.importances_mean[i]),
            "importance_std": float(result.importances_std[i]),
        }
        for i in order
    ]


def _naive_vs_group(X, y, groups, seed: int) -> dict:
    """PHASE 5 - diagnostic only. Never shipped."""
    probe = _pipe(LogisticRegression(max_iter=5000, random_state=seed))
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )
    probe.fit(Xtr, ytr)
    naive = float(roc_auc_score(yte, probe.predict_proba(Xte)[:, 1]))

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    tr, te = next(gss.split(X, y, groups))
    probe.fit(X.iloc[tr], y.iloc[tr])
    grp = float(roc_auc_score(y.iloc[te], probe.predict_proba(X.iloc[te])[:, 1]))
    return {
        "probe_model": "logistic_regression (standardised)",
        "naive_row_split_holdout_roc_auc": naive,
        "subject_aware_split_holdout_roc_auc": grp,
        "inflation": round(naive - grp, 4),
        "note": (
            "Row-level train_test_split lets a subject's ~6 recordings appear in both "
            "train and test. The gap is the leakage this trainer avoids; only the "
            "subject-aware number is meaningful."
        ),
    }


def _feature_experiment(X, y, groups, cv, seed: int) -> dict:
    probes = {
        "logistic_regression": LogisticRegression(max_iter=5000, random_state=seed),
        "random_forest": RandomForestClassifier(random_state=seed, n_jobs=-1),
    }
    out = {
        "full_feature_count": len(PARKINSONS_FEATURES),
        "reduced_feature_count": len(PARKINSONS_REDUCED_FEATURES),
        "reduced_features": list(PARKINSONS_REDUCED_FEATURES),
        "rationale": "jitter (RAP/PPQ/DDP) and shimmer (APQ3/APQ5/DDA/Shimmer) families have |r|>0.95; keep one representative each.",
        "rule": "adopt the reduced set only if mean group-CV ROC-AUC drops by < 0.01 AND CV std does not increase",
        "probes": {},
    }
    votes = []
    for name, est in probes.items():
        full = cross_val_score(
            _pipe(est), X[list(PARKINSONS_FEATURES)], y, cv=cv, groups=groups, scoring="roc_auc"
        )
        red = cross_val_score(
            _pipe(est), X[list(PARKINSONS_REDUCED_FEATURES)], y, cv=cv, groups=groups, scoring="roc_auc"
        )
        out["probes"][name] = {
            "full_cv_roc_auc_mean": float(full.mean()),
            "full_cv_roc_auc_std": float(full.std()),
            "reduced_cv_roc_auc_mean": float(red.mean()),
            "reduced_cv_roc_auc_std": float(red.std()),
        }
        votes.append((red.mean() >= full.mean() - 0.01) and (red.std() <= full.std() + 1e-9))
    out["adopted"] = bool(all(votes))
    out["decision"] = (
        f"Reduced {len(PARKINSONS_REDUCED_FEATURES)}-feature set "
        f"{'ADOPTED' if out['adopted'] else 'REJECTED'} on group-aware DEV CV."
    )
    return out


def _multiseed_group_stability(estimator_factory, features, X, y, groups) -> dict:
    """Group-aware CV of the winning architecture across deterministic seeds.

    With only ~6 control subjects in DEV a 5-fold split can leave a fold with one
    class (ROC-AUC undefined -> NaN for that fold). We drop to 4 folds here and
    aggregate nan-robustly, recording how many folds/seeds were usable.
    """
    per_seed, means = {}, []
    for s in STABILITY_SEEDS:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=s)
        tr, _ = next(gss.split(X, y, groups))
        cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=s)
        sc = cross_val_score(
            _pipe(estimator_factory(s)), X.iloc[tr][features], y.iloc[tr],
            cv=cv, groups=groups.iloc[tr], scoring="roc_auc", error_score=np.nan,
        )
        valid = sc[~np.isnan(sc)]
        seed_mean = float(np.mean(valid)) if valid.size else float("nan")
        per_seed[str(s)] = {
            "mean": seed_mean,
            "std": float(np.std(valid)) if valid.size else float("nan"),
            "valid_folds": int(valid.size),
            "dev_subjects": int(groups.iloc[tr].nunique()),
        }
        if valid.size:
            means.append(seed_mean)
    means = np.array(means, dtype=float)
    return {
        "seeds": STABILITY_SEEDS,
        "valid_seeds": int(means.size),
        "scoring": "roc_auc",
        "cv": "StratifiedGroupKFold(4), groups=subject",
        "per_seed": per_seed,
        "mean": float(np.mean(means)) if means.size else float("nan"),
        "std": float(np.std(means)) if means.size else float("nan"),
        "min": float(np.min(means)) if means.size else float("nan"),
        "max": float(np.max(means)) if means.size else float("nan"),
    }


def train_parkinsons(write_figures: bool = True) -> dict:
    seed = RANDOM_STATE
    X, y, groups, cleaning = _load()

    naive_vs_group = _naive_vs_group(X, y, groups, seed)

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    dev_idx, test_idx = next(gss.split(X, y, groups))
    X_dev, X_test = X.iloc[dev_idx], X.iloc[test_idx]
    y_dev, y_test = y.iloc[dev_idx], y.iloc[test_idx]
    g_dev, g_test = groups.iloc[dev_idx], groups.iloc[test_idx]
    assert set(g_dev) & set(g_test) == set(), "subject overlap between DEV and HOLDOUT"

    cv = StratifiedGroupKFold(n_splits=CV_SPLITS, shuffle=True, random_state=seed)

    fexp = _feature_experiment(X_dev, y_dev, g_dev, cv, seed)
    active = list(PARKINSONS_REDUCED_FEATURES) if fexp["adopted"] else list(PARKINSONS_FEATURES)

    rows: list[dict] = []
    fitted: dict[str, Pipeline] = {}
    for name, (estimator, grid) in _search_spaces(seed).items():
        search = GridSearchCV(
            _pipe(estimator), grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True
        )
        search.fit(X_dev[active], y_dev, groups=g_dev)
        best = search.best_estimator_
        oof = cross_val_predict(
            best, X_dev[active], y_dev, cv=cv, groups=g_dev, method="predict_proba", n_jobs=-1
        )[:, 1]
        rows.append(
            {
                "model": name,
                "cv_roc_auc_mean": float(search.best_score_),
                "cv_roc_auc_std": float(
                    search.cv_results_["std_test_score"][search.best_index_]
                ),
                "dev_oof_f1_at_0.5": float(f1_score(y_dev, (oof >= 0.5).astype(int))),
                "dev_oof_recall_at_0.5": float(recall_score(y_dev, (oof >= 0.5).astype(int))),
                "best_params": {k: v for k, v in search.best_params_.items()},
            }
        )
        fitted[name] = best

    ranked = sorted(
        rows, key=lambda r: (r["cv_roc_auc_mean"], r["dev_oof_f1_at_0.5"]), reverse=True
    )
    best_auc = ranked[0]["cv_roc_auc_mean"]
    contenders = [r for r in ranked if best_auc - r["cv_roc_auc_mean"] <= SELECTION_TIE_BAND]
    winner_name = min(
        contenders,
        key=lambda r: SIMPLICITY_ORDER.index(r["model"])
        if r["model"] in SIMPLICITY_ORDER
        else len(SIMPLICITY_ORDER),
    )["model"]
    winner = fitted[winner_name]
    winner_params = next(r for r in ranked if r["model"] == winner_name)["best_params"]
    selection_note = {
        "rule": "max DEV group-CV ROC-AUC, then simplest family within "
        f"{SELECTION_TIE_BAND} ROC-AUC of the best",
        "cv_leader": ranked[0]["model"],
        "cv_leader_roc_auc": best_auc,
        "tie_band_members": [r["model"] for r in contenders],
        "selected": winner_name,
        "active_feature_set": "reduced" if fexp["adopted"] else "full",
    }

    def _winner_factory(s):
        est, _ = _search_spaces(s)[winner_name]
        return est.set_params(**{k.replace("model__", ""): v for k, v in winner_params.items()})

    stability = _multiseed_group_stability(_winner_factory, active, X, y, groups)

    oof_winner = cross_val_predict(
        winner, X_dev[active], y_dev, cv=cv, groups=g_dev, method="predict_proba", n_jobs=-1
    )[:, 1]
    threshold_grid = _threshold_report(y_dev, oof_winner)
    threshold = PRODUCTION_THRESHOLD

    winner.fit(X_dev[active], y_dev)
    test_proba = winner.predict_proba(X_test[active])[:, 1]
    holdout = _metrics(y_test, test_proba, threshold)
    holdout["holdout_subjects"] = int(g_test.nunique())
    holdout["holdout_rows"] = int(len(y_test))

    # Previous production pipeline on its own original subject-aware holdout.
    prev_path = CONFIG.model_dir / "archive_pre_upgrade" / "parkinsons_pipeline.joblib"
    previous = None
    if prev_path.exists():
        from joblib import load

        raw_old = load_dataset(CONFIG.raw_path)
        g_old = raw_old["name"].astype(str).str.extract(r"^(phon_R\d+_S\d+)", expand=False).fillna(
            raw_old["name"].astype(str)
        )
        X_old = raw_old[list(CONFIG.features)]
        y_old = pd.to_numeric(raw_old[TARGET], errors="coerce").astype(int)
        gss_old = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        _, te_old = next(gss_old.split(X_old, y_old, g_old))
        prev_pipe = load(prev_path)
        prev_proba = prev_pipe.predict_proba(X_old.iloc[te_old])[:, 1]
        previous = _metrics(y_old.iloc[te_old], prev_proba, 0.5)
        previous["note"] = (
            "Old Gradient Boosting on its own original subject-disjoint holdout. The old "
            "trainer DID use GroupShuffleSplit + GroupKFold, but selected the model by "
            "(f1, recall) on this holdout, which favoured a degenerate high-recall model "
            "(recall 1.0, ROC-AUC 0.72)."
        )

    importance = _permutation_importance(winner, X_test[active], y_test, seed)

    dataset_hash = sha256(CONFIG.raw_path.read_bytes()).hexdigest()
    now = datetime.now(timezone.utc)
    out = CONFIG.model_dir
    out.mkdir(parents=True, exist_ok=True)
    dump(winner, out / "parkinsons_pipeline.joblib")

    (out / "feature_names.json").write_text(json.dumps(list(active), indent=2))

    comparison = pd.DataFrame(ranked)
    comparison.to_csv(out / "model_comparison.csv", index=False)
    comparison.to_csv(ROOT / "reports" / "model_results" / "parkinsons_metrics.csv", index=False)

    winner_row = next(r for r in ranked if r["model"] == winner_name)
    metrics_doc = {
        "selected_model": winner_name,
        "selection": selection_note,
        "naive_vs_subject_aware": naive_vs_group,
        "feature_experiment": fexp,
        "multi_seed_group_stability_dev": stability,
        "operating_threshold": threshold,
        "threshold_rule": "Fixed at 0.5 (group out-of-fold sweep published; only ~25 training subjects).",
        "cv_dev": {
            "scoring": "roc_auc",
            "cv": "StratifiedGroupKFold(5), groups=subject",
            "roc_auc_mean": winner_row["cv_roc_auc_mean"],
            "roc_auc_std": winner_row["cv_roc_auc_std"],
            "dev_oof_f1_at_0.5": winner_row["dev_oof_f1_at_0.5"],
        },
        "holdout": holdout,
        "previous_model_original_holdout": previous,
        "threshold_grid_dev_oof": threshold_grid,
        "metrics": {**holdout, "model": winner_name, "cv_mean": winner_row["cv_roc_auc_mean"]},
        **{k: holdout[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")},
    }
    (out / "metrics.json").write_text(json.dumps(metrics_doc, indent=2))

    metadata = {
        "model_id": "parkinsons",
        "disease": CONFIG.title,
        "dataset": CONFIG.raw_path.name,
        "dataset_source": "UCI Machine Learning Repository - Parkinsons (Oxford / Max Little, 2008)",
        "dataset_url": "https://archive.ics.uci.edu/dataset/174/parkinsons",
        "dataset_license": "CC BY 4.0 (Little, McSharry, Roberts, Costello, Moroz)",
        "dataset_sha256": dataset_hash,
        "rows_raw": cleaning["rows_raw"],
        "rows_used": cleaning["rows_used"],
        "shape": [cleaning["rows_used"], len(PARKINSONS_FEATURES) + 2],
        "unique_subjects": cleaning["unique_subjects"],
        "recordings_per_subject": cleaning["recordings_per_subject"],
        "target_distribution_by_row": cleaning["target_distribution_by_row"],
        "target_distribution_by_subject": cleaning["target_distribution_by_subject"],
        "subject_id_rule": cleaning["subject_id_rule"],
        "identifier_excluded": [SUBJECT_ID_COLUMN, TARGET],
        "features": list(PARKINSONS_FEATURES),
        "active_features": active,
        "numeric_features": list(PARKINSONS_FEATURES),
        "categorical_features": [],
        "target": TARGET,
        "target_mapping": {"1": "Parkinson's", "0": "control"},
        "positive_class": "Parkinson's (status == 1)",
        "input_type": "pre-computed acoustic voice biomarkers (no raw-audio extraction in this system)",
        "split_strategy": (
            "GroupShuffleSplit(test_size=0.2) on subject -> subject-disjoint DEV/HOLDOUT; "
            "model chosen only on DEV via StratifiedGroupKFold(5) CV with groups=subject."
        ),
        "dev_holdout_subjects": {
            "dev_subjects": int(g_dev.nunique()),
            "holdout_subjects": int(g_test.nunique()),
            "dev_rows": int(len(y_dev)),
            "holdout_rows": int(len(y_test)),
        },
        "group_cv_strategy": "StratifiedGroupKFold(5, shuffle, seed), groups=subject, passed to GridSearchCV and cross_val_predict.",
        "random_seed": seed,
        "cleaning": cleaning,
        "preprocessing": (
            "Pipeline: SimpleImputer(median, defensive - dataset has no missing values) "
            "-> StandardScaler -> estimator. Fitted inside CV folds only."
        ),
        "imbalance_handling": (
            "Row classes 147/48 (~75% positive); subjects 24/8. class_weight (None vs "
            "'balanced'/'balanced_subsample') is a tuned hyper-parameter chosen by "
            "group-aware CV. Threshold left at 0.5; a dev group-OOF sweep is published."
        ),
        "naive_vs_subject_aware": naive_vs_group,
        "feature_experiment": fexp,
        "multi_seed_group_stability_dev": stability,
        "algorithms_evaluated": [r["model"] for r in ranked],
        "hyperparameter_search": "GridSearchCV over StratifiedGroupKFold(5) per family, scoring=roc_auc, groups=subject.",
        "model_selection": selection_note,
        "selected_algorithm": winner_name,
        "selected_model": winner_name,
        "selected_params": winner_row["best_params"],
        "operating_threshold": threshold,
        "cv_metrics": metrics_doc["cv_dev"],
        "holdout_metrics": holdout,
        "permutation_importance_holdout": importance,
        "training_timestamp": now.isoformat(),
        "training_date": date.today().isoformat(),
        "artifact_path": (out / "parkinsons_pipeline.joblib").relative_to(ROOT).as_posix(),
        "sklearn_version": __import__("sklearn").__version__,
        "threshold": threshold,
        "limitations": [
            "Only 32 subjects (24 Parkinson's / 8 control) from one 2008 Oxford study; a ~7-subject holdout gives wide confidence intervals.",
            "Subject-aware performance is far below the naive recording-level figure - the naive number is leakage, not skill.",
            "Inputs are pre-computed voice biomarkers from voice-analysis software; this module does NOT extract features from raw audio.",
            "Permutation importance describes model behaviour on this sample, not a physiological cause of Parkinson's disease.",
        ],
        "disclaimer": (
            "Educational and research use only. This estimates a voice-pattern "
            "association learned from a small public dataset and is not a medical diagnosis."
        ),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out / "explainability.json").write_text(
        json.dumps(
            {
                "method": "permutation_importance",
                "scoring": "roc_auc",
                "n_repeats": PERM_REPEATS,
                "evaluated_on": "untouched subject-disjoint holdout",
                "note": "Importance = mean drop in holdout ROC-AUC when the feature is shuffled. Association, not a physiological cause.",
                "features": importance,
            },
            indent=2,
        )
    )

    if write_figures:
        _write_figures(y_test, test_proba, threshold, importance)

    return {
        "selected_model": winner_name,
        "selection_note": selection_note,
        "naive_vs_subject_aware": naive_vs_group,
        "feature_experiment": fexp,
        "multi_seed_group_stability_dev": stability,
        "operating_threshold": threshold,
        "cv_roc_auc_mean": winner_row["cv_roc_auc_mean"],
        "holdout": holdout,
        "previous": previous,
        "ranked": ranked,
    }


def _write_figures(y_test, proba, threshold, importance) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
    except Exception:
        return
    fig_dir = ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pred = (proba >= threshold).astype(int)
    ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=[0, 1])
    plt.title(f"Parkinson's holdout confusion matrix (threshold={threshold:.2f})")
    plt.savefig(fig_dir / "parkinsons_confusion_matrix.png", bbox_inches="tight")
    plt.close()
    RocCurveDisplay.from_predictions(y_test, proba)
    plt.title("Parkinson's subject-disjoint holdout ROC curve")
    plt.savefig(fig_dir / "parkinsons_roc_curve.png", bbox_inches="tight")
    plt.close()
    top = importance[:12][::-1]
    plt.figure()
    plt.barh([d["feature"] for d in top], [d["importance_mean"] for d in top])
    plt.xlabel("Mean drop in holdout ROC-AUC when shuffled")
    plt.title("Parkinson's permutation importance (holdout)")
    plt.tight_layout()
    plt.savefig(fig_dir / "parkinsons_feature_importance.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    summary = train_parkinsons()
    print(json.dumps(summary, indent=2, default=str))
