"""Leakage-safe training pipeline for the Liver disease module (UCI ILPD).

Run: python -m src.training.train_liver

Design (see PHASE 5 of the upgrade brief):

    raw ILPD
      -> drop rows with missing target
      -> drop exact duplicate rows      (leakage fix: identical rows must not span the split)
      -> map target {1 -> 1 disease, 2 -> 0 no disease}
      -> stratified 80/20 split into DEV / TEST   (TEST is an untouched holdout)
      -> for each candidate family: Pipeline(preprocessor, model) tuned with
         GridSearchCV over StratifiedKFold(5), scoring = ROC-AUC, on DEV only
      -> winner selected by mean CV ROC-AUC on DEV, simplest family within a
         small tie band  (never the TEST set)
      -> operating threshold fixed at 0.5 (a dev-only sweep is published, not tuned)
      -> winner refit on the whole DEV set
      -> evaluated exactly once on TEST
      -> persisted pipeline + metadata + metrics + permutation importance

Everything is seeded with src.utils.config.RANDOM_STATE for reproducibility.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from hashlib import sha256

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data.loader import load_dataset
from src.preprocessing.common import build_preprocessor
from src.utils.config import DISEASES, RANDOM_STATE, ROOT, TEST_SIZE

CONFIG = DISEASES["liver"]
PERM_REPEATS = 10

# When two families are within this much CV ROC-AUC of the best, prefer the one
# earlier in this list (simpler / more interpretable / cheaper to serve).
SELECTION_TIE_BAND = 0.01
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
    """Candidate estimator families with modest, defensible hyper-parameter grids."""
    # NOTE on class_weight: in ILPD the disease class is the *majority* (~71%),
    # so class_weight="balanced" up-weights the healthy minority and suppresses
    # sensitivity. We let the CV search choose None vs "balanced" per family
    # rather than hard-code either.
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=5000, random_state=seed),
            {"model__C": [0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]},
        ),
        "decision_tree": (
            DecisionTreeClassifier(random_state=seed),
            {
                "model__max_depth": [3, 5, 7],
                "model__min_samples_leaf": [3, 5],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=seed, n_jobs=-1),
            {
                "model__n_estimators": [300, 500],
                "model__max_depth": [None, 6, 10],
                "model__min_samples_leaf": [1, 3],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=seed),
            {
                "model__n_estimators": [150, 300],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
        "svm": (
            SVC(kernel="rbf", probability=True, random_state=seed),
            {
                "model__C": [1.0, 10.0],
                "model__gamma": ["scale"],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "knn": (
            KNeighborsClassifier(),
            {"model__n_neighbors": [7, 11, 15], "model__weights": ["uniform", "distance"]},
        ),
        "gaussian_nb": (GaussianNB(), {}),
    }


def _load_frame() -> tuple[pd.DataFrame, dict]:
    raw = load_dataset(CONFIG.raw_path)
    rows_raw = len(raw)
    frame = raw.dropna(subset=[CONFIG.target])
    dropped_target = rows_raw - len(frame)
    dup_mask = frame.duplicated(keep="first")
    n_dupes = int(dup_mask.sum())
    frame = frame.loc[~dup_mask].reset_index(drop=True)
    cleaning = {
        "rows_raw": rows_raw,
        "rows_dropped_missing_target": int(dropped_target),
        "exact_duplicate_rows_removed": n_dupes,
        "rows_used": len(frame),
        "missing_value_handling": (
            "A/G Ratio missing values imputed with the training-fold median inside the "
            "pipeline; Gender missing values imputed with the training-fold mode."
        ),
        "duplicate_policy": (
            "Exact duplicate rows removed before splitting so identical records cannot "
            "appear in both train and test (ILPD has no patient id)."
        ),
    }
    return frame, cleaning


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


PRODUCTION_THRESHOLD = 0.5


def _threshold_report(y_true, proba) -> list[dict]:
    """Precision/recall/specificity/F1/Youden's J across a threshold sweep.

    This is published for transparency only. We deliberately do NOT tune the
    operating threshold on ILPD: with 583 records and models that separate the
    classes weakly, the dev out-of-fold precision-recall curve is nearly flat
    and every optimum (F1, Youden's J, recall floor) is unstable across seeds.
    Production uses the conventional PRODUCTION_THRESHOLD (0.5), which also keeps
    the old-vs-new comparison on equal footing.
    """
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


def train_liver(write_figures: bool = True) -> dict:
    seed = RANDOM_STATE
    frame, cleaning = _load_frame()

    X = frame[list(CONFIG.features)].copy()
    y = frame[CONFIG.target].astype(str).str.strip().map({"1": 1, "2": 0})
    if y.isna().any() or y.nunique() != 2:
        raise ValueError("Liver: target did not map to a clean binary {0,1}")
    y = y.astype(int)

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows: list[dict] = []
    fitted: dict[str, Pipeline] = {}
    for name, (estimator, grid) in _search_spaces(seed).items():
        pipe = Pipeline(
            [("preprocessor", build_preprocessor(CONFIG)), ("model", estimator)]
        )
        search = GridSearchCV(
            pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True
        )
        search.fit(X_dev, y_dev)
        best = search.best_estimator_
        oof = cross_val_predict(
            best, X_dev, y_dev, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]
        rows.append(
            {
                "model": name,
                "cv_roc_auc_mean": float(search.best_score_),
                "cv_roc_auc_std": float(
                    search.cv_results_["std_test_score"][search.best_index_]
                ),
                "dev_oof_f1_at_0.5": float(f1_score(y_dev, (oof >= 0.5).astype(int))),
                "dev_oof_recall_at_0.5": float(
                    recall_score(y_dev, (oof >= 0.5).astype(int))
                ),
                "best_params": {k: v for k, v in search.best_params_.items()},
            }
        )
        fitted[name] = best

    ranked = sorted(
        rows, key=lambda r: (r["cv_roc_auc_mean"], r["dev_oof_f1_at_0.5"]), reverse=True
    )
    # Model selection uses DEV cross-validation only (never the holdout). The
    # top ILPD families sit within CV noise of each other, so among everything
    # within SELECTION_TIE_BAND of the best CV ROC-AUC we take the simplest /
    # most interpretable family rather than chase a fraction of a point.
    best_auc = ranked[0]["cv_roc_auc_mean"]
    contenders = [r for r in ranked if best_auc - r["cv_roc_auc_mean"] <= SELECTION_TIE_BAND]
    winner_name = min(
        contenders,
        key=lambda r: SIMPLICITY_ORDER.index(r["model"])
        if r["model"] in SIMPLICITY_ORDER
        else len(SIMPLICITY_ORDER),
    )["model"]
    winner = fitted[winner_name]
    selection_note = {
        "rule": "max DEV CV ROC-AUC, then simplest family within "
        f"{SELECTION_TIE_BAND} ROC-AUC of the best",
        "cv_leader": ranked[0]["model"],
        "cv_leader_roc_auc": best_auc,
        "tie_band_members": [r["model"] for r in contenders],
        "selected": winner_name,
    }

    # Threshold sweep on DEV out-of-fold probabilities (published, not tuned).
    oof_winner = cross_val_predict(
        winner, X_dev, y_dev, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    threshold_grid = _threshold_report(y_dev, oof_winner)
    threshold = PRODUCTION_THRESHOLD

    # Refit winner on the whole DEV set, then touch the holdout exactly once.
    winner.fit(X_dev, y_dev)
    test_proba = winner.predict_proba(X_test)[:, 1]
    holdout = _metrics(y_test, test_proba, threshold)

    # Previous production pipeline, evaluated on *its own* original holdout.
    # The old trainer split the non-deduplicated 583-row frame with
    # train_test_split(test_size=0.2, stratify=y, random_state=42) and had no
    # validation partition. Re-scoring it on the new dedup split would leak
    # (old training rows reappear in the new test set via the 13 duplicates),
    # so we reconstruct the old split exactly instead.
    prev_path = CONFIG.model_dir / "archive_pre_upgrade" / "liver_pipeline.joblib"
    previous = None
    if prev_path.exists():
        from joblib import load

        raw = load_dataset(CONFIG.raw_path).dropna(subset=[CONFIG.target])
        X_old = raw[list(CONFIG.features)]
        y_old = raw[CONFIG.target].astype(str).str.strip().map({"1": 1, "2": 0}).astype(int)
        _, X_old_te, _, y_old_te = train_test_split(
            X_old, y_old, test_size=TEST_SIZE, stratify=y_old, random_state=seed
        )
        prev_pipe = load(prev_path)
        prev_proba = prev_pipe.predict_proba(X_old_te)[:, 1]
        previous = _metrics(y_old_te, prev_proba, 0.5)
        previous["note"] = (
            "Old Random Forest on its own original non-deduplicated stratified "
            "holdout (no separate validation set; model was also selected on this "
            "same split, so these numbers are optimistic)."
        )

    importance = _permutation_importance(winner, X_test, y_test, seed)

    dataset_hash = sha256(CONFIG.raw_path.read_bytes()).hexdigest()
    now = datetime.now(timezone.utc)
    out = CONFIG.model_dir
    out.mkdir(parents=True, exist_ok=True)

    dump(winner, out / "liver_pipeline.joblib")

    feature_names = (
        winner.named_steps["preprocessor"].get_feature_names_out().tolist()
    )
    (out / "feature_names.json").write_text(json.dumps(feature_names, indent=2))

    comparison = pd.DataFrame(ranked)
    comparison.to_csv(out / "model_comparison.csv", index=False)
    comparison.to_csv(
        ROOT / "reports" / "model_results" / "liver_metrics.csv", index=False
    )

    winner_row = next(r for r in ranked if r["model"] == winner_name)
    metrics_doc = {
        "selected_model": winner_name,
        "selection": selection_note,
        "operating_threshold": threshold,
        "threshold_rule": "Fixed at 0.5 (not tuned on ILPD; see threshold_grid_dev_oof).",
        "cv_dev": {
            "scoring": "roc_auc",
            "roc_auc_mean": winner_row["cv_roc_auc_mean"],
            "roc_auc_std": winner_row["cv_roc_auc_std"],
            "dev_oof_f1_at_0.5": winner_row["dev_oof_f1_at_0.5"],
        },
        "holdout": holdout,
        "previous_model_original_holdout": previous,
        "threshold_grid_dev_oof": threshold_grid,
        # Back-compat: legacy consumers read top-level metric keys + "metrics".
        "metrics": {**holdout, "model": winner_name,
                    "cv_mean": winner_row["cv_roc_auc_mean"]},
        **{k: holdout[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")},
    }
    (out / "metrics.json").write_text(json.dumps(metrics_doc, indent=2))

    metadata = {
        "model_id": "liver",
        "disease": CONFIG.title,
        "dataset": CONFIG.raw_path.name,
        "dataset_source": "UCI Machine Learning Repository - Indian Liver Patient Dataset (ILPD)",
        "dataset_url": "https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset",
        "dataset_sha256": dataset_hash,
        "rows_raw": cleaning["rows_raw"],
        "rows_used": cleaning["rows_used"],
        "shape": [cleaning["rows_used"], len(CONFIG.features) + 1],
        "features": list(CONFIG.features),
        "categorical_features": list(CONFIG.categorical),
        "numeric_features": list(CONFIG.numeric),
        "target": CONFIG.target,
        "target_mapping": {"1": 1, "2": 0},
        "positive_class": "liver-disease pattern (ILPD Selector == 1)",
        "class_balance_used": {
            "positive": int((y == 1).sum()),
            "negative": int((y == 0).sum()),
        },
        "split_strategy": "Stratified 80/20 hold-out; model chosen only on the 80% dev partition via CV.",
        "random_seed": seed,
        "cleaning": cleaning,
        "preprocessing": (
            "ColumnTransformer: numeric -> SimpleImputer(median) + StandardScaler; "
            "categorical(Gender) -> SimpleImputer(most_frequent) + OneHotEncoder. "
            "Fitted inside the pipeline on training folds only."
        ),
        "imbalance_handling": (
            "class_weight (None vs 'balanced') is a tuned hyper-parameter chosen by "
            "CV per family. The operating threshold is left at 0.5; a dev-only "
            "precision/recall/Youden sweep is published but not used for tuning "
            "because it is unstable on 583 rows. No synthetic oversampling (SMOTE)."
        ),
        "algorithms_evaluated": [r["model"] for r in ranked],
        "hyperparameter_search": "GridSearchCV over StratifiedKFold(5, shuffle, seed) per family, scoring=roc_auc.",
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
        "artifact_path": (out / "liver_pipeline.joblib").relative_to(ROOT).as_posix(),
        "sklearn_version": __import__("sklearn").__version__,
        "threshold": threshold,
        "limitations": [
            "ILPD is a single-region (Andhra Pradesh, India) dataset of 583 records; it under-represents women and non-Indian populations.",
            "Labels indicate prior clinical grouping, not a specific liver disease or severity.",
            "Permutation importance describes model behaviour, not physiological causation.",
        ],
        "disclaimer": (
            "Educational and research use only. This estimates a risk pattern learned "
            "from a public dataset and is not a medical diagnosis."
        ),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out / "explainability.json").write_text(
        json.dumps(
            {
                "method": "permutation_importance",
                "scoring": "roc_auc",
                "n_repeats": PERM_REPEATS,
                "evaluated_on": "untouched holdout test set",
                "note": "Importance = mean drop in holdout ROC-AUC when the feature is shuffled. Not a causal claim.",
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
    except Exception:  # matplotlib is optional for training
        return
    fig_dir = ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pred = (proba >= threshold).astype(int)
    ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=[0, 1])
    plt.title(f"Liver holdout confusion matrix (threshold={threshold:.2f})")
    plt.savefig(fig_dir / "liver_confusion_matrix.png", bbox_inches="tight")
    plt.close()
    RocCurveDisplay.from_predictions(y_test, proba)
    plt.title("Liver holdout ROC curve")
    plt.savefig(fig_dir / "liver_roc_curve.png", bbox_inches="tight")
    plt.close()
    top = importance[:10][::-1]
    plt.figure()
    plt.barh([d["feature"] for d in top], [d["importance_mean"] for d in top])
    plt.xlabel("Mean drop in holdout ROC-AUC when shuffled")
    plt.title("Liver permutation importance (holdout)")
    plt.tight_layout()
    plt.savefig(fig_dir / "liver_feature_importance.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    summary = train_liver()
    print(json.dumps(summary, indent=2, default=str))
