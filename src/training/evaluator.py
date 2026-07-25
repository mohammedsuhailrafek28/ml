"""Evaluation helpers."""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
def evaluate(model, X, y):
    pred = model.predict(X); proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    return {"accuracy": accuracy_score(y,pred), "precision": precision_score(y,pred,zero_division=0), "recall": recall_score(y,pred,zero_division=0), "f1": f1_score(y,pred,zero_division=0), "roc_auc": roc_auc_score(y,proba) if proba is not None and len(np.unique(y))==2 else None}
