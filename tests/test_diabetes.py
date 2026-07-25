import pandas as pd, numpy as np
from src.utils.config import DISEASES
from src.data.loader import load_dataset
from src.training.model_factory import candidates
from src.prediction.predictor import predict
def test_diabetes_schema_and_target():
    d=load_dataset(DISEASES['diabetes'].raw_path); assert set(DISEASES['diabetes'].features).issubset(d.columns); assert set(d.target.unique())=={0,1}
def test_zero_placeholder_policy_preserves_pregnancies():
    d=load_dataset(DISEASES['diabetes'].raw_path); assert (d['pregnancies']==0).any(); assert (d['glucose']==0).any()
def test_all_requested_models_exist():
    assert {'logistic_regression','decision_tree','random_forest','svm','knn','gaussian_nb','gradient_boosting'} <= set(candidates())
def test_diabetes_saved_prediction_probability():
    d=load_dataset(DISEASES['diabetes'].raw_path); r=predict('diabetes',d.iloc[0].drop('target').to_dict()); assert r['prediction'] in (0,1); assert 0 <= r['risk_probability'] <= 1
