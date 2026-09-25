import json
from pathlib import Path
import pandas as pd
import pytest
from src.utils.config import DISEASES
from src.data.loader import load_dataset
from src.training.model_factory import candidates
from src.prediction.predictor import predict

ROOT=Path(__file__).resolve().parents[1]
def test_all_diseases_configured(): assert set(DISEASES)=={'liver','heart','diabetes','kidney','parkinsons'}
def test_all_dataset_paths_exist(): assert all(c.raw_path.exists() for c in DISEASES.values())
def test_target_not_in_features(): assert all(c.target not in c.features for c in DISEASES.values())
def test_model_factory_has_seven_entries(): assert len(candidates())==7
def test_missing_dataset_error(tmp_path):
    with pytest.raises(FileNotFoundError): load_dataset(tmp_path/'missing.csv')
def test_pipelines_load_and_predict():
    for k,c in DISEASES.items():
        d=pd.read_csv(c.raw_path); row=d.iloc[0].to_dict()
        row.pop(c.target,None); row.pop('name',None); row.pop('id',None)
        row={field:(None if pd.isna(value) else value) for field,value in row.items()}
        r=predict(k,row); assert r['prediction'] in (0,1); assert 0<=r['model_score']<=1
def test_metadata_required():
    required={'selected_model','random_seed'}
    for k in DISEASES: assert required <= set(json.loads((ROOT/'models'/k/'metadata.json').read_text()))
def test_pdf_signatures():
    for k in DISEASES: assert (ROOT/'reports/generated_reports'/f'{k}_sample_report.pdf').read_bytes()[:4]==b'%PDF'
def test_manifest_hashes_present():
    m=json.loads((ROOT/'reports/dataset_manifest.json').read_text()); assert len(m)==5 and all(x['sha256'] for x in m)
def test_parkinsons_identifier_excluded(): assert 'name' not in DISEASES['parkinsons'].features
def test_kidney_categorical_features(): assert 'htn' in DISEASES['kidney'].categorical
def test_diabetes_pregnancy_zero_preserved(): assert (pd.read_csv(DISEASES['diabetes'].raw_path).pregnancies==0).any()
