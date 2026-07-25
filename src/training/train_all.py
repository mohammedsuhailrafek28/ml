"""Train and persist all disease pipelines. Run: python -m src.training.train_all"""
from pathlib import Path
import json, pandas as pd, argparse, numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GroupShuffleSplit, GroupKFold
import re
from sklearn.pipeline import Pipeline
from joblib import dump
from src.utils.config import DISEASES, RANDOM_STATE, TEST_SIZE
from src.data.loader import load_dataset
from src.preprocessing.common import build_preprocessor
from src.training.model_factory import candidates
from src.training.evaluator import evaluate

def train_one(config):
    df=load_dataset(config.raw_path); df=df.dropna(subset=[config.target])
    if config.key == 'kidney':
        for col in df.columns:
            if df[col].dtype == object: df[col]=df[col].astype(str).str.replace('\\t','',regex=False).str.strip().replace({'?':np.nan,'nan':np.nan})
        for col in config.numeric: df[col]=pd.to_numeric(df[col],errors='coerce')
    if config.key == 'diabetes':
        for col in config.zero_as_missing: df[col]=df[col].replace(0, np.nan)
    X=df[list(config.features)]
    y=df[config.target].astype(str).str.strip()
    # ILPD convention: 1 = liver disease, 2 = no disease.
    if config.key == "liver": y=y.map({"1":1,"2":0})
    else:
        if config.key == 'kidney': y=y.str.lower().str.strip().map({'ckd':1,'notckd':0})
        else: y=pd.to_numeric(y, errors="coerce")
        if config.key == "heart": y=(y>0).astype(int)
    if y.isna().any() or y.nunique()!=2: raise ValueError(f"{config.title}: target encoding is not binary")
    y=y.astype(int)
    if config.key == 'parkinsons':
        groups=df['name'].astype(str).str.extract(r'^(phon_R\d+_S\d+)', expand=False).fillna(df['name'].astype(str))
        splitter=GroupShuffleSplit(n_splits=1,test_size=TEST_SIZE,random_state=RANDOM_STATE)
        tr_idx,te_idx=next(splitter.split(X,y,groups)); Xtr,Xte=X.iloc[tr_idx],X.iloc[te_idx]; ytr,yte=y.iloc[tr_idx],y.iloc[te_idx]
        cv=GroupKFold(n_splits=5); cv_groups=groups.iloc[tr_idx]
    else:
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=TEST_SIZE,stratify=y,random_state=RANDOM_STATE); cv=5; cv_groups=None
    rows=[]; best=None
    for name, estimator in candidates(RANDOM_STATE).items():
        pipe=Pipeline([("preprocessor",build_preprocessor(config)),("model",estimator)]); pipe.fit(Xtr,ytr)
        m=evaluate(pipe,Xte,yte)
        scores=cross_val_score(pipe,Xtr,ytr,cv=cv,groups=cv_groups,scoring="f1")
        m.update({"model":name,"cv_mean":scores.mean(),"cv_std":scores.std()}); rows.append(m)
        if best is None or (m["f1"],m["recall"])>(best[0]["f1"],best[0]["recall"]): best=(m,pipe)
    out=config.model_dir; out.mkdir(parents=True,exist_ok=True); dump(best[1],out/f"{config.key}_pipeline.joblib")
    pd.DataFrame(rows).to_csv(Path(__file__).resolve().parents[2]/"reports/model_results"/f"{config.key}_metrics.csv",index=False)
    (out/"metrics.json").write_text(json.dumps({"selected_model":best[0]["model"],"metrics":best[0]},indent=2))

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--disease', choices=list(DISEASES)); args=parser.parse_args()
    configs=[DISEASES[args.disease]] if args.disease else DISEASES.values()
    summary=[]
    for c in configs:
        try: train_one(c); print(f"trained {c.key}")
        except (FileNotFoundError, KeyError, ValueError) as e: print(f"SKIP {c.key}: {e}")
if __name__=="__main__": main()
