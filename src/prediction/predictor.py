"""Load persisted pipelines and return cautious prediction objects."""
from joblib import load
import pandas as pd
from src.utils.config import DISEASES
def predict(disease, values):
    c=DISEASES[disease]; model=load(c.model_dir/f"{disease}_pipeline.joblib"); frame=pd.DataFrame([{k:values[k] for k in c.features}]); p=int(model.predict(frame)[0]); prob=float(model.predict_proba(frame)[0,1]) if hasattr(model,"predict_proba") else None
    return {"disease":c.title,"prediction":p,"risk_probability":prob,"interpretation":"Model-estimated pattern only; this is not a diagnosis."}
