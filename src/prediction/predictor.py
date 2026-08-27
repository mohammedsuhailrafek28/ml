"""Load persisted pipelines and return cautious prediction objects."""
import json
from joblib import load
import pandas as pd
from src.utils.config import DISEASES
def predict(disease, values):
    c=DISEASES[disease]
    if disease=="heart":
        # ca/thal etc. must reach the one-hot encoder as the exact string tokens
        # the model was trained on (see src/preprocessing/heart_schema.py).
        from src.preprocessing.heart_schema import normalize_measurements
        values=normalize_measurements(values)
    if disease=="kidney":
        # coerce numerics to float and clean categorical tokens (stray tabs / '?'),
        # matching src/training/train_kidney.py.
        from src.preprocessing.kidney_schema import normalize_measurements as _nk
        values=_nk(values)
    cols=list(c.features)
    meta_path=c.model_dir/"metadata.json"
    if meta_path.exists():
        active=json.loads(meta_path.read_text()).get("active_features")
        # parkinsons is a bare imputer+scaler+model fitted on a feature subset and
        # checks feature names on transform, so it must get exactly those columns.
        if active and disease=="parkinsons":
            cols=[f for f in active if f in c.features]
    model=load(c.model_dir/f"{disease}_pipeline.joblib"); frame=pd.DataFrame([{k:values[k] for k in cols}]); p=int(model.predict(frame)[0]); prob=float(model.predict_proba(frame)[0,1]) if hasattr(model,"predict_proba") else None
    return {"disease":c.title,"prediction":p,"risk_probability":prob,"interpretation":"Model-estimated pattern only; this is not a diagnosis."}
