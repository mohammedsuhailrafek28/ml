import json
from pathlib import Path
import pandas as pd
from joblib import load
from src.utils.config import DISEASES
class Registry:
 def __init__(self): self.models={}
 def _config(self,k):
  if k not in DISEASES: raise KeyError('Unknown disease')
  return DISEASES[k]
 def _model(self,k):
  if k not in self.models: self.models[k]=load(self._config(k).model_dir/f'{k}_pipeline.joblib')
  return self.models[k]
 def available(self): return [k for k,c in DISEASES.items() if (c.model_dir/f'{k}_pipeline.joblib').exists()]
 def metadata(self,k):
  c=self._config(k); m=json.loads((c.model_dir/'metrics.json').read_text()); return {'slug':k,'name':c.title,'dataset':c.raw_path.name,'selected_model':m['selected_model'],'input_features':list(c.features),'target':c.target}
 def catalog(self): return [self.metadata(k) for k in self.available()]
 def predict(self,k,values):
  c=self._config(k); unknown=set(values)-set(c.features)
  if unknown: raise ValueError(f'Unexpected fields: {sorted(unknown)}')
  if k=='diabetes':
   for field in ('pregnancies','age'):
    val=values.get(field)
    if val is None or not isinstance(val,(int,float)) or int(val)!=val: raise ValueError(f'{field} must be a whole number')
   if values['pregnancies']<0 or values['age']<0: raise ValueError('pregnancies and age cannot be negative')
   for field in ('bmi','diabetes_pedigree'):
    if values.get(field) is not None and values[field]<0: raise ValueError(f'{field} cannot be negative')
  frame=pd.DataFrame([{f:values.get(f) for f in c.features}]); model=self._model(k); pred=int(model.predict(frame)[0]); prob=float(model.predict_proba(frame)[0,1]) if hasattr(model,'predict_proba') else None
  return {'disease':k,'prediction':pred,'label':'Higher-risk pattern detected' if pred else 'Lower-risk pattern detected','probability':prob,'threshold':0.5,'selectedModel':self.metadata(k)['selected_model'],'limitations':['Educational model only','Not clinically validated'],'disclaimer':'This prediction is not a medical diagnosis.'}
registry=Registry()
