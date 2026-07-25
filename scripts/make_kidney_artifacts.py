import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd,numpy as np,joblib,matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay,RocCurveDisplay
from sklearn.inspection import permutation_importance
from src.utils.config import DISEASES
c=DISEASES['kidney']; d=pd.read_csv(c.raw_path); 
for col in d.columns:
 if d[col].dtype==object: d[col]=d[col].astype(str).str.replace('\t','',regex=False).str.strip().replace({'?':np.nan,'nan':np.nan})
for col in c.numeric: d[col]=pd.to_numeric(d[col],errors='coerce')
d[c.target]=d[c.target].astype(str).str.strip(); d.to_csv('datasets/processed/kidney_cleaned.csv',index=False); X=d[list(c.features)]; y=d[c.target].map({'ckd':1,'notckd':0}); _,te=train_test_split(range(len(d)),test_size=.2,stratify=y,random_state=42); m=joblib.load(c.model_dir/'kidney_pipeline.joblib'); pred=m.predict(X.iloc[te]); proba=m.predict_proba(X.iloc[te])[:,1]; out=Path('reports/figures'); out.mkdir(parents=True,exist_ok=True)
ConfusionMatrixDisplay.from_predictions(y.iloc[te],pred); plt.savefig(out/'kidney_confusion_matrix.png'); plt.close(); RocCurveDisplay.from_predictions(y.iloc[te],proba); plt.savefig(out/'kidney_roc_curve.png'); plt.close(); y.value_counts().sort_index().plot.bar(title='Kidney target distribution'); plt.savefig(out/'kidney_target_distribution.png'); plt.close(); d.isna().sum().sort_values(ascending=False).head(15).plot.bar(title='Kidney missing values'); plt.savefig(out/'kidney_missing_values.png'); plt.close(); imp=permutation_importance(m,X.iloc[te],y.iloc[te],n_repeats=3,random_state=42,scoring='f1'); pd.Series(imp.importances_mean,index=c.features).sort_values().plot.barh(title='Kidney permutation importance'); plt.savefig(out/'kidney_feature_importance.png'); plt.close()
(c.model_dir/'feature_names.json').write_text(json.dumps(m.named_steps['preprocessor'].get_feature_names_out().tolist(),indent=2)); (c.model_dir/'metadata.json').write_text(json.dumps({'dataset':'UCI Chronic Kidney Disease','source':'https://archive.ics.uci.edu/dataset/336/chronic','shape':list(d.shape),'target_mapping':{'ckd':1,'notckd':0},'selected_model':json.loads((c.model_dir/'metrics.json').read_text())['selected_model'],'random_seed':42},indent=2)); pd.read_csv('reports/model_results/kidney_metrics.csv').to_csv(c.model_dir/'model_comparison.csv',index=False)
