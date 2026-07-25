import json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd, joblib, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from src.utils.config import DISEASES
c=DISEASES['liver']; d=pd.read_csv(c.raw_path); X=d[list(c.features)]; y=d[c.target].map({1:1,2:0})
_,te=train_test_split(range(len(d)),test_size=.2,stratify=y,random_state=42); m=joblib.load(c.model_dir/'liver_pipeline.joblib'); pred=m.predict(X.iloc[te]); proba=m.predict_proba(X.iloc[te])[:,1]
out=Path('reports/figures'); out.mkdir(parents=True,exist_ok=True)
ConfusionMatrixDisplay.from_predictions(y.iloc[te],pred); plt.savefig(out/'liver_confusion_matrix.png'); plt.close()
RocCurveDisplay.from_predictions(y.iloc[te],proba); plt.savefig(out/'liver_roc_curve.png'); plt.close()
y.value_counts().sort_index().plot.bar(title='Liver target distribution'); plt.savefig(out/'liver_target_distribution.png'); plt.close()
names=m.named_steps['preprocessor'].get_feature_names_out().tolist(); (c.model_dir/'feature_names.json').write_text(json.dumps(names,indent=2)); (c.model_dir/'metadata.json').write_text(json.dumps({'dataset':'liver.csv','shape':list(d.shape),'target_mapping':{'1':1,'2':0},'selected_algorithm':'random_forest','random_seed':42},indent=2)); pd.read_csv('reports/model_results/liver_metrics.csv').to_csv(c.model_dir/'model_comparison.csv',index=False)
