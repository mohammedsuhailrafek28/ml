import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd,numpy as np,joblib,matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay,RocCurveDisplay
from sklearn.inspection import permutation_importance
from src.utils.config import DISEASES
c=DISEASES['diabetes']; d=pd.read_csv(c.raw_path); X=d[list(c.features)].replace({0:np.nan}); X['pregnancies']=d['pregnancies']; y=d['target']; _,te=train_test_split(range(len(d)),test_size=.2,stratify=y,random_state=42); m=joblib.load(c.model_dir/'diabetes_pipeline.joblib'); pred=m.predict(X.iloc[te]); proba=m.predict_proba(X.iloc[te])[:,1]; out=Path('reports/figures'); out.mkdir(parents=True,exist_ok=True)
ConfusionMatrixDisplay.from_predictions(y.iloc[te],pred); plt.savefig(out/'diabetes_confusion_matrix.png'); plt.close(); RocCurveDisplay.from_predictions(y.iloc[te],proba); plt.savefig(out/'diabetes_roc_curve.png'); plt.close(); y.value_counts().sort_index().plot.bar(title='Diabetes target distribution'); plt.savefig(out/'diabetes_target_distribution.png'); plt.close(); imp=permutation_importance(m,X.iloc[te],y.iloc[te],n_repeats=5,random_state=42,scoring='f1'); pd.Series(imp.importances_mean,index=c.features).sort_values().plot.barh(title='Diabetes permutation importance'); plt.savefig(out/'diabetes_feature_importance.png'); plt.close()
(c.model_dir/'feature_names.json').write_text(json.dumps(c.features,indent=2)); (c.model_dir/'metadata.json').write_text(json.dumps({'dataset':'Pima Indians Diabetes Database','source':'UCI/OpenML mirror','shape':list(d.shape),'target':'Outcome: 1 diabetes, 0 no diabetes','zero_placeholder_columns':list(c.zero_as_missing),'selected_model':json.loads((c.model_dir/'metrics.json').read_text())['selected_model'],'random_seed':42},indent=2)); pd.read_csv('reports/model_results/diabetes_metrics.csv').to_csv(c.model_dir/'model_comparison.csv',index=False)
