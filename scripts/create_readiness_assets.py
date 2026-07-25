import json
from pathlib import Path
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; nb=ROOT/'notebooks'; nb.mkdir(exist_ok=True)
for k,title in [('liver','Liver Disease'),('heart','Heart Disease'),('diabetes','Diabetes'),('kidney','Chronic Kidney Disease'),('parkinsons',"Parkinson's Disease")]:
 cells=[{'cell_type':'markdown','metadata':{},'source':[f'# {title} analysis\n','Educational analysis using the validated public dataset.\n','Medical disclaimer: this is not a diagnosis.']},{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':[f"import pandas as pd\nfrom pathlib import Path\n\ndf=pd.read_csv(Path('../datasets/raw/{k}.csv'))\nprint(df.shape)\n"]},{'cell_type':'markdown','metadata':{},'source':['Model comparison results are loaded from `reports/model_results`. Interpret metrics cautiously; they are not clinical evidence.']}]
 (nb/f'{k}_analysis.ipynb').write_text(json.dumps({'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5},indent=2))
assets=ROOT/'presentation_assets'; assets.mkdir(exist_ok=True)
def card(path,title,lines):
 fig,ax=plt.subplots(figsize=(12,6)); ax.axis('off'); ax.text(.05,.88,title,fontsize=24,weight='bold'); ax.text(.06,.75,'\n'.join(lines),fontsize=16,va='top'); fig.savefig(path,dpi=160,bbox_inches='tight'); plt.close(fig)
card(assets/'system_architecture.png','Medical AI Suite architecture',['Five UCI datasets → validation → cleaning','Leakage-safe preprocessing → seven-model comparison','Persisted Joblib pipelines → Streamlit → PDF reports'])
card(assets/'project_workflow.png','Prediction workflow',['Choose disease → enter measurements → validate','Apply fitted pipeline → predict class/probability','Show global influences → generate educational PDF'])
card(assets/'dataset_summary.png','Dataset summary',['Liver 583 | Heart 303 | Diabetes 768 | Kidney 400 | Parkinsons 195','Targets are binary disease/risk indicators with documented mappings'])
card(assets/'model_comparison.png','Model comparison',['Models: Logistic Regression, Decision Tree, Random Forest','Gradient Boosting, SVM, KNN, Gaussian Naive Bayes','Selection prioritizes recall, F1, ROC-AUC and stability'])
card(assets/'final_results.png','Final results',['Five persisted pipelines and measured evaluation artifacts','Parkinsons uses subject-grouped evaluation','All outputs are educational—not clinical evidence'])
card(assets/'parkinsons_group_split.png','Parkinsons leakage control',['Recording name → derived subject ID','GroupShuffleSplit → separated train/test subjects','Zero subject overlap verified'])
