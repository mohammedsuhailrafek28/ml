from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from src.utils.config import DISCLAIMER
import json
def create_report(path, disease, values, result):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    c=canvas.Canvas(str(path), pagesize=letter); y=760
    c.setFont('Helvetica-Bold',16); c.drawString(50,y,'Medical AI Suite — Educational report'); y-=35
    c.setFont('Helvetica',10)
    key={'Liver disease':'liver','Heart disease':'heart','Diabetes':'diabetes','Kidney disease':'kidney','Chronic kidney disease':'kidney',"Parkinsons disease":'parkinsons',"Parkinson's disease":'parkinsons'}.get(disease)
    selected='Unavailable'
    try: selected=json.loads((Path(__file__).resolve().parents[2]/'models'/key/'metrics.json').read_text())['selected_model']
    except Exception: pass
    prob=result.get('risk_probability', result.get('probability'))
    thr=result.get('threshold', 0.5)
    factors=result.get('topFactors') or []
    factor_line=('Top model factors: '+', '.join(f.get('feature','') for f in factors)) if factors \
        else 'Global model influences: See the model comparison and feature-importance artifacts.'
    lines=["Medical AI Suite — Disease prediction report",f'Module: {disease}',f"Prediction: {result.get('prediction')}",f"Risk probability: {prob}",f'Selected model: {selected}',f'Threshold: {thr}',f'Generated: {datetime.now():%Y-%m-%d %H:%M}','','Input measurements:']+[f'{k}: {v if v is not None else "Not provided"}' for k,v in values.items()]+['',factor_line,'',DISCLAIMER,'This output is not a diagnosis and must not replace evaluation by a qualified healthcare professional.']
    for line in lines:
        c.drawString(50,y,line[:110]); y-=18
    c.save()
