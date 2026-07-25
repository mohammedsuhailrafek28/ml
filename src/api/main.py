from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from src.api.schemas import PredictionRequest
from src.api.services.model_registry import registry
from src.reporting.pdf_report import create_report
from pathlib import Path
import tempfile

app=FastAPI(title='Medical AI Suite API',version='1.0')
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:3000'],allow_methods=['*'],allow_headers=['*'])
@app.get('/api/v1/health')
def health(): return {'status':'ok','loaded_diseases':registry.available(),'version':'1.0'}
@app.get('/api/v1/diseases')
def diseases(): return registry.catalog()
@app.get('/api/v1/diseases/{disease}')
def disease(disease:str):
    try:return registry.metadata(disease)
    except KeyError: raise HTTPException(404,'Unknown disease')
@app.post('/api/v1/predictions/{disease}')
def prediction(disease:str, req:PredictionRequest):
    try:return registry.predict(disease,req.measurements)
    except KeyError as e: raise HTTPException(400,str(e))
    except ValueError as e: raise HTTPException(422,str(e))
@app.post('/api/v1/reports/{disease}')
def report(disease:str, req:PredictionRequest):
    result=registry.predict(disease,req.measurements); p=Path(tempfile.gettempdir())/f'medical_ai_{disease}.pdf'; create_report(p,registry.metadata(disease)['name'],req.measurements,result); return FileResponse(p,media_type='application/pdf',filename=f'{disease}_report.pdf')
