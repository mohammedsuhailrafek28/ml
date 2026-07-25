# Medical AI Suite

Medical AI Suite is a university-level, multi-disease machine-learning demonstrator for liver, heart, diabetes, chronic kidney, and Parkinson's disease datasets. It estimates patterns in public data; it does **not** diagnose or provide medical advice.

## Next.js + FastAPI experience

The primary product surface now includes a calm Next.js App Router frontend under `frontend/` and a FastAPI backend under `src/api/`. Start the API with `.venv\\Scripts\\python.exe -m uvicorn src.api.main:app --reload`, then start the frontend with `cd frontend; npm install; npm run dev`. The legacy Streamlit app remains available as a demonstration interface.

API endpoints include `/api/v1/health`, `/api/v1/diseases`, `/api/v1/diseases/{disease}`, `/api/v1/predictions/{disease}`, and `/api/v1/reports/{disease}`. The backend remains the only inference layer and loads the persisted Joblib pipelines.

## Quick start
```bash
python -m venv .venv
.venv\\Scripts\\activate       # Windows
pip install -r requirements.txt
python -m src.training.train_all
streamlit run app.py
```

Exact reproduction commands:
```bash
cd C:\\Users\\HAIL\\Documents\\ML-PROJECT\\medical-ai-suite
pip install -r requirements.txt
python scripts/download_datasets.py
python -m src.training.train_all
pytest -v
streamlit run app.py
```
The downloader retrieves public teaching-data mirrors where available and prints manual instructions for kidney data. Verify schemas against `src/utils/config.py` before training; no medical records are synthesized by this project. Training uses stratified holdout evaluation, imputation/encoding inside a scikit-learn pipeline, cross-validation, and recall/F1-aware model selection. Results are written to `reports/model_results/` and pipelines to `models/`.

**Current status:** foundation and runnable workflow implemented; final model metrics depend on acquiring and validating the source CSVs. This repository intentionally does not claim clinical performance.

## Architecture
`src/data` validates inputs; `src/preprocessing` builds shared leakage-safe transformers; `src/training` compares and persists models; `src/prediction` serves them; `app.py` provides the Streamlit UI. The `reports/` directory contains cleaning, evaluation, ethics, and project documentation.

## Dataset sources and attribution
See `datasets/info/*.md` for source URLs, licenses, target/feature definitions, and acquisition notes. The architecture was informed by the four repositories named in the project brief; no repository branding or wholesale code was copied. Verify each upstream dataset's current license before redistribution.

## Limitations and ethics
Public datasets may be small, biased, and collected under different protocols. Metrics are not clinical validation. Outputs are educational, probabilistic model estimates and must not guide treatment. Consult a qualified healthcare professional.
