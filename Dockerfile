# Backend (FastAPI) image for the Medical AI Suite API.
# Build from the repo root:  docker build -t medical-ai-api .
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install runtime deps first for layer caching.
COPY requirements.lock .
RUN pip install --upgrade pip && pip install --require-hashes -r requirements.lock

# Application code + persisted model artifacts (the only inference layer).
COPY src/ ./src/
COPY models/ ./models/

# Non-root runtime user.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

ENV APP_ENV=production \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    LOG_LEVEL=INFO
EXPOSE 8000

# Readiness-aware healthcheck (uses stdlib, no curl needed).
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health').status==200 else 1)"

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
