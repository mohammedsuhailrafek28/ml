# Build from repository root. The multi-platform manifest digest was verified
# against the registry with Docker Buildx before this release-candidate build.
FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3 AS wheels
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /build
COPY requirements.lock ./requirements.lock
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip wheel --require-hashes --wheel-dir /wheelhouse -r requirements.lock

FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3 AS runtime
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    APP_ENV=production API_HOST=0.0.0.0 API_PORT=8000 LOG_LEVEL=INFO \
    MODEL_ROOT=/app/models HOME=/tmp
ARG APP_VERSION
ARG APP_SOURCE_REVISION=unknown
ARG OCI_CREATED=unknown
ENV APP_VERSION=$APP_VERSION APP_SOURCE_REVISION=$APP_SOURCE_REVISION
LABEL org.opencontainers.image.title="Medical AI Suite API" \
      org.opencontainers.image.version="$APP_VERSION" \
      org.opencontainers.image.source="https://github.com/mohammedsuhailrafek28/ml" \
      org.opencontainers.image.revision="$APP_SOURCE_REVISION" \
      org.opencontainers.image.created="$OCI_CREATED" \
      org.opencontainers.image.description="Educational/research inference API; not clinically validated"
RUN apt-get update && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home app
WORKDIR /app
COPY --from=wheels /wheelhouse /wheelhouse
COPY requirements.lock ./requirements.lock
RUN python -m pip install --no-index --find-links=/wheelhouse --require-hashes -r requirements.lock \
    && python -m pip uninstall -y pip setuptools wheel \
    && rm -rf /wheelhouse requirements.lock /root/.cache/pip /tmp/pip-*
COPY src/api/ ./src/api/
COPY src/preprocessing/ ./src/preprocessing/
COPY src/reporting/ ./src/reporting/
COPY src/utils/ ./src/utils/
COPY src/model_release.py src/application_release.py ./src/
COPY release/application.json ./release/application.json
COPY models/release_manifest.json ./models/release_manifest.json
COPY models/liver/liver_pipeline.joblib models/liver/metadata.json models/liver/metrics.json models/liver/feature_names.json ./models/liver/
COPY models/diabetes/diabetes_pipeline.joblib models/diabetes/metadata.json models/diabetes/metrics.json models/diabetes/feature_names.json ./models/diabetes/
COPY models/heart/heart_pipeline.joblib models/heart/metadata.json models/heart/metrics.json models/heart/feature_names.json ./models/heart/
COPY models/kidney/kidney_pipeline.joblib models/kidney/metadata.json models/kidney/metrics.json models/kidney/feature_names.json ./models/kidney/
COPY models/parkinsons/parkinsons_pipeline.joblib models/parkinsons/metadata.json models/parkinsons/metrics.json models/parkinsons/feature_names.json ./models/parkinsons/
RUN chown -R 10001:10001 /app
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health',timeout=3).status==200 else 1)"
STOPSIGNAL SIGTERM
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "15"]
