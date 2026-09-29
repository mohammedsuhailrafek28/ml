# Backup and recovery notes

Back up the two matching image IDs (or verified image archives), `.env` release
configuration, the mounted service-key file, and release-manifest files in an
access-controlled encrypted location. Do not add secrets or generated patient
reports to Git, CI artifacts, or ordinary backups. The application does not
persist submitted measurements or generated PDFs; reports are generated in
memory for download.

Example for a local operator-selected tag; secure the output file using the
host's approved encrypted backup storage:

```powershell
docker image save $env:API_IMAGE $env:WEB_IMAGE -o .\medical-ai-release-images.tar
Copy-Item .env .\release-config.env
Copy-Item .secrets\service_api_key .\release-service-key
```

Verify backups using checksums in the approved backup system. Test recovery by
loading both images, restoring the matching configuration and secret file, then
checking frontend health, readiness, and the five golden prediction/report
smokes. Never print or include the service key in command logs.

Application images and config must be restored as a pair. Persisted models,
preprocessing, thresholds, contracts, and model release manifest must remain a
single compatible release. If any hash or golden prediction differs, keep the
stack stopped and recover a known-good complete release.

This is an educational/research product only, not a diagnosis or screening
service. It is not clinically validated; Parkinson's remains Experimental and
scores are uncalibrated. No persistent patient record store, end-user identity,
distributed rate limiter, centralized monitoring, or deployed availability SLO
exists. Local load results do not prove availability.
