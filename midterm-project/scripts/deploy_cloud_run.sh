#!/usr/bin/env bash
set -euo pipefail

# Deploy BikePulse container service to Google Cloud Run
PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID to your Google Cloud project ID}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-bikepulse-api}"

cd "$(dirname "$0")/.."

echo "Deploying ${SERVICE_NAME} to Cloud Run (${PROJECT_ID} / ${REGION})..."
gcloud run deploy "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --source . \
  --port 9696 \
  --memory 512Mi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 4 \
  --allow-unauthenticated

SERVICE_URL="$(gcloud run services describe "${SERVICE_NAME}" --project "${PROJECT_ID}" --region "${REGION}" --format='value(status.url)')"
echo "Deployed URL: ${SERVICE_URL}"
uv run python smoke_test.py --url "${SERVICE_URL}"

