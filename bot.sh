#!/bin/bash

# --- Define Variables ---
# Project ID is set globally via `gcloud config set project`
# Or uncomment and set explicitly:
# PROJECT_ID="businesschat-staging"
PROJECT_ID=businesschat-staging
REGION="europe-west3" # Or your preferred region
SERVICE_NAME="haroun-bot" # Choose a unique name for your service
IMAGE_URI="docker.io/haroun9/bot:latest" # Your Docker Hub image

# --- Secrets Configuration ---
OPENAI_API_KEY_SECRET="OPENAI_API_KEY_SECRET" # Must match the secret name in Secret Manager
OPENAI_API_KEY_SECRET_VERSION="latest"

# Add LangSmith secret variables here if you use LangSmith
# LANGCHAIN_API_KEY_SECRET="your-langsmith-secret-name"
# LANGCHAIN_API_KEY_SECRET_VERSION="latest"

echo "Deploying service '${SERVICE_NAME}' in project '${PROJECT_ID}' region '${REGION}' using image '${IMAGE_URI}'..."

# --- Prepare Secret Arguments ---
SECRET_ARGS="OPENAI_API_KEY=${OPENAI_API_KEY_SECRET}:${OPENAI_API_KEY_SECRET_VERSION}"
# Append LangSmith secrets if configured
# SECRET_ARGS+=",LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY_SECRET}:${LANGCHAIN_API_KEY_SECRET_VERSION}"

# --- Prepare Environment Variable Arguments (Optional) ---
# No specific app-level env vars needed now besides secrets and PORT (set by Cloud Run)
# If you used LANGCHAIN_TRACING_V2=true etc., add them here:
ENV_VAR_ARGS=""
# Example: ENV_VAR_ARGS="^##^LANGCHAIN_TRACING_V2=true##LANGCHAIN_PROJECT=your-project" # Use ^##^ as separator

# --- Deploy Command ---
gcloud run deploy ${SERVICE_NAME} \
  --image=${IMAGE_URI} \
  --platform=managed \
  --region=${REGION} \
  --project=${PROJECT_ID} \
  --allow-unauthenticated \
  --set-secrets=${SECRET_ARGS} \
  $( [ -n "$ENV_VAR_ARGS" ] && echo "--set-env-vars=${ENV_VAR_ARGS}" ) \
  --memory=2Gi \
  --cpu=1 \
  --quiet # Add quiet to reduce interactive prompts if preferred

# Check deployment status (optional)
if [ $? -eq 0 ]; then
  echo "Deployment command submitted successfully."
  echo "Getting service URL..."
  SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --platform=managed --region=${REGION} --project=${PROJECT_ID} --format='value(status.url)')
  echo "-------------------------------------"
  echo "Service Name: ${SERVICE_NAME}"
  echo "Region:       ${REGION}"
  echo "Project:      ${PROJECT_ID}"
  echo "Service URL:  ${SERVICE_URL}"
  echo "-------------------------------------"
else
  echo "Deployment command failed."
  exit 1
fi