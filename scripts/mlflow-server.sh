#!/usr/bin/env bash
# Start the local MLflow tracking server for this project.
# Runs and metadata live in mlflow.db, artifacts under mlartifacts/.
# Port 5001 because macOS AirPlay occupies 5000.
set -euo pipefail
cd "$(dirname "$0")/.."

exec .venv/bin/mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts \
  --host 127.0.0.1 \
  --port 5001
