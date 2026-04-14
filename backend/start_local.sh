#!/bin/bash
# Start BrickScan backend in local mode
cd "$(dirname "$0")"

# Use the existing .env file
export $(grep -v '^#' .env | xargs 2>/dev/null) 2>/dev/null || true

# Update ML model path to local model location
export ML_MODEL_PATH="$HOME/brickscan/ml/models/lego_classifier.onnx"

echo "Starting BrickScan backend on http://localhost:8000"
echo "ML model: $ML_MODEL_PATH"
echo "Note: PostgreSQL not required — local inventory uses SQLite"
echo ""

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
