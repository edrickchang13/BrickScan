#!/usr/bin/env bash
# spark_export.sh — Pull DINOv2 checkpoint from DGX Spark, export ONNX, SCP back.
#
# Edit the four SPARK_* variables below with your actual credentials,
# then run:   bash ml/spark_export.sh
#
# What this does:
#   1. SSHs to the Spark and runs export_dinov2_onnx.py remotely
#   2. SCPs the resulting output dir back to brickscan/ml/spark_output/
#   3. Copies the ONNX + labels into the backend model dir
#   4. Prints the next commands to run (benchmark + CoreML export)

set -euo pipefail

# ── Configure these ────────────────────────────────────────────────────────────
SPARK_HOST="192.168.x.x"                      # DGX Spark IP or hostname
SPARK_USER="edrick"                            # SSH username
SPARK_KEY="$HOME/.ssh/id_rsa"                 # Path to SSH private key
SPARK_CHECKPOINT="/workspace/brickscan/checkpoints/best_model.pt"  # Checkpoint path on Spark
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
LOCAL_OUTPUT="$SCRIPT_DIR/spark_output"
BACKEND_MODEL_DIR="$REPO_ROOT/backend/models/dinov2"

SSH="ssh -i $SPARK_KEY -o StrictHostKeyChecking=no $SPARK_USER@$SPARK_HOST"
SCP="scp -i $SPARK_KEY -o StrictHostKeyChecking=no"

echo "═══════════════════════════════════════════════════"
echo "  BrickScan — DINOv2 ONNX Export from DGX Spark"
echo "═══════════════════════════════════════════════════"
echo "  Spark:      $SPARK_USER@$SPARK_HOST"
echo "  Checkpoint: $SPARK_CHECKPOINT"
echo "  Local out:  $LOCAL_OUTPUT"
echo ""

# ── Step 1: Copy export script to Spark ──────────────────────────────────────
echo "[1/4] Uploading export_dinov2_onnx.py to Spark …"
$SSH "mkdir -p /tmp/brickscan_export"
$SCP "$SCRIPT_DIR/export_dinov2_onnx.py" \
     "$SPARK_USER@$SPARK_HOST:/tmp/brickscan_export/export_dinov2_onnx.py"

# ── Step 2: Run export on the Spark ──────────────────────────────────────────
echo "[2/4] Running ONNX export on Spark …"
$SSH "cd /tmp/brickscan_export && \
      pip install timm onnx onnxruntime --quiet && \
      python export_dinov2_onnx.py \
        --checkpoint $SPARK_CHECKPOINT \
        --output-dir /tmp/brickscan_export/output \
        --fp16 \
        --device cuda"

# ── Step 3: SCP output back ───────────────────────────────────────────────────
echo "[3/4] Downloading ONNX output …"
mkdir -p "$LOCAL_OUTPUT"
$SCP -r "$SPARK_USER@$SPARK_HOST:/tmp/brickscan_export/output/." "$LOCAL_OUTPUT/"

echo "      Files downloaded to: $LOCAL_OUTPUT"
ls -lh "$LOCAL_OUTPUT/"

# ── Step 4: Copy to backend model dir ────────────────────────────────────────
echo "[4/4] Installing model into backend …"
mkdir -p "$BACKEND_MODEL_DIR"
cp "$LOCAL_OUTPUT/dinov2_lego.onnx"   "$BACKEND_MODEL_DIR/"
cp "$LOCAL_OUTPUT/part_labels.json"   "$BACKEND_MODEL_DIR/"
cp "$LOCAL_OUTPUT/export_info.json"   "$BACKEND_MODEL_DIR/"
[ -f "$LOCAL_OUTPUT/dinov2_lego_fp16.onnx" ] && \
  cp "$LOCAL_OUTPUT/dinov2_lego_fp16.onnx" "$BACKEND_MODEL_DIR/"

echo ""
echo "✓ Backend model ready at: $BACKEND_MODEL_DIR"
echo "  Update your .env:"
echo "    ML_MODEL_PATH=$BACKEND_MODEL_DIR/dinov2_lego.onnx"
echo "    ML_MODEL_TYPE=dinov2"
echo ""

# ── Next steps ────────────────────────────────────────────────────────────────
EFFNET_ONNX="$REPO_ROOT/backend/models/lego_classifier.onnx"
EFFNET_LABELS="$REPO_ROOT/backend/models/part_labels.json"

echo "══════════════════════════════════════════════════════"
echo "  Next: run the accuracy benchmark"
echo ""
echo "  python ml/benchmark_models.py \\"
echo "    --dinov2       $BACKEND_MODEL_DIR/dinov2_lego.onnx \\"
echo "    --effnet       $EFFNET_ONNX \\"
echo "    --dinov2-labels $BACKEND_MODEL_DIR/part_labels.json \\"
echo "    --effnet-labels $EFFNET_LABELS \\"
echo "    --sample-images backend/data_pipeline/sample_images/images \\"
echo "    --download-n   50 \\"
echo "    --output       ml/benchmark_results.json"
echo ""
echo "  Then CoreML export (run on Mac):"
echo ""
echo "  python ml/export_coreml_dinov2.py \\"
echo "    --onnx    $BACKEND_MODEL_DIR/dinov2_lego.onnx \\"
echo "    --labels  $BACKEND_MODEL_DIR/part_labels.json \\"
echo "    --output  mobile/ios/BrickScanDINOv2.mlpackage \\"
echo "    --quantize int8 \\"
echo "    --benchmark"
echo "══════════════════════════════════════════════════════"
