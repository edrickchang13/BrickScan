#!/usr/bin/env bash
# smoke_retrain_dry_run.sh — end-to-end validation of retrain_from_feedback.py
# without actually training anything.
#
# Generates a mock feedback CSV + a stub ONNX checkpoint, then invokes the
# trainer in --dry-run mode. Passes when:
#   1. retrain_from_feedback.py loads the CSV and validates required columns
#   2. Its BrickClassifier import (from train_two_stage) resolves
#   3. Path + image-existence spot checks pass
#
# Run locally on Mac before pushing to Spark so schema / import drift
# surfaces here, not 40 min into a remote training job.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ML_DIR="$REPO_ROOT/ml"

WORK_DIR="$(mktemp -d -t brickscan_dryrun.XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT

echo "[smoke] Workspace: $WORK_DIR"

# ── Mock feedback CSV (5 rows, matching the /feedback/export.csv schema) ────
MOCK_CSV="$WORK_DIR/mock_feedback.csv"
MOCK_IMG_DIR="$WORK_DIR/images"
mkdir -p "$MOCK_IMG_DIR"

# Emit tiny JPEGs so the dry-run "spot-check image paths" step finds them.
# Python one-liner avoids a tool dependency; Pillow ships with the backend venv.
python3 - <<PY
import os, io
from PIL import Image
out = "$MOCK_IMG_DIR"
for p in ("3001", "3002", "3003", "3004", "3005"):
    Image.new("RGB", (64, 64), (30, 80, 200)).save(os.path.join(out, f"{p}.jpg"))
PY

cat > "$MOCK_CSV" <<EOF
image_path,correct_part_num,correct_color_id,original_prediction,source,confidence,timestamp,scan_id,feedback_type,correct_rank
$MOCK_IMG_DIR/3001.jpg,3001,4,3001,brickognize,0.92,2026-04-14T12:00:00Z,scan_a,top_correct,0
$MOCK_IMG_DIR/3002.jpg,3002,1,3001,gemini,0.55,2026-04-14T12:01:00Z,scan_b,alternative_correct,1
$MOCK_IMG_DIR/3003.jpg,3003,14,3002,brickognize,0.71,2026-04-14T12:02:00Z,scan_c,none_correct,-1
$MOCK_IMG_DIR/3004.jpg,3004,15,3004,brickognize+gemini,0.88,2026-04-14T12:03:00Z,scan_d,partially_correct,0
$MOCK_IMG_DIR/3005.jpg,3005,25,3099,contrastive_knn,0.40,2026-04-14T12:04:00Z,scan_e,none_correct,-1
EOF
echo "[smoke] Mock CSV: $MOCK_CSV ($(wc -l < "$MOCK_CSV") rows incl. header)"

# ── Stub ONNX checkpoint (just needs to exist for --dry-run path checks) ────
STUB_CHECKPOINT="$WORK_DIR/stub_checkpoint.pt"
touch "$STUB_CHECKPOINT"

# ── Pick a Python that has torch + pandas available ──────────────────────────
VENV_PY="$REPO_ROOT/backend/venv/bin/python3"
if [[ -x "$VENV_PY" ]]; then
    PY="$VENV_PY"
else
    PY="python3"
fi
echo "[smoke] Using: $PY"

# ── Invoke retrain_from_feedback.py --dry-run ────────────────────────────────
cd "$ML_DIR"
if "$PY" retrain_from_feedback.py \
    --checkpoint "$STUB_CHECKPOINT" \
    --feedback-csv "$MOCK_CSV" \
    --base-data "$WORK_DIR" \
    --output-dir "$WORK_DIR/out" \
    --dry-run; then
    echo "[smoke] ✓ Dry run PASSED"
else
    ret=$?
    echo "[smoke] ✗ Dry run FAILED (exit $ret)" >&2
    exit "$ret"
fi
