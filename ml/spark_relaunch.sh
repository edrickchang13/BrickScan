#!/usr/bin/env bash
# spark_relaunch.sh — Re-deploy all scripts and restart training.
# Run locally on Mac whenever Spark comes back online.
#
# Usage: bash spark_relaunch.sh
#
# What it does:
#   1. Verifies SSH connectivity
#   2. Deploys latest versions of all training scripts
#   3. Installs / verifies Python dependencies (peft, ultralytics, etc.)
#   4. Kills any stale training processes
#   5. Relaunches contrastive + YOLO workstreams

set -euo pipefail

SPARK="edrick@100.124.143.19"
LOCAL_ML="$HOME/Documents/Claude/Projects/Lego/brickscan/ml"
REMOTE_ML="~/brickscan/ml"

echo "===== BrickScan Spark Relaunch ====="
echo "$(date)"
echo ""

# ── 1. Test connectivity ──────────────────────────────────────────────────────
echo "[1/5] Checking connectivity..."
ssh -o ConnectTimeout=10 "$SPARK" "echo connected && uptime" || {
    echo "ERROR: Cannot connect to Spark. Check that Tailscale is running."
    exit 1
}

# ── 2. Deploy all scripts ─────────────────────────────────────────────────────
echo ""
echo "[2/5] Deploying scripts..."
ssh "$SPARK" "mkdir -p ~/brickscan/ml/blender ~/brickscan/ml/data ~/brickscan/ml/logs ~/brickscan/ml/output"

scp "$LOCAL_ML/train_contrastive.py"              "$SPARK:$REMOTE_ML/train_contrastive.py"
scp "$LOCAL_ML/train_yolo.py"                     "$SPARK:$REMOTE_ML/train_yolo.py"
scp "$LOCAL_ML/build_reference_index.py"          "$SPARK:$REMOTE_ML/build_reference_index.py"
scp "$LOCAL_ML/data/download_lego_datasets.py"    "$SPARK:$REMOTE_ML/data/download_lego_datasets.py"

echo "Scripts deployed."

# ── 3. Install / verify dependencies ─────────────────────────────────────────
echo ""
echo "[3/5] Verifying dependencies..."
ssh "$SPARK" "
source ~/brickscan/ml/venv/bin/activate 2>/dev/null || true
pip install peft ultralytics tqdm requests --quiet
echo '  peft:' \$(pip show peft | grep Version)
echo '  ultralytics:' \$(pip show ultralytics | grep Version)
echo 'deps ok'
"

# ── 4. Ensure labeled_by_part symlinks exist ──────────────────────────────────
echo ""
echo "[4/5] Verifying labeled_by_part symlinks..."
ssh "$SPARK" "python3 - <<'PYEOF'
import json, os
from pathlib import Path

info_path = Path('/home/edrick/brickscan/ml/training_data/huggingface_legobricks/cache')
info_files = list(info_path.rglob('dataset_info.json'))
if not info_files:
    print('  WARNING: dataset_info.json not found — skipping symlink creation')
    exit(0)

with open(info_files[0]) as f:
    info = json.load(f)

names = info['features']['label']['names']
images_dir = Path('/home/edrick/brickscan/ml/training_data/huggingface_legobricks/images')
out_dir = Path('/home/edrick/brickscan/ml/data/labeled_by_part')
out_dir.mkdir(parents=True, exist_ok=True)

created, existing = 0, 0
for i, part_name in enumerate(names):
    src = images_dir / str(i)
    dst = out_dir / part_name
    if dst.exists():
        existing += 1
        continue
    if src.exists():
        os.symlink(src, dst)
        created += 1

print(f'  labeled_by_part: {created} new symlinks, {existing} already existed')
print(f'  Total: {len(list(out_dir.iterdir()))} part classes')
PYEOF"

# ── 5. Launch training workstreams ────────────────────────────────────────────
echo ""
echo "[5/5] Launching training workstreams..."

ssh "$SPARK" "
source ~/brickscan/ml/venv/bin/activate 2>/dev/null || true
mkdir -p ~/brickscan/ml/logs ~/brickscan/models/contrastive ~/brickscan/models/yolo

# Kill any stale training processes
pkill -f 'train_contrastive|train_yolo' 2>/dev/null || true
sleep 2

# ── Workstream A: Contrastive (DINOv2 + LoRA + gradient checkpointing) ──
# Key flags:
#   --image-size 224   : required — DINOv2 native is 518, we train at 224
#   --batch-size 256   : fits in GB10 130GB with LoRA + grad checkpointing
nohup python3 ~/brickscan/ml/train_contrastive.py \
  --data-dir    ~/brickscan/ml/training_data/huggingface_legobricks/images/ \
  --output-dir  ~/brickscan/models/contrastive/ \
  --epochs      50 \
  --batch-size  256 \
  --image-size  224 \
  > ~/brickscan/ml/logs/contrastive_train.log 2>&1 &
echo \"Contrastive PID: \$!\"
sleep 3

# ── Workstream C: YOLO detection ──
if [ -f ~/brickscan/yolo_dataset/lego.yaml ]; then
  echo 'YOLO dataset exists — launching training directly'
  nohup python3 ~/brickscan/ml/train_yolo.py \
    --data       ~/brickscan/yolo_dataset/lego.yaml \
    --output     ~/brickscan/models/yolo/ \
    --epochs     100 \
    --batch-size 32 \
    > ~/brickscan/ml/logs/yolo_full.log 2>&1 &
  echo \"YOLO training PID: \$!\"
else
  echo 'WARNING: YOLO dataset not found at ~/brickscan/yolo_dataset/lego.yaml'
  echo 'YOLO training skipped — run generate_multipiece_scenes.py first'
fi

# Quick health check after 10s
sleep 10
echo ''
echo '=== Health check ==='
pgrep -fa train_contrastive | grep -v grep | head -2 || echo 'WARNING: contrastive not running!'
pgrep -fa train_yolo | grep -v grep | head -1 || echo 'WARNING: yolo not running'
echo 'Contrastive log tail:'
tail -5 ~/brickscan/ml/logs/contrastive_train.log 2>/dev/null || echo '(no log yet)'
"

echo ""
echo "===== All workstreams launched ====="
echo ""
echo "Monitor:"
echo "  ssh spark 'tail -f ~/brickscan/ml/logs/contrastive_train.log'"
echo "  ssh spark 'tail -f ~/brickscan/ml/logs/yolo_full.log'"
echo "  ssh spark 'nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader'"
