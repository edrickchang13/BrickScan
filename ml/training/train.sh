#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# BrickScan ML Training Launcher — NVIDIA GB10 Grace Blackwell (ARM64)
# Hardware: 20 ARM cores, 128 GB unified memory, Blackwell GPU (OptiX/CUDA CC10)
# OS: Ubuntu + NVIDIA DGX OS
# ═══════════════════════════════════════════════════════════════════════════════

set -e

# ── GB10: expose the unified GPU/CPU memory pool to CUDA ─────────────────────
# With 128 GB unified memory there is no separate VRAM limit — tell PyTorch
# to allow large allocations from the shared pool.
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# ARM64 thread affinity — use all 20 Grace cores for data loading
export OMP_NUM_THREADS=20

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Paths
DATA_DIR="$(dirname "$SCRIPT_DIR")/data"
OUTPUT_DIR="$(dirname "$SCRIPT_DIR")/models"
MODELS_DIR="$OUTPUT_DIR"
mkdir -p "$MODELS_DIR"

# Activate venv if it exists
VENV_PATH="$(dirname "$(dirname "$SCRIPT_DIR")")/venv/bin/activate"
if [ -f "$VENV_PATH" ]; then
    echo "Activating virtual environment..."
    source "$VENV_PATH"
fi

# ── Hardware info ─────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════"
echo " GB10 Grace Blackwell — System Info"
echo "═══════════════════════════════════════════"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>/dev/null \
    || echo "(nvidia-smi not in PATH — GPU will still be used via CUDA)"
echo "CPU cores: $(nproc)"
echo "RAM: $(free -h | awk '/^Mem:/{print $2}') total"
echo ""

# ── Configuration ─────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════"
echo " Training Configuration"
echo "═══════════════════════════════════════════"
echo "Data dir:   $DATA_DIR"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Check data
if [ ! -f "$DATA_DIR/index.csv" ]; then
    echo "ERROR: $DATA_DIR/index.csv not found."
    echo "Run ml/blender/download_rebrickable.py first to download training data."
    exit 1
fi

DATASET_ROWS=$(tail -n +2 "$DATA_DIR/index.csv" | wc -l)
echo "Dataset rows: $DATASET_ROWS images"
echo ""

# ── Launch training ───────────────────────────────────────────────────────────
# GB10 tuning notes:
#   --batch-size 256   → 128 GB unified memory easily fits 256× 224px RGB tensors
#   --workers 16       → 20 ARM cores; leave 4 for main process + system
#   --lr 3e-4          → slightly higher LR is stable with large batches
echo "═══════════════════════════════════════════"
echo " Starting Training"
echo "═══════════════════════════════════════════"
cd "$SCRIPT_DIR"
python train.py \
    --data-dir   "$DATA_DIR" \
    --output-dir "$MODELS_DIR" \
    --epochs     60 \
    --batch-size 256 \
    --lr         3e-4 \
    --workers    16 \
    --val-split  0.15

echo ""
echo "═══════════════════════════════════════════"
echo " Training Complete — models at $MODELS_DIR"
echo "═══════════════════════════════════════════"
