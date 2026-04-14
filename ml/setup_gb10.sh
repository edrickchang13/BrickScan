#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  BrickScan ML — First-time Setup for NVIDIA GB10 Grace Blackwell (ARM64)
#  Dell Pro Max with GB10 · Ubuntu + NVIDIA DGX OS · 128 GB unified memory
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Run once after cloning the repo:
#      chmod +x ml/setup_gb10.sh && ./ml/setup_gb10.sh
#
# ═══════════════════════════════════════════════════════════════════════════════

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_DIR="$SCRIPT_DIR"
VENV="$ML_DIR/venv"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
error() { echo -e "${RED}[error]${NC} $*"; exit 1; }

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   BrickScan ML — GB10 Grace Blackwell Setup              ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── 0. Verify architecture ────────────────────────────────────────────────────
ARCH=$(uname -m)
if [[ "$ARCH" != "aarch64" ]]; then
    warn "Expected ARM64 (aarch64) but found: $ARCH"
    warn "This script targets the Dell Pro Max GB10. Continuing anyway..."
else
    info "Architecture: aarch64 ✓"
fi

# ── 1. System dependencies ────────────────────────────────────────────────────
info "Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    curl wget unzip git \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    libgomp1 libopenexr-dev

# ── 2. Python virtual environment ─────────────────────────────────────────────
info "Creating Python virtual environment at $VENV ..."
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip wheel setuptools -q

# ── 3. PyTorch for ARM64 + CUDA (Blackwell CC 10.x) ──────────────────────────
info "Installing PyTorch for ARM64 + CUDA (this may take a few minutes)..."
# The DGX OS ships CUDA 12.x; we install the matching PyTorch wheel.
# Blackwell (CC 10.x) support is in torch>=2.5.0
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124 \
    --quiet

# Verify CUDA is visible
python3 -c "
import torch
print(f'  torch {torch.__version__}')
print(f'  CUDA available : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    dev = torch.cuda.get_device_properties(0)
    print(f'  GPU            : {dev.name}')
    print(f'  VRAM (unified) : {dev.total_memory / 1e9:.1f} GB')
    print(f'  Compute cap    : {dev.major}.{dev.minor}')
" && info "PyTorch CUDA ✓" || warn "CUDA not detected — training will use CPU"

# ── 4. ONNX Runtime for ARM64 + CUDA ─────────────────────────────────────────
info "Installing ONNX Runtime GPU (ARM64)..."
# NVIDIA distributes the ARM64 CUDA-enabled onnxruntime via pypi.nvidia.com
pip install onnxruntime-gpu \
    --extra-index-url https://pypi.nvidia.com \
    --quiet \
    || pip install onnxruntime --quiet   # CPU fallback if GPU package unavailable

# ── 5. Remaining ML training deps ─────────────────────────────────────────────
info "Installing remaining training dependencies..."
pip install -r "$ML_DIR/training/requirements.txt" --quiet

# ── 6. Blender ARM64 ──────────────────────────────────────────────────────────
BLENDER_DIR="$ML_DIR/blender/blender_app"
BLENDER_BIN="$BLENDER_DIR/blender"

if [ -f "$BLENDER_BIN" ]; then
    info "Blender already installed at $BLENDER_BIN"
else
    info "Downloading Blender 4.x for Linux ARM64..."
    mkdir -p "$BLENDER_DIR"
    BLENDER_URL="https://mirrors.ocf.berkeley.edu/blender/release/Blender4.2/blender-4.2.0-linux-arm64.tar.xz"
    BLENDER_TMP="/tmp/blender_arm64.tar.xz"

    if command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$BLENDER_TMP" "$BLENDER_URL"
    else
        wget --progress=bar:force -O "$BLENDER_TMP" "$BLENDER_URL"
    fi

    info "Extracting Blender..."
    tar -xf "$BLENDER_TMP" --strip-components=1 -C "$BLENDER_DIR"
    rm -f "$BLENDER_TMP"
    chmod +x "$BLENDER_BIN"
    info "Blender installed: $($BLENDER_BIN --version 2>&1 | head -1)"
fi

# ── 7. Download LDraw parts library ───────────────────────────────────────────
info "Setting up LDraw parts library..."
bash "$ML_DIR/blender/setup_ldraw.sh"

# ── 8. Create directory structure ─────────────────────────────────────────────
info "Creating data/models directories..."
mkdir -p \
    "$ML_DIR/data/renders" \
    "$ML_DIR/models" \
    "$ML_DIR/models/logs"

# ── 9. Write .env with paths for convenience ──────────────────────────────────
cat > "$ML_DIR/.env" <<EOF
ML_DIR=$ML_DIR
BLENDER_BIN=$BLENDER_BIN
VENV=$VENV
DATA_DIR=$ML_DIR/data
MODELS_DIR=$ML_DIR/models
LDRAW_DIR=$ML_DIR/data/ldraw
EOF
info "Environment saved to $ML_DIR/.env"

# ── 10. Blender batch_render dep install ──────────────────────────────────────
info "Installing Blender orchestrator dependencies..."
pip install -r "$ML_DIR/blender/requirements.txt" --quiet

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   Setup complete!  Next steps:                           ║"
echo "║                                                           ║"
echo "║   1. Generate training data:                             ║"
echo "║      source ml/venv/bin/activate                         ║"
echo "║      python ml/blender/batch_render.py \\                 ║"
echo "║        --blender ml/blender/blender_app/blender \\        ║"
echo "║        --workers 8                                        ║"
echo "║                                                           ║"
echo "║   2. Train the model:                                    ║"
echo "║      bash ml/training/train.sh                           ║"
echo "║                                                           ║"
echo "║   3. Export to ONNX:                                     ║"
echo "║      python ml/training/export_onnx.py \\                 ║"
echo "║        --checkpoint ml/models/best_model.pt \\            ║"
echo "║        --output ml/models/lego_classifier.onnx \\         ║"
echo "║        --labels-dir ml/models                            ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
