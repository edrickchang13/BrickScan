#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  BrickScan ML — Full Pipeline Launcher (GB10 Grace Blackwell / ARM64)
#
#  Usage:
#    ./launch.sh setup          # First-time Python venv + deps setup (no sudo)
#    ./launch.sh download       # Download training images from Rebrickable/BrickLink
#    ./launch.sh train          # Train EfficientNet-B3 classifier
#    ./launch.sh export         # Export best checkpoint → ONNX
#    ./launch.sh all            # setup → download → train → export
#    ./launch.sh status         # Show pipeline status
# ═══════════════════════════════════════════════════════════════════════════════

set -e

ML_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ML_DIR/venv"
DATA_DIR="$ML_DIR/data"
MODELS_DIR="$ML_DIR/models"

GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'
step()  { echo -e "\n${CYAN}${BOLD}▶ $*${NC}"; }
ok()    { echo -e "${GREEN}✓ $*${NC}"; }
warn()  { echo -e "${YELLOW}⚠ $*${NC}"; }
error() { echo -e "${RED}✗ $*${NC}"; exit 1; }

activate_venv() {
    if [ -f "$VENV/bin/activate" ]; then
        source "$VENV/bin/activate"
    else
        warn "Virtual environment not found. Run './launch.sh setup' first."
    fi
}

# ── STATUS ────────────────────────────────────────────────────────────────────
cmd_status() {
    echo -e "\n${BOLD}BrickScan ML Pipeline Status${NC}"
    echo "────────────────────────────────────────"

    [ -f "$VENV/bin/python" ] \
        && ok "Python venv ($(source "$VENV/bin/activate" && python --version))" \
        || warn "Python venv (run: ./launch.sh setup)"

    INDEX="$DATA_DIR/index.csv"
    if [ -f "$INDEX" ]; then
        IMGS=$(tail -n +2 "$INDEX" 2>/dev/null | wc -l)
        PARTS=$(tail -n +2 "$INDEX" 2>/dev/null | cut -d, -f2 | sort -u | wc -l)
        ok "Training data: ${IMGS} images / ${PARTS} parts"
    else
        warn "No training data yet (run: ./launch.sh download)"
    fi

    [ -f "$MODELS_DIR/best_model.pt" ] \
        && ok "Best checkpoint ($(du -sh "$MODELS_DIR/best_model.pt" | cut -f1))" \
        || warn "No trained checkpoint (run: ./launch.sh train)"

    [ -f "$MODELS_DIR/lego_classifier.onnx" ] \
        && ok "ONNX model ($(du -sh "$MODELS_DIR/lego_classifier.onnx" | cut -f1))" \
        || warn "ONNX not exported yet (run: ./launch.sh export)"

    [ -f "$MODELS_DIR/part_labels.json" ]  && ok "part_labels.json"  || warn "part_labels.json missing"
    [ -f "$MODELS_DIR/color_labels.json" ] && ok "color_labels.json" || warn "color_labels.json missing"
    echo ""
}

# ── SETUP ─────────────────────────────────────────────────────────────────────
cmd_setup() {
    step "Setting up Python environment on GB10 (no sudo required)…"
    bash "$ML_DIR/setup_nosudo.sh"
    ok "Setup complete"
}

# ── DOWNLOAD ──────────────────────────────────────────────────────────────────
cmd_download() {
    step "Downloading LEGO training images from Rebrickable + BrickLink…"
    activate_venv

    MAX_PARTS="${DOWNLOAD_MAX_PARTS:-}"
    WORKERS="${DOWNLOAD_WORKERS:-32}"
    API_KEY="${REBRICKABLE_KEY:-}"

    ARGS=""
    [ -n "$MAX_PARTS" ] && ARGS="$ARGS --max-parts $MAX_PARTS"
    [ -n "$API_KEY"   ] && ARGS="$ARGS --api-key $API_KEY"

    python "$ML_DIR/data/download_rebrickable.py" \
        --workers    "$WORKERS" \
        --image-size 256 \
        $ARGS

    ok "Download complete. Index: $DATA_DIR/index.csv"
}

# ── TRAIN ─────────────────────────────────────────────────────────────────────
cmd_train() {
    step "Training EfficientNet-B3 on GB10 Grace Blackwell…"
    activate_venv
    bash "$ML_DIR/training/train.sh"
    ok "Training complete"
}

# ── EXPORT ────────────────────────────────────────────────────────────────────
cmd_export() {
    step "Exporting best checkpoint → ONNX…"
    activate_venv

    CHECKPOINT="$MODELS_DIR/best_model.pt"
    [ -f "$CHECKPOINT" ] || error "No checkpoint at $CHECKPOINT. Run train first."

    python "$ML_DIR/training/export_onnx.py" \
        --checkpoint "$CHECKPOINT" \
        --output     "$MODELS_DIR/lego_classifier.onnx" \
        --labels-dir "$MODELS_DIR"

    # Symlink to backend
    BACKEND_MODELS="$ML_DIR/../backend/models"
    mkdir -p "$BACKEND_MODELS"
    ln -sf "$MODELS_DIR/lego_classifier.onnx" "$BACKEND_MODELS/lego_classifier.onnx" 2>/dev/null || true
    ln -sf "$MODELS_DIR/part_labels.json"     "$BACKEND_MODELS/part_labels.json"     2>/dev/null || true
    ln -sf "$MODELS_DIR/color_labels.json"    "$BACKEND_MODELS/color_labels.json"    2>/dev/null || true

    ok "ONNX ready at $MODELS_DIR/lego_classifier.onnx"
    ok "Symlinked to backend/models/"
}

# ── ALL ───────────────────────────────────────────────────────────────────────
cmd_all() {
    cmd_setup
    cmd_download
    cmd_train
    cmd_export
    echo ""
    echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════╗"
    echo -e "║  Full pipeline complete!                 ║"
    echo -e "║  ONNX model is live in the backend.      ║"
    echo -e "╚══════════════════════════════════════════╝${NC}"
}

# ── DISPATCH ─────────────────────────────────────────────────────────────────
CMD="${1:-status}"
shift 2>/dev/null || true

case "$CMD" in
    setup)    cmd_setup    "$@" ;;
    download) cmd_download "$@" ;;
    train)    cmd_train    "$@" ;;
    export)   cmd_export   "$@" ;;
    all)      cmd_all      "$@" ;;
    status)   cmd_status   "$@" ;;
    *)
        echo "Usage: $0 {setup|download|train|export|all|status}"
        exit 1
        ;;
esac
