#!/bin/bash
# QuickStart script for BrickScan rendering pipeline
# Usage: ./quickstart.sh [blender_path] [num_workers]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

BLENDER_EXE="${1:-blender}"
NUM_WORKERS="${2:-4}"

echo "=============================================="
echo "BrickScan Rendering Pipeline - Quick Start"
echo "=============================================="
echo ""

# Check Blender is available
if ! command -v "$BLENDER_EXE" &> /dev/null; then
    echo "ERROR: Blender not found at '$BLENDER_EXE'"
    echo "Install Blender 4.x or specify path: ./quickstart.sh /path/to/blender"
    exit 1
fi

echo "✓ Blender found: $("$BLENDER_EXE" --version 2>/dev/null | head -1)"
echo ""

# Check Python dependencies
echo "Checking Python dependencies..."
python3 -c "import tqdm, pandas" 2>/dev/null || {
    echo "Installing dependencies..."
    pip install -q -r "$SCRIPT_DIR/requirements.txt"
}
echo "✓ Python dependencies ready"
echo ""

# Check LDraw library
if [ ! -d "$PROJECT_ROOT/data/ldraw/ldraw/parts" ]; then
    echo "LDraw library not found. Downloading..."
    cd "$SCRIPT_DIR"
    ./setup_ldraw.sh
    cd -
else
    PARTS_COUNT=$(find "$PROJECT_ROOT/data/ldraw/ldraw/parts" -name "*.dat" 2>/dev/null | wc -l)
    echo "✓ LDraw library found ($PARTS_COUNT parts)"
fi
echo ""

# Check colors CSV
COLORS_CSV="$PROJECT_ROOT/data/colors.csv"
if [ ! -f "$COLORS_CSV" ]; then
    echo "ERROR: colors.csv not found at $COLORS_CSV"
    echo ""
    echo "To get colors from Rebrickable:"
    echo "  1. Visit: https://rebrickable.com/downloads/"
    echo "  2. Download: colors.csv"
    echo "  3. Save to: $COLORS_CSV"
    echo ""
    echo "Or use the example file:"
    echo "  cp $PROJECT_ROOT/data/colors_example.csv $COLORS_CSV"
    exit 1
fi
echo "✓ Colors CSV found"
echo ""

# Ask for test or full render
echo "Select rendering mode:"
echo ""
echo "1) TEST MODE       - 5 parts, 5 angles, 224px (2-3 minutes)"
echo "2) SMALL BATCH     - 50 parts, 36 angles, 224px (~30 minutes)"
echo "3) PRODUCTION      - All parts, 36 angles, 224px (many hours)"
echo "4) CUSTOM          - Specify your own parameters"
echo ""
read -p "Enter choice (1-4): " MODE

case $MODE in
    1)
        echo ""
        echo "Running TEST MODE..."
        python3 "$SCRIPT_DIR/batch_render.py" \
            --blender "$BLENDER_EXE" \
            --ldraw-dir "$PROJECT_ROOT/data/ldraw" \
            --colors-csv "$COLORS_CSV" \
            --output-dir "$PROJECT_ROOT/data/renders" \
            --max-parts 5 \
            --num-angles 5 \
            --resolution 224 \
            --workers "$NUM_WORKERS"
        ;;
    2)
        echo ""
        echo "Running SMALL BATCH..."
        python3 "$SCRIPT_DIR/batch_render.py" \
            --blender "$BLENDER_EXE" \
            --ldraw-dir "$PROJECT_ROOT/data/ldraw" \
            --colors-csv "$COLORS_CSV" \
            --output-dir "$PROJECT_ROOT/data/renders" \
            --max-parts 50 \
            --num-angles 36 \
            --resolution 224 \
            --workers "$NUM_WORKERS"
        ;;
    3)
        echo ""
        echo "Running PRODUCTION (this will take a while)..."
        python3 "$SCRIPT_DIR/batch_render.py" \
            --blender "$BLENDER_EXE" \
            --ldraw-dir "$PROJECT_ROOT/data/ldraw" \
            --colors-csv "$COLORS_CSV" \
            --output-dir "$PROJECT_ROOT/data/renders" \
            --num-angles 36 \
            --resolution 224 \
            --workers "$NUM_WORKERS"
        ;;
    4)
        echo ""
        echo "Custom mode - run batch_render.py with your own arguments:"
        echo ""
        echo "python3 $SCRIPT_DIR/batch_render.py \\"
        echo "  --blender $BLENDER_EXE \\"
        echo "  --ldraw-dir $PROJECT_ROOT/data/ldraw \\"
        echo "  --colors-csv $COLORS_CSV \\"
        echo "  --output-dir $PROJECT_ROOT/data/renders \\"
        echo "  --workers $NUM_WORKERS \\"
        echo "  --num-angles 36 \\"
        echo "  --resolution 224"
        echo ""
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Rendering complete!"
echo "=============================================="
echo ""
echo "Output location:"
echo "  $PROJECT_ROOT/data/renders/"
echo ""
echo "Metadata index:"
echo "  $PROJECT_ROOT/data/index.csv"
echo ""
echo "Error log:"
echo "  $PROJECT_ROOT/data/failed_renders.log"
echo ""
echo "Next steps:"
echo "  1. Check output quality:"
echo "     find $PROJECT_ROOT/data/renders -name '*.png' | wc -l"
echo ""
echo "  2. Use renders for ML training:"
echo "     import pandas as pd"
echo "     df = pd.read_csv('$PROJECT_ROOT/data/index.csv')"
echo "     print(df.head())"
echo ""
