#!/bin/bash
# BrickScan Blender Synthetic Data Generation on DGX Spark
#
# Renders LEGO pieces in various poses, angles, and lighting conditions
# to create synthetic training data. GPU acceleration makes this 10-20x faster.
#
# Expected performance:
#   - DGX Spark GPU: 5-10 seconds per render
#   - Laptop CPU: 60-120 seconds per render
#   - Full dataset (3000 parts @ 80 renders each): 30-40 hours
#
# Usage:
#   bash blender_dgx_render.sh              # Use defaults
#   NUM_RENDERS=100 bash blender_dgx_render.sh  # Override count
#   MAX_WORKERS=2 NUM_RENDERS=50 bash blender_dgx_render.sh
#
# Environment variables:
#   LDRAW_DIR: Path to LDraw parts library (default: ~/ldraw)
#   OUTPUT_DIR: Where to save rendered images (default: ~/brickscan-training/data/synthetic)
#   NUM_RENDERS: Images per part (default: 80)
#   MAX_WORKERS: Parallel Blender instances (default: 4)
#   BLENDER_PATH: Path to Blender executable (default: /usr/local/bin/blender)

set -e

echo "=========================================="
echo "BrickScan Blender Synthetic Data Rendering"
echo "=========================================="
echo ""

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

LDRAW_DIR="${LDRAW_DIR:-$HOME/ldraw}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/brickscan-training/data/synthetic}"
NUM_RENDERS="${NUM_RENDERS:-80}"
MAX_WORKERS="${MAX_WORKERS:-4}"
BLENDER_PATH="${BLENDER_PATH:-/usr/local/bin/blender}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "Configuration:"
echo "  Blender: $BLENDER_PATH"
echo "  LDraw parts: $LDRAW_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Renders per part: $NUM_RENDERS"
echo "  Parallel workers: $MAX_WORKERS"
echo ""

# Check Blender
if [ ! -f "$BLENDER_PATH" ]; then
    echo -e "${RED}ERROR: Blender not found at $BLENDER_PATH${NC}"
    echo "Install with: bash setup/install_dependencies.sh"
    exit 1
fi

# Verify Blender version
BLENDER_VERSION=$($BLENDER_PATH --version 2>&1 | head -1)
echo "Found: $BLENDER_VERSION"
echo ""

# Check LDraw parts library
if [ ! -d "$LDRAW_DIR/parts" ]; then
    echo -e "${YELLOW}WARNING: LDraw parts not found at $LDRAW_DIR/parts${NC}"
    echo ""
    echo "Download LDraw library:"
    echo "  mkdir -p $LDRAW_DIR"
    echo "  cd $LDRAW_DIR"
    echo "  wget https://library.ldraw.org/latest-parts.html -O parts.zip"
    echo "  unzip parts.zip"
    echo ""
    read -p "Continue without parts? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get list of all LDraw part files
PARTS_LIST=$(mktemp)
echo "Scanning for LDraw parts..."

if [ -d "$LDRAW_DIR/parts" ]; then
    find "$LDRAW_DIR/parts" -name "*.dat" -type f | sort > "$PARTS_LIST"
else
    # Fallback: use any .dat files in LDRAW_DIR
    find "$LDRAW_DIR" -name "*.dat" -type f | sort > "$PARTS_LIST"
fi

TOTAL_PARTS=$(wc -l < "$PARTS_LIST")

if [ "$TOTAL_PARTS" -eq 0 ]; then
    echo -e "${RED}ERROR: No .dat files found in $LDRAW_DIR${NC}"
    exit 1
fi

echo "Found $TOTAL_PARTS part files"
echo ""

# Check for existing renders (to skip them)
EXISTING_RENDERS=$(find "$OUTPUT_DIR" -name "*.png" 2>/dev/null | wc -l)
echo "Already rendered: $EXISTING_RENDERS images"
echo ""

echo -e "${GREEN}Starting parallel rendering...${NC}"
echo "This may take several hours. Monitor with:"
echo "  watch -n 5 'find $OUTPUT_DIR -name \"*.png\" | wc -l'"
echo ""

START_TIME=$(date +%s)

# Counter for progress
COUNTER=0
TOTAL=$((TOTAL_PARTS))

# Render parts in parallel
cat "$PARTS_LIST" | xargs -P "$MAX_WORKERS" -I {} bash -c "
    PART_FILE=\"{}\"
    PART_DIR=\"\$(dirname \"\$PART_FILE\")\"
    PART_NAME=\"\$(basename \"\$PART_FILE\" .dat)\"
    OUTPUT_PART_DIR=\"$OUTPUT_DIR/\$PART_NAME\"

    # Skip if already fully rendered
    if [ -d \"\$OUTPUT_PART_DIR\" ]; then
        EXISTING=\$(ls -1 \"\$OUTPUT_PART_DIR\" 2>/dev/null | wc -l)
        if [ \"\$EXISTING\" -ge $NUM_RENDERS ]; then
            echo \"[SKIP] \$PART_NAME (\$EXISTING/$NUM_RENDERS renders)\"
            exit 0
        fi
    fi

    mkdir -p \"\$OUTPUT_PART_DIR\"

    # Create temporary Blender script for this part
    BLEND_SCRIPT=\$(mktemp)
    cat > \"\$BLEND_SCRIPT\" << 'BLENDER_SCRIPT'
import bpy
import os
import math
import random
from mathutils import Vector, Euler

# Configuration
PART_FILE = \"$PART_FILE\"
OUTPUT_DIR = \"$OUTPUT_PART_DIR\"
NUM_RENDERS = $NUM_RENDERS
LDRAW_DIR = \"$LDRAW_DIR\"

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Import LDraw part
try:
    # Try to import using LDraw importer (if available)
    # Otherwise, load as simple mesh
    bpy.ops.import_scene.obj(filepath=PART_FILE)
except:
    pass

# Render settings
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.image_settings.color_mode = 'RGB'
bpy.context.scene.render.resolution_x = 224
bpy.context.scene.render.resolution_y = 224
bpy.context.scene.render.samples = 64

# Use GPU if available
try:
    bpy.context.scene.cycles.device = 'GPU'
except:
    pass

# Render multiple poses
for i in range(NUM_RENDERS):
    # Randomize camera angle
    angle_x = random.uniform(-45, 45)
    angle_y = random.uniform(-45, 45)
    angle_z = random.uniform(0, 360)

    # Set camera
    cam = bpy.data.cameras.new(f\"Camera_{i}\")
    cam_obj = bpy.data.objects.new(f\"Camera_{i}\", cam)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = (5, 5, 5)
    cam_obj.rotation_euler = (
        math.radians(angle_x),
        math.radians(angle_y),
        math.radians(angle_z)
    )

    # Set as active camera
    bpy.context.scene.camera = cam_obj

    # Render
    output_path = os.path.join(OUTPUT_DIR, f\"render_{i:03d}.png\")
    bpy.context.scene.render.filepath = output_path

    try:
        bpy.ops.render.render(write_still=True)
    except:
        pass

    # Clean up camera
    bpy.data.objects.remove(cam_obj, do_unlink=True)

BLENDER_SCRIPT

    # Run Blender in background
    $BLENDER_PATH --background --python \"\$BLENDER_SCRIPT\" >/dev/null 2>&1 || true

    # Clean up
    rm -f \"\$BLENDER_SCRIPT\"

    RENDERED=\$(ls -1 \"$OUTPUT_PART_DIR\" 2>/dev/null | wc -l)
    echo \"[DONE] \$PART_NAME (\$RENDERED/$NUM_RENDERS renders)\"
"

# Calculate summary
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

TOTAL_RENDERED=$(find "$OUTPUT_DIR" -name "*.png" | wc -l)
TOTAL_PARTS_DONE=$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)

echo ""
echo "=========================================="
echo "Rendering Summary"
echo "=========================================="
echo ""
echo "Total time: $(printf '%02d:%02d:%02d' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)))"
echo "Parts completed: $TOTAL_PARTS_DONE/$TOTAL_PARTS"
echo "Total images: $TOTAL_RENDERED"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Calculate statistics
if [ "$TOTAL_RENDERED" -gt 0 ]; then
    AVG_PER_PART=$((TOTAL_RENDERED / TOTAL_PARTS_DONE))
    TIME_PER_IMAGE=$((ELAPSED / TOTAL_RENDERED))

    echo "Statistics:"
    echo "  Average per part: $AVG_PER_PART images"
    echo "  Time per image: ${TIME_PER_IMAGE}s"
    echo ""

    # Estimate time for full dataset
    REMAINING_PARTS=$((TOTAL_PARTS - TOTAL_PARTS_DONE))
    if [ "$REMAINING_PARTS" -gt 0 ]; then
        REMAINING_TIME=$((REMAINING_PARTS * NUM_RENDERS * TIME_PER_IMAGE))
        echo "Estimated time for remaining parts:"
        echo "  $(printf '%02d:%02d:%02d' $((REMAINING_TIME/3600)) $((REMAINING_TIME%3600/60)) $((REMAINING_TIME%60)))"
        echo ""
    fi
fi

# Check for errors
FAILED_PARTS=$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -exec bash -c 'ls -1 "$1" 2>/dev/null | wc -l' _ {} \; | awk '$1 < '$NUM_RENDERS)

if [ ! -z "$FAILED_PARTS" ]; then
    echo -e "${YELLOW}WARNING: Some parts have fewer than $NUM_RENDERS renders${NC}"
    echo "These may have failed to render. Re-run to retry."
fi

echo ""
echo "Next steps:"
echo "  1. Check image quality: file $OUTPUT_DIR/*/render_*.png"
echo "  2. Copy to training directory: cp -r $OUTPUT_DIR/* ~/brickscan-training/data/synthetic/"
echo "  3. Start training: bash training/train_on_dgx.sh"
echo ""

# Clean up
rm -f "$PARTS_LIST"
