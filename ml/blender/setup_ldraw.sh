#!/bin/bash
# Setup script to download and extract the complete LDraw parts library

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LDRAW_DIR="$PROJECT_ROOT/data/ldraw"

echo "=================================================="
echo "BrickScan LDraw Library Setup"
echo "=================================================="
echo "Download destination: $LDRAW_DIR"

# Create target directory
mkdir -p "$LDRAW_DIR"

# URLs
LDRAW_URL="https://library.ldraw.org/library/updates/complete.zip"

# Download
echo ""
echo "1. Downloading complete LDraw library (this may take a few minutes)..."
TEMP_ZIP="/tmp/ldraw_complete.zip"

if command -v curl &> /dev/null; then
    curl -L --progress-bar --output "$TEMP_ZIP" "$LDRAW_URL"
elif command -v wget &> /dev/null; then
    wget --show-progress -O "$TEMP_ZIP" "$LDRAW_URL"
else
    echo "ERROR: Neither curl nor wget found. Please install one of them."
    exit 1
fi

# Extract
echo ""
echo "2. Extracting LDraw library..."
if command -v unzip &> /dev/null; then
    unzip -q "$TEMP_ZIP" -d "$LDRAW_DIR"
elif command -v 7z &> /dev/null; then
    7z x "$TEMP_ZIP" -o"$LDRAW_DIR"
else
    echo "ERROR: unzip or 7z not found. Cannot extract."
    exit 1
fi

rm -f "$TEMP_ZIP"

# Summary
echo ""
echo "3. Library extraction complete. Scanning..."
echo ""

# Count files
PARTS_COUNT=$(find "$LDRAW_DIR/ldraw/parts" -name "*.dat" 2>/dev/null | wc -l)
PRIMS_COUNT=$(find "$LDRAW_DIR/ldraw/p" -name "*.dat" 2>/dev/null | wc -l)
TOTAL_COUNT=$((PARTS_COUNT + PRIMS_COUNT))

echo "=================================================="
echo "LDraw Library Summary"
echo "=================================================="
echo "Parts directory: $LDRAW_DIR/ldraw/parts"
echo "Part .dat files found: $PARTS_COUNT"
echo "Primitives .dat files: $PRIMS_COUNT"
echo "Total .dat files: $TOTAL_COUNT"
echo ""
echo "Setup complete! You can now run batch_render.py"
echo "=================================================="
