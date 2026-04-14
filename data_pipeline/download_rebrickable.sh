#!/bin/bash
# Downloads all Rebrickable CSV database files
# Usage: ./download_rebrickable.sh ./rebrickable_data

OUTPUT_DIR=${1:-./rebrickable_data}
mkdir -p "$OUTPUT_DIR"

BASE_URL="https://cdn.rebrickable.com/media/downloads"

files=(
    "colors.csv.gz"
    "part_categories.csv.gz"
    "parts.csv.gz"
    "part_relationships.csv.gz"
    "elements.csv.gz"
    "themes.csv.gz"
    "sets.csv.gz"
    "inventories.csv.gz"
    "inventory_parts.csv.gz"
    "inventory_sets.csv.gz"
    "inventory_minifigs.csv.gz"
)

echo "======================================="
echo "Rebrickable CSV Downloader"
echo "======================================="
echo ""
echo "Downloading to: $OUTPUT_DIR"
echo ""

downloaded=0
failed=0

for file in "${files[@]}"; do
    echo "Downloading $file..."
    if curl -L -f -o "$OUTPUT_DIR/$file" "$BASE_URL/$file" 2>/dev/null; then
        echo "  ✓ Downloaded successfully"
        echo "  Extracting..."
        if gunzip -f "$OUTPUT_DIR/$file" 2>/dev/null; then
            echo "  ✓ Extracted"
            ((downloaded++))
        else
            echo "  ✗ Failed to extract"
            ((failed++))
        fi
    else
        echo "  ✗ Download failed"
        ((failed++))
    fi
    echo ""
done

echo "======================================="
echo "Download Summary"
echo "======================================="
echo "Successfully downloaded: $downloaded"
echo "Failed: $failed"
echo ""
echo "Files in $OUTPUT_DIR:"
ls -lh "$OUTPUT_DIR" | tail -n +2 | awk '{print "  " $9 " (" $5 ")"}'
echo ""

if [ $failed -eq 0 ]; then
    echo "All downloads complete!"
    exit 0
else
    echo "Some downloads failed. Check your internet connection and try again."
    exit 1
fi
