#!/usr/bin/env bash
# render_all.sh — Batch-render top 500 LEGO parts via Blender
#
# Usage:
#   bash render_all.sh [--parts-csv PATH] [--ldraw-dir PATH] \
#                      [--output-dir PATH] [--jobs N] [--top N]
#
# Defaults:
#   --parts-csv  ~/Documents/Claude/Projects/Lego/brickscan/ml/training_data/rebrickable_csv/parts.csv
#   --ldraw-dir  ~/Documents/Claude/Projects/Lego/ldraw
#   --output-dir ~/Documents/Claude/Projects/Lego/synthetic_dataset
#   --jobs       4   (parallel Blender processes)
#   --top        500 (render top N parts by set frequency)

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"
RENDER_SCRIPT="$SCRIPT_DIR/blender_render.py"
PARTS_CSV="$HOME/Documents/Claude/Projects/Lego/brickscan/ml/training_data/rebrickable_csv/parts.csv"
LDRAW_DIR="$HOME/Documents/Claude/Projects/Lego/ldraw"
OUTPUT_DIR="$HOME/Documents/Claude/Projects/Lego/synthetic_dataset"
JOBS=4
TOP=500
COLORS=(4 1 2 14 15 7 19 0 25 71)   # Red, Blue, Green, Yellow, White, LGray, Tan, Black, Orange, LBluishGray

# ── Parse args ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --parts-csv)  PARTS_CSV="$2";  shift 2 ;;
        --ldraw-dir)  LDRAW_DIR="$2";  shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --jobs)       JOBS="$2";       shift 2 ;;
        --top)        TOP="$2";        shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Preflight checks ─────────────────────────────────────────────────────────
if [[ ! -f "$BLENDER" ]]; then
    echo "ERROR: Blender not found at $BLENDER"
    exit 1
fi
if [[ ! -f "$RENDER_SCRIPT" ]]; then
    echo "ERROR: blender_render.py not found at $RENDER_SCRIPT"
    exit 1
fi
if [[ ! -f "$PARTS_CSV" ]]; then
    echo "ERROR: parts.csv not found at $PARTS_CSV"
    echo "  Download from https://rebrickable.com/downloads/ → parts.csv"
    exit 1
fi
if [[ ! -d "$LDRAW_DIR/parts" ]]; then
    echo "ERROR: LDraw library not found at $LDRAW_DIR/parts"
    echo "  Run: bash setup_ldraw.sh first"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/render_all.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "  BrickScan Batch Renderer"
echo "  $(date)"
echo "  Top $TOP parts | $JOBS parallel jobs"
echo "  Output: $OUTPUT_DIR"
echo "============================================================"

# ── Extract top N part numbers from parts.csv ────────────────────────────────
# parts.csv columns: part_num,name,part_cat_id,part_material
# We just take the first $TOP lines (sorted by assumed frequency in the CSV)
TOP_PARTS_FILE=$(mktemp /tmp/top_parts_XXXX.txt)

python3 - "$PARTS_CSV" "$TOP" "$TOP_PARTS_FILE" <<'PYEOF'
import csv, sys, re
from pathlib import Path

csv_path, top_n, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
parts = []
with open(csv_path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        part_num = row.get('part_num', '').strip()
        # Skip non-standard parts (prints, stickers, etc.)
        if re.search(r'[a-zA-Z]{3,}', part_num):
            continue
        parts.append(part_num)

# Deduplicate while preserving order
seen, unique = set(), []
for p in parts:
    if p not in seen:
        seen.add(p)
        unique.append(p)

with open(out_path, 'w') as f:
    for p in unique[:top_n]:
        f.write(p + '\n')

print(f"[setup] Wrote {min(len(unique), top_n)} part numbers to {out_path}")
PYEOF

TOTAL=$(wc -l < "$TOP_PARTS_FILE")
echo "Rendering $TOTAL parts × ${#COLORS[@]} colors × 36 angles × 5 lights × 3 zooms"
echo ""

# ── Render function (called in parallel) ─────────────────────────────────────
render_part_color() {
    local part_id="$1"
    local color="$2"
    local part_out="$OUTPUT_DIR/$part_id"
    local done_flag="$part_out/.done_color_${color}"

    if [[ -f "$done_flag" ]]; then
        echo "[skip] $part_id color=$color already complete"
        return 0
    fi

    "$BLENDER" --background --factory-startup \
        --python "$RENDER_SCRIPT" -- \
        --part-id    "$part_id" \
        --ldraw-dir  "$LDRAW_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --color      "$color" \
        --num-angles 36 \
        --num-lights 5 \
        --num-zooms  3 \
        --resolution 224 \
        2>&1 | grep -E '\[blender_render\]|ERROR|WARNING' \
        || true

    if ls "$part_out"/*_${color}_*.png &>/dev/null 2>&1 || \
       ls "$part_out"/*.png &>/dev/null 2>&1; then
        touch "$done_flag"
        echo "[done] $part_id color=$color"
    else
        echo "[fail] $part_id color=$color — check $OUTPUT_DIR/$part_id"
    fi
}

export -f render_part_color
export BLENDER OUTPUT_DIR LDRAW_DIR RENDER_SCRIPT

# ── Build job list ───────────────────────────────────────────────────────────
JOB_LIST=$(mktemp /tmp/job_list_XXXX.txt)
while IFS= read -r part_id; do
    for color in "${COLORS[@]}"; do
        echo "$part_id $color"
    done
done < "$TOP_PARTS_FILE" > "$JOB_LIST"

TOTAL_JOBS=$(wc -l < "$JOB_LIST")
echo "Total render tasks: $TOTAL_JOBS (parts × colors)"
echo ""

# ── Run in parallel using xargs ──────────────────────────────────────────────
cat "$JOB_LIST" | xargs -P "$JOBS" -L 1 bash -c 'render_part_color "$@"' _

rm -f "$TOP_PARTS_FILE" "$JOB_LIST"

# ── Write dataset_manifest.json ──────────────────────────────────────────────
echo ""
echo "Writing dataset_manifest.json..."
python3 - "$OUTPUT_DIR" <<'PYEOF'
import json, sys
from pathlib import Path

output_dir = Path(sys.argv[1])
manifest = {}

for part_dir in sorted(output_dir.iterdir()):
    if not part_dir.is_dir():
        continue
    images = sorted(str(p) for p in part_dir.glob('*.png'))
    if images:
        manifest[part_dir.name] = images

manifest_path = output_dir / 'dataset_manifest.json'
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)

total_images = sum(len(v) for v in manifest.values())
print(f"Manifest: {len(manifest)} parts, {total_images} total images → {manifest_path}")
PYEOF

echo ""
echo "============================================================"
echo "  Render complete: $(date)"
echo "============================================================"
