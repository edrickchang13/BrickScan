#!/usr/bin/env bash
# render_dr_pilot.sh — runs blender_render_dr.py on the 50-part pilot list
# with full domain randomization enabled. Produces a complementary corpus
# to the existing synthetic_dataset/, using HDRI backgrounds + material
# jitter for a tighter sim-to-real fit.
#
# Runtime: roughly 2-3 minutes per part on Mac CPU → ~2 hrs for 50 parts
# at default 540 renders/part. Resumable: the renderer skips any part that
# already has its full render count in the output directory.
#
# Launch it in the background — it'll chug along without supervision:
#   bash ml/scripts/render_dr_pilot.sh &
#
# Keep an eye on it with:
#   tail -f ml/data/synthetic_dataset_dr_pilot/render.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PARTS_FILE="${PARTS_FILE:-ml/data/pilot_parts.txt}"
LDRAW_DIR="${LDRAW_DIR:-$REPO_ROOT/ml/data/ldraw/ldraw}"
HDRI_DIR="${HDRI_DIR:-$REPO_ROOT/ml/blender/hdris}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/ml/data/synthetic_dataset_dr_pilot}"
RESOLUTION="${RESOLUTION:-224}"
NUM_ANGLES="${NUM_ANGLES:-36}"
NUM_LIGHTS="${NUM_LIGHTS:-5}"
NUM_ZOOMS="${NUM_ZOOMS:-3}"

BLENDER="${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}"
RENDER_SCRIPT="$REPO_ROOT/ml/blender/blender_render_dr.py"

mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/render.log"
: > "$LOG"

echo "=== DR render pilot ===" | tee -a "$LOG"
echo "Started: $(date -u)" | tee -a "$LOG"
echo "Parts:   $PARTS_FILE" | tee -a "$LOG"
echo "LDraw:   $LDRAW_DIR" | tee -a "$LOG"
echo "HDRIs:   $HDRI_DIR ($(ls "$HDRI_DIR"/*.hdr 2>/dev/null | wc -l | tr -d ' ') files)" | tee -a "$LOG"
echo "Output:  $OUTPUT_DIR" | tee -a "$LOG"
echo "Size:    ${RESOLUTION}px, ${NUM_ANGLES}a × ${NUM_LIGHTS}l × ${NUM_ZOOMS}z = $((NUM_ANGLES * NUM_LIGHTS * NUM_ZOOMS)) renders/part" | tee -a "$LOG"

# Sanity checks
for required in "$BLENDER" "$RENDER_SCRIPT" "$LDRAW_DIR" "$PARTS_FILE"; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: missing $required" | tee -a "$LOG"
    exit 2
  fi
done

total=0
skipped=0
rendered=0
start=$(date +%s)

while IFS= read -r part; do
  # Skip comments + blank lines
  [[ "$part" =~ ^#.*$ ]] && continue
  [[ -z "$(echo "$part" | tr -d '[:space:]')" ]] && continue
  part="$(echo "$part" | tr -d '[:space:]')"
  total=$((total + 1))

  part_dir="$OUTPUT_DIR/$part"
  expected=$((NUM_ANGLES * NUM_LIGHTS * NUM_ZOOMS))
  actual=0
  if [[ -d "$part_dir" ]]; then
    actual=$(ls "$part_dir"/*.png 2>/dev/null | wc -l | tr -d ' ')
  fi
  if [[ "$actual" -ge "$expected" ]]; then
    echo "[$total] $part — skipping (already has $actual/$expected renders)" | tee -a "$LOG"
    skipped=$((skipped + 1))
    continue
  fi

  echo "[$total] $part — rendering ($actual/$expected existing)…" | tee -a "$LOG"
  part_start=$(date +%s)

  "$BLENDER" --background --factory-startup \
      --python "$RENDER_SCRIPT" -- \
      --part-id "$part" \
      --ldraw-dir "$LDRAW_DIR" \
      --output-dir "$OUTPUT_DIR" \
      --num-angles "$NUM_ANGLES" \
      --num-lights "$NUM_LIGHTS" \
      --num-zooms "$NUM_ZOOMS" \
      --resolution "$RESOLUTION" \
      --hdri-dir "$HDRI_DIR" \
      --domain-randomize \
      >> "$LOG" 2>&1 || echo "  (render errored — continuing)" | tee -a "$LOG"

  part_elapsed=$(( $(date +%s) - part_start ))
  rendered=$((rendered + 1))
  echo "  → done in ${part_elapsed}s" | tee -a "$LOG"
done < "$PARTS_FILE"

elapsed=$(( $(date +%s) - start ))
echo "" | tee -a "$LOG"
echo "=== Pilot complete ===" | tee -a "$LOG"
echo "Total parts in list:  $total" | tee -a "$LOG"
echo "Skipped (already done): $skipped" | tee -a "$LOG"
echo "Rendered this run:    $rendered" | tee -a "$LOG"
echo "Wall-clock:           ${elapsed}s ($((elapsed / 60)) min)" | tee -a "$LOG"
