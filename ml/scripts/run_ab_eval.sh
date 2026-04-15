#!/usr/bin/env bash
# run_ab_eval.sh — sweep every relevant feature-flag combination through
# the eval runner and emit a comparison table.
#
# Takes a long time (a backend restart per config + ~N eval scans per
# config), so the defaults exercise only 4 configurations that span the
# interesting corners of the feature-flag space.
#
# Each config sets env vars in docker-compose via an --env-file override,
# restarts the backend, runs ml/scripts/eval_against_feedback.py with a
# unique --config-label, and collects the JSON result.
#
# At the end, it prints a comparison table and writes a combined report
# to ml/data/eval_results/ab_report_<timestamp>.md.
#
# Usage:
#   bash ml/scripts/run_ab_eval.sh              # defaults: 4 configs, 100 samples each
#   AB_LIMIT=200 bash ml/scripts/run_ab_eval.sh # 200 samples per config
#   AB_CONFIGS="baseline,grounded,all_on" bash run_ab_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

LIMIT="${AB_LIMIT:-100}"
BASE_URL="${AB_BASE_URL:-http://localhost:8000}"
EVAL_DIR="ml/data/eval_results"
mkdir -p "$EVAL_DIR"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
REPORT="$EVAL_DIR/ab_report_${TIMESTAMP}.md"

# Each config is: name|env-var-list
#   env vars can be SCAN_TTA_ENABLED, SCAN_ALWAYS_RUN_GEMINI,
#   SCAN_GROUNDED_GEMINI, SCAN_COLLAPSE_VARIANTS, SCAN_COLOR_RERANK
#   (all default "false" when not listed)
declare -a DEFAULT_CONFIGS=(
  "baseline|"
  "grounded|SCAN_GROUNDED_GEMINI=true"
  "collapse|SCAN_COLLAPSE_VARIANTS=true"
  "color|SCAN_COLOR_RERANK=true"
  "all_on|SCAN_GROUNDED_GEMINI=true SCAN_COLLAPSE_VARIANTS=true SCAN_COLOR_RERANK=true"
)

# Let user override via $AB_CONFIGS="baseline,grounded,all_on"
if [[ -n "${AB_CONFIGS:-}" ]]; then
  IFS=',' read -r -a REQUESTED <<< "$AB_CONFIGS"
  CONFIGS=()
  for req in "${REQUESTED[@]}"; do
    for c in "${DEFAULT_CONFIGS[@]}"; do
      [[ "${c%%|*}" == "$req" ]] && CONFIGS+=("$c")
    done
  done
else
  CONFIGS=("${DEFAULT_CONFIGS[@]}")
fi

echo "Configurations to run: ${#CONFIGS[@]}"
for c in "${CONFIGS[@]}"; do echo "  - ${c%%|*}"; done
echo

# Pre-flight: is the backend reachable?
if ! curl -sf -o /dev/null "$BASE_URL/health"; then
  echo "ERROR: backend not reachable at $BASE_URL" >&2
  echo "Start it with: docker-compose up -d backend" >&2
  exit 2
fi

# Pre-flight: is the eval set non-empty?
EVAL_ROWS=$(curl -s "$BASE_URL/api/local-inventory/feedback/eval-set.json?limit=5" | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))')
if [[ "$EVAL_ROWS" == "0" ]]; then
  echo "WARNING: eval set is empty. Use the feedback flywheel first (scan bricks + tap correct/alternative on ScanResultScreen)."
  echo "Aborting — nothing to evaluate."
  exit 3
fi
echo "Eval set has data (${EVAL_ROWS} sample rows visible)."

run_config() {
  local name="$1" envs="$2"
  echo
  echo "=================================================================="
  echo "Configuration: $name"
  echo "Env vars:      ${envs:-<none>}"
  echo "=================================================================="

  # Write a temporary env override file
  local env_override="/tmp/ab_env_${name}.env"
  : > "$env_override"
  for v in SCAN_TTA_ENABLED SCAN_ALWAYS_RUN_GEMINI SCAN_GROUNDED_GEMINI SCAN_COLLAPSE_VARIANTS SCAN_COLOR_RERANK; do
    if echo "$envs" | grep -q "${v}=true"; then
      echo "${v}=true" >> "$env_override"
    else
      echo "${v}=false" >> "$env_override"
    fi
  done

  # Restart backend with these env overrides. docker-compose supports
  # --env-file but only at 'up' time; we use --force-recreate + override.
  docker-compose --env-file "$env_override" up -d --force-recreate backend 2>&1 | tail -1
  sleep 3
  # Wait for health
  for i in 1 2 3 4 5 6 7 8 9 10; do
    if curl -sf -o /dev/null "$BASE_URL/health"; then break; fi
    sleep 2
  done

  # Run the eval
  ./backend/venv/bin/python3 ml/scripts/eval_against_feedback.py \
      --base-url "$BASE_URL" \
      --limit "$LIMIT" \
      --config-label "$name" \
      --output-dir "$EVAL_DIR" 2>&1 | grep -E "^\s*(Top-1|Top-3|Config|Rows|  [a-z])" | head -20
  echo
}

for c in "${CONFIGS[@]}"; do
  run_config "${c%%|*}" "${c#*|}"
done

# Build the comparison report
{
  echo "# A/B evaluation report"
  echo ""
  echo "- Date: $(date -u)"
  echo "- Backend: $BASE_URL"
  echo "- Samples per config: $LIMIT"
  echo ""
  echo "## Results"
  echo ""
  echo "| Config | Rows | Top-1 | Top-3 |"
  echo "|--------|------|-------|-------|"
  for c in "${CONFIGS[@]}"; do
    name="${c%%|*}"
    latest=$(ls -t "$EVAL_DIR"/*_"$name".json 2>/dev/null | head -1 || true)
    if [[ -z "$latest" ]]; then
      echo "| $name | (missing) | - | - |"
      continue
    fi
    python3 - "$latest" "$name" <<'PY'
import json, sys
path, name = sys.argv[1], sys.argv[2]
d = json.load(open(path))
print(f"| {name} | {d['total_evaluated']} | {d['top1_accuracy']*100:.1f}% | {d['top3_accuracy']*100:.1f}% |")
PY
  done
  echo ""
  echo "## Notes"
  echo ""
  echo "- Each row shows accuracy on the same eval set with the named flag combination."
  echo "- Deltas > ~2% are likely meaningful given sample size; below that is noise."
  echo "- Raw per-config JSON is in \`$EVAL_DIR/*_<name>.json\`."
} > "$REPORT"

echo
echo "Report: $REPORT"
cat "$REPORT"
