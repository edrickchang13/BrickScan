#!/usr/bin/env bash
# weekly_eval_cron.sh — freeze a feedback accuracy snapshot.
#
# Runs POST /api/local-inventory/feedback/snapshot which computes the
# rolling 30-day top-1 / top-3 accuracy + per-source breakdown and persists
# it to feedback_eval_snapshots. The FeedbackStatsScreen trend chart reads
# these snapshots.
#
# Schedule with a host cron (macOS launchd plist works too):
#   # Every Monday 3am local time
#   0 3 * * 1  bash /path/to/brickscan/backend/scripts/weekly_eval_cron.sh
#
# Env vars:
#   BACKEND_URL       (default http://localhost:8000)
#   ADMIN_AUTH_TOKEN  (optional — if your /feedback/snapshot requires auth)
#   WINDOW_DAYS       (default 30)

set -euo pipefail

BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"
WINDOW_DAYS="${WINDOW_DAYS:-30}"
ENDPOINT="${BACKEND_URL%/}/api/local-inventory/feedback/snapshot?window_days=${WINDOW_DAYS}"

headers=(-H "Content-Type: application/json")
if [[ -n "${ADMIN_AUTH_TOKEN:-}" ]]; then
  headers+=(-H "Authorization: Bearer ${ADMIN_AUTH_TOKEN}")
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Snapshot → ${ENDPOINT}"
response="$(curl -sf -X POST "${headers[@]}" "${ENDPOINT}")" || {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ERROR: snapshot failed" >&2
  exit 1
}

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] OK"
echo "${response}"
