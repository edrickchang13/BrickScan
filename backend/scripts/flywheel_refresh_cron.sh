#!/usr/bin/env bash
# flywheel_refresh_cron.sh — the OPERATIONAL flywheel loop (see ml/FLYWHEEL.md §OPS).
#
# FAST cadence (default, e.g. every 15 min): fold the confirmed-feedback backlog
# into the galleries via the SAME hot path a live /flywheel/confirm uses — NO
# retraining — then journal a monitoring snapshot. Idempotent + resumable: each
# appended row is flipped used_for_training=True server-side, so overlapping or
# repeated runs never double-ingest.
#
#   1. POST /api/local-inventory/flywheel/refresh   (append confirmed backlog)
#   2. POST /api/local-inventory/flywheel/metrics/snapshot   (coverage/volume/confusions → disk)
#
# SLOW cadence (--heavy, e.g. weekly): additionally freeze an accuracy snapshot
# and, if ml/scripts/active_learning_cron.sh exists, hand off to the heavier
# re-embed / student-distillation ("Track D"). The fast loop already raises
# accuracy without this; the heavy pass just compresses the grown gallery back
# into the backbone occasionally.
#
#   3. POST /api/local-inventory/feedback/snapshot   (weekly top1/top3 trend point)
#   4. bash ml/scripts/active_learning_cron.sh       (if present; heavy GPU job)
#
# Schedule with a host cron (macOS launchd plist + Linux systemd timer templates
# ship alongside this file in backend/scripts/):
#   # fast append loop, every 15 minutes
#   */15 * * * *  bash /path/to/brickscan/backend/scripts/flywheel_refresh_cron.sh
#   # heavy refresh, Mondays 3am
#   0 3 * * 1     bash /path/to/brickscan/backend/scripts/flywheel_refresh_cron.sh --heavy
#
# Env vars:
#   BACKEND_URL        (default http://localhost:8000)
#   ADMIN_AUTH_TOKEN   (optional Bearer token if the endpoints require auth)
#   REFRESH_LIMIT      (default 1000 — max backlog rows folded per pass)
#   ONLY_CORRECTIONS   (default false — set true to replay only model-was-wrong rows)
#   WINDOW_DAYS        (default 30 — accuracy-snapshot window on --heavy)
#   HEAVY_SCRIPT       (default <repo>/ml/scripts/active_learning_cron.sh)
#   FLYWHEEL_LOG       (optional path to append a run log line to)

set -euo pipefail

# ── args ──────────────────────────────────────────────────────────────────────
HEAVY=0
for arg in "$@"; do
  case "$arg" in
    --heavy) HEAVY=1 ;;
    -h|--help)
      sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg (use --heavy or --help)" >&2; exit 2 ;;
  esac
done

# ── config ──────────────────────────────────────────────────────────────────────
BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"
API="${BACKEND_URL%/}/api/local-inventory"
REFRESH_LIMIT="${REFRESH_LIMIT:-1000}"
ONLY_CORRECTIONS="${ONLY_CORRECTIONS:-false}"
WINDOW_DAYS="${WINDOW_DAYS:-30}"

# This script lives in backend/scripts/; the repo root is two levels up.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
HEAVY_SCRIPT="${HEAVY_SCRIPT:-${_REPO_ROOT}/ml/scripts/active_learning_cron.sh}"

headers=(-H "Content-Type: application/json")
if [[ -n "${ADMIN_AUTH_TOKEN:-}" ]]; then
  headers+=(-H "Authorization: Bearer ${ADMIN_AUTH_TOKEN}")
fi

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() {
  echo "[$(ts)] $*"
  if [[ -n "${FLYWHEEL_LOG:-}" ]]; then echo "[$(ts)] $*" >>"${FLYWHEEL_LOG}"; fi
}

# POST helper: prints the response body, fails the script on a non-2xx so the
# cron MAILTO surfaces it. Uses -sS so curl is quiet but still reports errors.
post() {
  local url="$1"
  curl -sS -f -X POST "${headers[@]}" "${url}"
}

log "flywheel refresh start (heavy=${HEAVY}) → ${API}"

# ── 1. FAST: fold the confirmed-feedback backlog into the galleries ─────────────
refresh_url="${API}/flywheel/refresh?limit=${REFRESH_LIMIT}&only_corrections=${ONLY_CORRECTIONS}"
log "append backlog → ${refresh_url}"
if refresh_resp="$(post "${refresh_url}")"; then
  log "refresh OK: ${refresh_resp}"
else
  log "ERROR: /flywheel/refresh failed"
  exit 1
fi

# ── 2. FAST: journal a monitoring snapshot (coverage / volume / confusions) ─────
log "metrics snapshot → ${API}/flywheel/metrics/snapshot"
if metrics_resp="$(post "${API}/flywheel/metrics/snapshot")"; then
  log "metrics OK: ${metrics_resp}"
else
  # Monitoring is non-fatal — a failed snapshot must not fail the append loop.
  log "WARN: /flywheel/metrics/snapshot failed (non-fatal)"
fi

# ── 3 & 4. SLOW (--heavy only): accuracy snapshot + heavy re-embed handoff ──────
if [[ "${HEAVY}" -eq 1 ]]; then
  log "heavy pass: accuracy snapshot → ${API}/feedback/snapshot?window_days=${WINDOW_DAYS}"
  if snap_resp="$(post "${API}/feedback/snapshot?window_days=${WINDOW_DAYS}")"; then
    log "accuracy snapshot OK: ${snap_resp}"
  else
    log "WARN: /feedback/snapshot failed (non-fatal)"
  fi

  if [[ -f "${HEAVY_SCRIPT}" ]]; then
    log "heavy refresh: delegating to ${HEAVY_SCRIPT}"
    # The heavy job (re-embed / distillation) is a GPU workload; run it inline
    # on the --heavy cadence only. It is responsible for its own resource limits.
    if bash "${HEAVY_SCRIPT}"; then
      log "heavy refresh OK"
    else
      log "WARN: heavy refresh script exited non-zero"
    fi
  else
    log "heavy refresh script not present (${HEAVY_SCRIPT}) — skipping; fast append loop already improved accuracy"
  fi
fi

log "flywheel refresh done (heavy=${HEAVY})"
