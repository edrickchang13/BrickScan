#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# start-dev.sh  —  BrickScan development server launcher
#
# Solves the recurring "Could not connect to development server" error by:
#   1. Auto-detecting the current USB link-local IP (any interface, not just en6)
#   2. Setting REACT_NATIVE_PACKAGER_HOSTNAME so Metro advertises the right IP
#   3. Keeping Metro alive — auto-restarts if it crashes, re-detects IP each time
#   4. Printing the connection URL for easy reference
#
# Detection priority:
#   1. Any interface with a 169.254.x.x link-local IP  (USB tether — most reliable)
#   2. Any en* interface with a routable LAN IP         (WiFi / Ethernet)
#   3. Any non-loopback IPv4                            (last resort)
#   4. localhost                                        (simulator-only fallback)
#
# Why interface-agnostic? macOS assigns USB tether to whichever en* slot is
# free at plug-in time (en1, en4, en6, en8 — it varies). Hardcoding en6 breaks
# whenever the OS picks a different slot.
#
# Usage:
#   ./start-dev.sh              # auto-detect IP
#   ./start-dev.sh 192.168.1.5  # force a specific IP
#   npm run dev                  # same as ./start-dev.sh (via package.json)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${METRO_PORT:-8081}"
FORCE_IP="${1:-}"

# ── IP detection ─────────────────────────────────────────────────────────────
# Returns the best IP for Metro to bind/advertise.
detect_ip() {
  local ip=""

  # Priority 1: USB link-local (169.254.x.x) on ANY interface.
  # The OS assigns this dynamically on each cable plug — the interface name
  # (en1/en4/en6/en8) changes, but the 169.254 subnet is always the marker.
  ip=$(ifconfig 2>/dev/null \
    | grep "inet 169\.254\." \
    | head -1 \
    | awk '{print $2}' || true)
  [[ -n "$ip" ]] && { echo "$ip"; return; }

  # Priority 2: Any routable LAN IP on an en* interface (WiFi / USB Ethernet).
  for iface in en0 en1 en2 en3 en4 en5 en6 en7 en8 en9; do
    ip=$(ifconfig "$iface" 2>/dev/null \
      | grep "inet " \
      | grep -v "127\.\|169\.254\." \
      | head -1 \
      | awk '{print $2}' || true)
    [[ -n "$ip" ]] && { echo "$ip"; return; }
  done

  # Priority 3: Any non-loopback, non-link-local IPv4
  ip=$(ifconfig 2>/dev/null \
    | grep "inet " \
    | grep -v "127\.\|169\.254\." \
    | head -1 \
    | awk '{print $2}' || true)
  [[ -n "$ip" ]] && { echo "$ip"; return; }

  echo "localhost"
}

# Show which interface owns the detected IP (for diagnostics)
detect_iface_for_ip() {
  local target="$1"
  ifconfig 2>/dev/null \
    | awk -v ip="$target" '
        /^[a-z]/ { iface = $1 }
        /inet / && $2 == ip { print iface; exit }
    ' || true
}

# ── Kill any stale Metro on the port ────────────────────────────────────────
kill_stale_metro() {
  local pids
  pids=$(lsof -ti tcp:"$PORT" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    echo "⚠  Killing stale Metro on port $PORT (PIDs: $pids)..."
    echo "$pids" | xargs kill 2>/dev/null || true
    sleep 1
  fi
}

# ── Banner ───────────────────────────────────────────────────────────────────
print_banner() {
  local host="$1"
  local iface
  iface=$(detect_iface_for_ip "$host")
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  BrickScan Metro"
  echo "  Host : $host${iface:+ (via $iface)}"
  echo "  URL  : exp://$host:$PORT"
  echo ""
  echo "  If the app can't connect:"
  echo "    1. Re-plug the USB cable, then run: npm run dev"
  echo "    2. Or on the iPhone: Settings → Developer → Enable UI Automation"
  echo "       then rebuild from Xcode"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
}

# ── Main ─────────────────────────────────────────────────────────────────────
cd "$SCRIPT_DIR"

if [[ -n "$FORCE_IP" ]]; then
  HOST="$FORCE_IP"
  echo "ℹ  Using forced IP: $HOST"
else
  HOST="$(detect_ip)"
  echo "ℹ  Detected host IP: $HOST"
fi

kill_stale_metro
print_banner "$HOST"

# ── Keepalive loop: restart Metro on crash, re-detect IP each time ───────────
RESTART_COUNT=0
MAX_RESTARTS=20   # generous; each restart re-detects the current USB IP

while true; do
  # Re-detect before every start — the cable may have been reconnected
  NEW_HOST="$(detect_ip)"
  if [[ "$NEW_HOST" != "$HOST" && "$NEW_HOST" != "localhost" ]]; then
    echo "⚡ IP changed: $HOST → $NEW_HOST  (cable reconnected?)"
    HOST="$NEW_HOST"
    print_banner "$HOST"
  fi

  export REACT_NATIVE_PACKAGER_HOSTNAME="$HOST"
  npx expo start --port "$PORT" || true

  RESTART_COUNT=$((RESTART_COUNT + 1))
  if [[ $RESTART_COUNT -ge $MAX_RESTARTS ]]; then
    echo "❌ Metro crashed $MAX_RESTARTS times consecutively. Giving up."
    exit 1
  fi

  # Re-detect IP immediately after exit (before sleeping)
  HOST="$(detect_ip)"
  echo ""
  echo "🔄 Metro exited (restart $RESTART_COUNT/$MAX_RESTARTS). Restarting in 3 s with IP: $HOST..."
  sleep 3
done
