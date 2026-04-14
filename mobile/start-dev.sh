#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# start-dev.sh  —  BrickScan development server launcher (v2, IP-watching)
#
# Solves the recurring "Could not connect to development server" error by:
#   1. Auto-detecting the current USB link-local IP on startup
#   2. Exporting REACT_NATIVE_PACKAGER_HOSTNAME so Metro advertises that IP in
#      every manifest it serves
#   3. BACKGROUND WATCHER: monitors IP every 2 s while Metro is running. If the
#      USB cable is unplugged/replugged and the IP changes, the watcher kills
#      Metro so the outer loop respawns it with the new IP — the phone can
#      then reconnect without any manual URL entry.
#   4. Keeping Metro alive — auto-restarts if it crashes
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
#   ./start-dev.sh              # auto-detect IP, watch for changes
#   ./start-dev.sh 192.168.1.5  # force a specific IP (disables watcher)
#   ./start-dev.sh --no-watch   # auto-detect but disable the IP watcher
#   npm run dev                  # same as ./start-dev.sh (via package.json)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${METRO_PORT:-8081}"

# Argument parsing
FORCE_IP=""
WATCH_IP=1
for arg in "$@"; do
  case "$arg" in
    --no-watch) WATCH_IP=0 ;;
    -*)         echo "Unknown flag: $arg"; exit 2 ;;
    *)          FORCE_IP="$arg" ;;
  esac
done

# ── IP detection ─────────────────────────────────────────────────────────────
detect_ip() {
  local ip=""

  # Priority 1: USB link-local (169.254.x.x) on ANY interface.
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
  if [[ "$WATCH_IP" == "1" && -z "$FORCE_IP" ]]; then
    echo "  Watch: active — will auto-restart if USB IP changes"
  fi
  echo ""
  echo "  Phone can't connect?"
  echo "    • Re-plug USB → watcher detects IP change → Metro auto-restarts"
  echo "    • Or point phone at: exp://\$(scutil --get LocalHostName).local:$PORT"
  echo "      (mDNS hostname — works across IP changes without restart)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
}

# ── IP-change watcher ────────────────────────────────────────────────────────
# Runs in background while Metro is up. Polls the current IP every 2 s;
# if it changes, sends SIGTERM to Metro so the outer loop respawns it with
# REACT_NATIVE_PACKAGER_HOSTNAME set to the new IP.
watch_for_ip_change() {
  local metro_pid="$1"
  local baseline_ip="$2"
  local tick=0

  while kill -0 "$metro_pid" 2>/dev/null; do
    sleep 2
    tick=$((tick + 1))

    local current
    current="$(detect_ip)"

    # Don't trigger on localhost fallback (happens briefly during cable reconnect)
    [[ "$current" == "localhost" ]] && continue
    [[ "$current" == "$baseline_ip" ]] && continue

    echo ""
    echo "⚡ [watcher] IP changed: $baseline_ip → $current"
    echo "⚡ [watcher] Restarting Metro so new IP is in manifests..."

    # Kill entire Metro process group so we don't leave zombies
    kill -TERM "$metro_pid" 2>/dev/null || true
    # Graceful exit wait — Metro may need a moment
    for _ in 1 2 3 4 5; do
      kill -0 "$metro_pid" 2>/dev/null || break
      sleep 1
    done
    # Force kill if still alive
    kill -KILL "$metro_pid" 2>/dev/null || true
    return 0
  done
}

# ── Cleanup on script exit ───────────────────────────────────────────────────
cleanup() {
  # Kill background watcher if it's still around
  if [[ -n "${WATCHER_PID:-}" ]]; then
    kill "$WATCHER_PID" 2>/dev/null || true
  fi
  # Kill Metro if we have a PID for it
  if [[ -n "${METRO_PID:-}" ]]; then
    kill "$METRO_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# ── Main ─────────────────────────────────────────────────────────────────────
cd "$SCRIPT_DIR"

if [[ -n "$FORCE_IP" ]]; then
  HOST="$FORCE_IP"
  echo "ℹ  Using forced IP: $HOST"
  WATCH_IP=0  # no point watching when you've asked for a fixed IP
else
  HOST="$(detect_ip)"
  echo "ℹ  Detected host IP: $HOST"
fi

kill_stale_metro
print_banner "$HOST"

# ── Keepalive loop ───────────────────────────────────────────────────────────
RESTART_COUNT=0
MAX_RESTARTS=20

while true; do
  # Re-detect before every start — the cable may have been reconnected
  NEW_HOST="$(detect_ip)"
  if [[ "$NEW_HOST" != "$HOST" && "$NEW_HOST" != "localhost" ]]; then
    echo "⚡ IP updated: $HOST → $NEW_HOST"
    HOST="$NEW_HOST"
    print_banner "$HOST"
  fi

  export REACT_NATIVE_PACKAGER_HOSTNAME="$HOST"

  # Start Metro in the background so we can run the watcher alongside it
  npx expo start --port "$PORT" &
  METRO_PID=$!

  # Start the IP-change watcher (unless disabled)
  WATCHER_PID=""
  if [[ "$WATCH_IP" == "1" ]]; then
    watch_for_ip_change "$METRO_PID" "$HOST" &
    WATCHER_PID=$!
  fi

  # Wait for Metro to exit (killed by watcher, crash, or user Ctrl+C)
  wait "$METRO_PID" 2>/dev/null || true
  METRO_EXIT=$?

  # Clean up watcher
  if [[ -n "$WATCHER_PID" ]]; then
    kill "$WATCHER_PID" 2>/dev/null || true
    wait "$WATCHER_PID" 2>/dev/null || true
    WATCHER_PID=""
  fi
  METRO_PID=""

  RESTART_COUNT=$((RESTART_COUNT + 1))
  if [[ $RESTART_COUNT -ge $MAX_RESTARTS ]]; then
    echo "❌ Metro restarted $MAX_RESTARTS times. Giving up."
    exit 1
  fi

  echo ""
  echo "🔄 Metro exited (restart $RESTART_COUNT/$MAX_RESTARTS). Respawning in 2 s..."
  sleep 2
done
