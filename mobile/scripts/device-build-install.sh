#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# device-build-install.sh — build the BrickScan iOS app for a TETHERED iPhone
# and install + launch it, via xcodebuild + xcrun devicectl. This is the OUTER
# loop's "deploy" step for the Phase 4 closed-loop eval: flash a real device,
# run a live sweep, and (with the liveScanTelemetry debug flag on) collect the
# session telemetry that diffs against the inner-loop harness.
#
# Prereqs (done once — see mobile/DEVICE_LOOP.md):
#   • mobile/scripts/device-setup.sh has been run (ios/BrickScan.xcworkspace exists)
#   • Signing configured in Xcode (your Apple Team selected on the BrickScan target)
#   • iPhone plugged in, unlocked, "Trust This Computer" accepted, Developer Mode on
#   • Metro running for a dev build (npm run dev) — or use --release for standalone
#
# Usage:
#   mobile/scripts/device-build-install.sh                  # Debug build → first connected device
#   mobile/scripts/device-build-install.sh --release        # Release configuration
#   mobile/scripts/device-build-install.sh --device <UDID>  # target a specific device
#   mobile/scripts/device-build-install.sh --build-only      # build, don't install/launch
#   mobile/scripts/device-build-install.sh --no-launch       # install but don't launch
#
# It auto-detects: the workspace, the scheme, the first connected & available
# device, and the built .app path (from xcodebuild's settings, not a guess).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"   # absolute, survives the cd below
MOBILE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$MOBILE_DIR"

WORKSPACE="ios/BrickScan.xcworkspace"
SCHEME="BrickScan"
CONFIG="Debug"
BUNDLE_ID="com.edrickchang.brickscan"
DEVICE_UDID=""
DO_INSTALL=1
DO_LAUNCH=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --release)    CONFIG="Release" ;;
    --debug)      CONFIG="Debug" ;;
    --device)     DEVICE_UDID="${2:-}"; shift ;;
    --scheme)     SCHEME="${2:-}"; shift ;;
    --build-only) DO_INSTALL=0; DO_LAUNCH=0 ;;
    --no-launch)  DO_LAUNCH=0 ;;
    -h|--help)    sed -n '2,30p' "$SELF"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

say()  { printf '\033[1;36m[device]\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m[device]\033[0m %s\n' "$*" >&2; }

command -v xcodebuild >/dev/null || { err "xcodebuild not found — install Xcode."; exit 1; }
command -v xcrun       >/dev/null || { err "xcrun not found — install Xcode CLT."; exit 1; }

if [[ ! -e "$WORKSPACE" ]]; then
  err "$WORKSPACE not found. Run mobile/scripts/device-setup.sh first (it runs pod install)."
  exit 1
fi

# --- 1. Resolve the target device ----------------------------------------------
# devicectl writes the table to stdout and the structured JSON to --json-output,
# so we point it at a temp file and parse THAT (parsing stdout would choke on the
# table). Heuristic: prefer a currently-available/connected device, else the
# first paired one, else the first listed.
if [[ -z "$DEVICE_UDID" ]]; then
  say "detecting connected device…"
  DEV_JSON_FILE="$(mktemp -t bs_devices.XXXXXX.json)"
  trap 'rm -f "$DEV_JSON_FILE"' EXIT
  xcrun devicectl list devices --json-output "$DEV_JSON_FILE" >/dev/null 2>&1 || true
  DEVICE_UDID="$(python3 - "$DEV_JSON_FILE" <<'PY' 2>/dev/null || true
import json,sys
try:
    d=json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
devs=d.get("result",{}).get("devices",[])
def conn(x): return x.get("connectionProperties") or {}
# 1) connected/available right now (wired or active tunnel)
for x in devs:
    c=conn(x)
    if c.get("tunnelState")=="connected" or c.get("transportType") in ("wired","localNetwork"):
        print(x.get("identifier","")); sys.exit(0)
# 2) first PAIRED device (known to this mac; just needs to be plugged in/woken)
for x in devs:
    if conn(x).get("pairingState")=="paired":
        print(x.get("identifier","")); sys.exit(0)
# 3) anything at all
if devs: print(devs[0].get("identifier",""))
PY
)"
fi

if [[ -z "$DEVICE_UDID" ]]; then
  err "no tethered device found. Plug in your iPhone, unlock it, accept 'Trust This Computer',"
  err "and ensure Developer Mode is on (Settings → Privacy & Security → Developer Mode)."
  err "List devices with:  xcrun devicectl list devices"
  exit 1
fi
say "target device: $DEVICE_UDID"

# --- 2. Build for the device (generic iOS device destination) ------------------
DERIVED="$MOBILE_DIR/ios/build"
say "building $SCHEME ($CONFIG)…  (first build ~6-8 min; incremental ~30s-2min)"
set -x
xcodebuild \
  -workspace "$WORKSPACE" \
  -scheme "$SCHEME" \
  -configuration "$CONFIG" \
  -destination "id=$DEVICE_UDID" \
  -derivedDataPath "$DERIVED" \
  -allowProvisioningUpdates \
  build
set +x

# --- 3. Find the built .app from xcodebuild's own settings (no guessing) -------
APP_PATH="$(xcodebuild -workspace "$WORKSPACE" -scheme "$SCHEME" -configuration "$CONFIG" \
  -destination "id=$DEVICE_UDID" -derivedDataPath "$DERIVED" \
  -showBuildSettings 2>/dev/null \
  | awk -F' = ' '/ BUILT_PRODUCTS_DIR /{d=$2} / FULL_PRODUCT_NAME /{n=$2} END{if(d&&n) print d"/"n}')"

if [[ -z "$APP_PATH" || ! -d "$APP_PATH" ]]; then
  # Fallback: search DerivedData for the .app.
  APP_PATH="$(find "$DERIVED/Build/Products/$CONFIG-iphoneos" -maxdepth 1 -name '*.app' 2>/dev/null | head -1)"
fi
if [[ -z "$APP_PATH" || ! -d "$APP_PATH" ]]; then
  err "could not locate the built .app under $DERIVED. Build may have failed."
  exit 1
fi
say "built: $APP_PATH"

if [[ "$DO_INSTALL" -eq 0 ]]; then
  say "build-only: skipping install."
  exit 0
fi

# --- 4. Install + launch via devicectl -----------------------------------------
say "installing onto device…"
xcrun devicectl device install app --device "$DEVICE_UDID" "$APP_PATH"

if [[ "$DO_LAUNCH" -eq 1 ]]; then
  say "launching $BUNDLE_ID…"
  xcrun devicectl device process launch --device "$DEVICE_UDID" "$BUNDLE_ID" || \
    err "launch failed (the app is installed — open it from the home screen)."
fi

say "done."
say "Live-scan telemetry: enable the 'liveScanTelemetry' flag in Settings, run a"
say "sweep, tap Done. The session JSON lands in the app document dir and (unless"
say "local-only) POSTs to /api/local-inventory/telemetry/livescan."
