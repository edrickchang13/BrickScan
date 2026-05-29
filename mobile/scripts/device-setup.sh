#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# device-setup.sh — one-time (idempotent) native toolchain prep for the
# Phase 4 on-device loop. Run this once per fresh checkout (and again whenever
# native deps change). After it succeeds, use device-build-install.sh to build
# + flash a tethered iPhone.
#
# What it does, in order:
#   1. npm install                       (JS deps, runs patch-package postinstall)
#   2. npx expo prebuild -p ios          (sync app.json/plugins → ios/ native project)
#   3. pod install                       (generates ios/BrickScan.xcworkspace)
#
# It does NOT touch signing — that's the one-time manual step you do in Xcode
# (open the workspace, pick your Team, let Xcode auto-provision). See
# mobile/DEVICE_LOOP.md "One-time signing setup".
#
# Usage:
#   mobile/scripts/device-setup.sh            # full setup
#   mobile/scripts/device-setup.sh --pods     # just re-run pod install
#   mobile/scripts/device-setup.sh --clean    # nuke ios/Pods + Podfile.lock-derived state first
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Resolve repo paths relative to this script (works from any cwd).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"   # absolute, survives the cd below
MOBILE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$MOBILE_DIR"

PODS_ONLY=0
CLEAN=0
for arg in "$@"; do
  case "$arg" in
    --pods)  PODS_ONLY=1 ;;
    --clean) CLEAN=1 ;;
    -h|--help) sed -n '2,30p' "$SELF"; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

say() { printf '\033[1;36m[device-setup]\033[0m %s\n' "$*"; }

# --- preflight: required tools ---
need() { command -v "$1" >/dev/null 2>&1 || { echo "MISSING: $1 — $2" >&2; exit 1; }; }
need node "install Node 18+ (brew install node)"
need pod  "install CocoaPods (sudo gem install cocoapods OR brew install cocoapods)"
need xcodebuild "install Xcode + run: xcode-select --install"

if [[ "$PODS_ONLY" -eq 0 ]]; then
  say "1/3 npm install"
  npm install

  say "2/3 expo prebuild (sync native ios/ project from app.json)"
  # --no-install: we run pod install ourselves below so we control the flags.
  npx expo prebuild -p ios --no-install
fi

if [[ "$CLEAN" -eq 1 ]]; then
  say "clean: removing ios/Pods and ios/build"
  rm -rf ios/Pods ios/build "$HOME/Library/Developer/Xcode/DerivedData/BrickScan-"* 2>/dev/null || true
fi

say "3/3 pod install (generates ios/BrickScan.xcworkspace)"
( cd ios && pod install )

say "done. Next:"
say "  • ONE-TIME signing: open ios/BrickScan.xcworkspace in Xcode, select the"
say "    BrickScan target → Signing & Capabilities → set your Team (auto-provision)."
say "  • Then flash a tethered iPhone:  mobile/scripts/device-build-install.sh"
