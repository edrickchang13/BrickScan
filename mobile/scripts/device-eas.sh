#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# device-eas.sh — CLOUD alternative to the local Xcode loop: build a signed iOS
# dev-client (or preview) via EAS, so you don't need to manage signing locally.
# EAS handles provisioning in the cloud; you install the resulting build over USB
# or via the QR/link. Use this when you'd rather not run device-build-install.sh.
#
# The project is already EAS-configured: app.json has the projectId, eas.json has
# `development` (dev client, internal), `preview` (release, internal), and
# `production` (store) profiles.
#
# One-time:  npm i -g eas-cli && eas login   (and `eas device:create` to register
#            your iPhone's UDID for ad-hoc/internal builds — see DEVICE_LOOP.md).
#
# Usage:
#   mobile/scripts/device-eas.sh                 # development profile (dev client)
#   mobile/scripts/device-eas.sh preview         # preview profile (release, internal)
#   mobile/scripts/device-eas.sh development --local   # build on THIS mac instead of the cloud
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"   # absolute, survives the cd below
MOBILE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$MOBILE_DIR"

PROFILE="${1:-development}"
case "$PROFILE" in
  development|preview|production) ;;
  -h|--help) sed -n '2,24p' "$SELF"; exit 0 ;;
  *) echo "unknown profile: $PROFILE (use development|preview|production)" >&2; exit 2 ;;
esac
shift || true
EXTRA_ARGS=("$@")  # e.g. --local, --no-wait

say() { printf '\033[1;36m[eas]\033[0m %s\n' "$*"; }

# Resolve an eas binary: global, or via npx (no global install required).
if command -v eas >/dev/null 2>&1; then
  EAS=(eas)
else
  say "eas-cli not found globally; using 'npx eas-cli' (install globally for speed: npm i -g eas-cli)"
  EAS=(npx eas-cli)
fi

say "building iOS profile '$PROFILE' ${EXTRA_ARGS[*]:-}"
"${EAS[@]}" build --platform ios --profile "$PROFILE" "${EXTRA_ARGS[@]}"

say "done. Install the build on your iPhone (EAS prints an install URL / QR; or"
say "use:  ${EAS[*]} build:run -p ios  to install the latest build over USB)."
