# BrickScan — Development Guide

This doc captures the working setup and the **single gotcha** that ate most of the
upgrade session. Read it before debugging anything network-related.

---

## ⚠️ The one thing that keeps breaking: USB link-local IP drift

**The USB IP between your Mac and iPhone changes every time you unplug and replug
the cable.** Not sometimes — every time. This is how macOS assigns link-local
addresses (APIPA) and it's not something you can "configure away."

Example of what happens:

```
Plug phone in at 10am  →  USB IP becomes 169.254.16.163
Unplug at noon          →  interface dies
Replug at noon-thirty   →  macOS assigns 169.254.27.158 (different IP)
```

Every IP-referencing piece of the stack that was "baked" at 10am now points at
a dead address:

| What's baked with the old IP | Symptom when IP changes |
|---|---|
| `mobile/.env.local` (`EXPO_PUBLIC_API_URL=http://<old-ip>:8000`) | Phone POSTs to scan endpoint → ECONNABORTED timeout after 60s |
| Metro's `REACT_NATIVE_PACKAGER_HOSTNAME` env var | Metro serves manifests that reference the old IP — phone can't download bundle |
| expo-dev-client's saved URL in NSUserDefaults | Red "Could not connect to development server" screen with the old IP shown |
| Xcode build artifacts? | ❌ NOT affected — native binary is IP-independent |
| Backend (Docker) | ❌ NOT affected — listens on 0.0.0.0, reachable on any interface |

**This was THE thing that wasted hours during the SDK 51 → 55 upgrade.** Every
"the app can't connect" issue traced back to a stale IP somewhere.

---

## How the current setup defends against IP drift

Three layers of protection so you should rarely have to think about this:

### 1. `start-dev.sh` has a background IP watcher

When you run `npm run dev`, the script:
- Detects the current USB IP
- Exports `REACT_NATIVE_PACKAGER_HOSTNAME` so Metro serves manifests with that IP
- **Launches a background watcher** that polls every 2 seconds
- If the IP changes, kills Metro → outer loop respawns with the new IP

**What to do on IP change:** nothing. Just wait 2–4 seconds after replug; Metro
auto-restarts. You'll see `⚡ [watcher] IP changed: X → Y` in the terminal.

### 2. `AppDelegate.swift` purges stale dev-launcher URLs on cold launch

On every cold start (DEBUG builds only), `AppDelegate.swift` wipes all
`NSUserDefaults` keys that expo-dev-client uses to remember URLs
(`EXDevLauncher*`, `RCTDevMenu*`, etc.). This way the dev-launcher can never
get stuck on a stale URL from a previous session.

**What to do:** force-quit the app on the phone and relaunch after IP change.

### 3. `.env.local` is the single source of truth for the API URL

Because `NativeModules.SourceCode.scriptURL` is empty in SDK 55 + RN 0.83 +
expo-dev-client, the runtime Metro-host discovery in `config.ts` doesn't work.
Instead, the mobile app always reads `EXPO_PUBLIC_API_URL` from
`mobile/.env.local`. Put your current Mac-reachable URL there.

Current contents of `.env.local` (raw USB IP, stable per-plug):

```
EXPO_PUBLIC_API_URL=http://169.254.27.158:8000
```

**If USB IP changes, update this line and restart Metro** (`Ctrl+C`, then `npm run dev`).

Alternative (works when mDNS is reachable, stable across IP changes):

```
EXPO_PUBLIC_API_URL=http://Edricks-MacBook-Air.local:8000
```

Requires Local Network permission granted on the phone (it is). Flakier on some
corporate WiFi networks that block mDNS.

---

## Checklist when the app can't connect

Run these in order. First one that shows a mismatch is your problem:

### 1. Is the USB IP current?

```bash
ifconfig | grep "inet 169.254"
```

Note the number. Compare against what's in `mobile/.env.local`. If different → update `.env.local`.

### 2. Is Metro running on that IP?

```bash
lsof -ti tcp:8081
curl -s http://<current-usb-ip>:8081/status
```

Should print `packager-status:running`. If not, restart Metro.

### 3. Is the backend reachable on that IP?

```bash
curl -s http://<current-usb-ip>:8000/health
```

Should print `{"status":"ok"}`. If not, check Docker:

```bash
cd /Users/edrickchang/Documents/Claude/Projects/Lego/brickscan
docker-compose ps
docker-compose restart backend   # if needed
```

### 4. Is the phone reaching the backend?

On the phone, open **Safari**, navigate to:

```
http://<current-usb-ip>:8000/health
```

If Safari shows `{"status":"ok"}` but the app still fails → `.env.local` is
pointing somewhere else. Check the Metro log line
`[Config] API_BASE_URL (from EXPO_PUBLIC_API_URL): ...` — that tells you what
URL the app is actually using.

### 5. Is iOS Local Network permission on?

**Settings → BrickScan → Local Network → ON**.

### 6. Nuclear option

```bash
# Kill everything, start fresh
cd /Users/edrickchang/Documents/Claude/Projects/Lego/brickscan
docker-compose down && docker-compose up -d --force-recreate

cd mobile
lsof -ti tcp:8081 | xargs kill -9 2>/dev/null

# Update .env.local to current USB IP
CURRENT_IP=$(ifconfig | grep "inet 169.254" | awk '{print $2}' | head -1)
echo "EXPO_PUBLIC_API_URL=http://${CURRENT_IP}:8000" > .env.local

npm run dev
```

On phone: long-press BrickScan → Remove App → rebuild from Xcode for a fully clean install.

---

## Typical dev workflow (once everything's working)

```bash
# Terminal 1: backend (keep running, auto-reloads on code changes)
cd /Users/edrickchang/Documents/Claude/Projects/Lego/brickscan
docker-compose up -d

# Terminal 2: Metro (keep running)
cd /Users/edrickchang/Documents/Claude/Projects/Lego/brickscan/mobile
npm run dev

# Xcode: select "Edrick's iPhone 15 Pro Max" → ⌘R
```

For JS code changes: edit → save → press `r` in Metro terminal → hot reload.
For backend code changes: edit → save → Docker auto-reloads.
For native code changes: edit → save → `⌘R` in Xcode (~3–10 min rebuild).

---

## Known non-blocking log noise (safe to ignore)

- `[DepthCapture] Capture failed: Failed to capture synchronized RGB/depth frames`
  — expected. `expo-camera` owns the camera session; LiDAR can't get a second
  concurrent session. We've disabled the call in `ScanScreen.tsx` but the warning
  may still appear from a stale hook subscription. It does NOT break scanning.
- `[Worklets] Mismatch between C++ code version and JavaScript code version`
  — cosmetic; doesn't affect functionality.
- 2000+ Xcode build warnings (nullability, deprecation) — all from pods, none actionable.
- `WARN [Config] NativeModules.SourceCode.scriptURL is empty` — expected under
  SDK 55; `.env.local` is used instead.

---

## Stack summary (post-upgrade)

| Component | Version |
|---|---|
| Xcode | 26.4 |
| iOS SDK | 26.4 |
| Expo SDK | 55.0.15 |
| React Native | 0.83.4 |
| React | 19.2.0 |
| TypeScript | 5.9 |
| Reanimated | 4.2.1 |
| Gesture Handler | 2.30.0 |
| Safe Area | 5.6 |
| Async Storage | 2.2 |
| Navigation | @react-navigation v6 |
| babel-preset-expo | 55.x |

---

## Milestone checkpoint: scan pipeline is live (2026-04-14)

End-to-end working:

```
User taps shutter
  → expo-camera captures JPEG
  → ImageManipulator resizes to 512×512, compresses q=0.7
  → FileSystem base64-encodes
  → axios POST to EXPO_PUBLIC_API_URL/api/local-inventory/scan
  → backend runs cascade: Brickognize → Gemini 2.5 Flash → local ONNX
  → Rebrickable catalog lookup for part details
  → returns top-3 predictions with confidence
  → ScanResultScreen displays
```

Average scan latency: 5–8 seconds.

---

## Milestone checkpoint: ML data pipeline + feedback flywheel (2026-04-15)

Pre-positioned for the next training run:

- **Synthetic render corpus** — `ml/data/synthetic_dataset` (symlink to
  `~/Desktop/synthetic_dataset`, ~500 LEGO parts × 540 renders, 5.8 GB).
  Matches `train_contrastive.py`'s class-folder layout out of the box.
- **Nature 2023 real-photo corpus** — downloadable via
  `./backend/venv/bin/python3 ml/data/download_nature_2023.py`.
  Provides the ~52k real photos (closes sim-to-real gap) and the
  PASCAL-VOC → YOLO bounding-box set for detector training.
- **Renderer upgrades** — `ml/blender/blender_render.py` now accepts
  `--elevation-mode hemisphere` (uniform camera sampling on upper hemisphere
  instead of the old fixed 30°) and `--aggressive-aug` (Gaussian noise +
  JPEG artifact pass + extra random-position light per frame). The
  domain-randomisation variant `blender_render_dr.py` is also present for
  higher-quality runs with HDRI backgrounds and material jitter.
- **LDraw colour palette** — `ml/blender/ldraw_colors.py` fallback expanded
  from 15 → all 38 official LEGO solid colours.
- **MOG2 multi-piece detector** — `backend/app/ml/multipiece_detector.py`
  now tries MOG2 background subtraction first (faster + more accurate on
  stable scan backgrounds), falls through to the legacy HSV detector.
- **Feedback flywheel** — three-way feedback UI
  (top-correct / pick-alternative / none-of-these / wrong-colour), CSV
  export at `/api/local-inventory/feedback/export.csv`, weekly accuracy
  snapshots, in-app FeedbackStatsScreen.
- **Dry-run validation** — `bash ml/scripts/smoke_retrain_dry_run.sh`
  exercises the retrain pipeline locally without training anything. Backed
  by `backend/tests/test_retrain_dry_run.py`.
- **Full data layout docs** — `ml/data/README.md` is the source of truth
  for what's on disk + regenerate commands.

**When DGX Spark is back up**, the single command remains:
```bash
bash ~/Desktop/spark_wait_and_relaunch.sh
```
All staged work lands as trained weights in `backend/models/` and lights up
the local cascade automatically.
