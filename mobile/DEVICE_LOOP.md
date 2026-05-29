# Phase 4 — On-device closed-loop eval (the device loop)

This is the **OUTER loop** of the Phase 4 closed-loop eval: build BrickScan onto
your physical iPhone, run a live sweep over real bricks, and collect telemetry
that diffs against the **INNER loop** (the autonomous replay harness,
`ml/scripts/livescan_harness.py`). Both loops emit the same schema
(`brickscan.livescan.telemetry/v1`), so a device sweep is directly comparable to
the offline baseline.

```
INNER (no phone, autonomous)                OUTER (this doc — your phone)
  recorded frames                             live camera sweep
    → student.onnx embed                        → student.onnx embed (on device)
    → int8 k-NN → conf-weighted fusion          → int8 k-NN → conf-weighted fusion
    → commit gate                               → commit gate → auto-inventory
    → telemetry JSON  ───────── diff ─────────   → telemetry JSON (file + backend)
```

Everything the **app code** needs is already committed. The two things only YOU
can do — because they need your Apple account and your physical device — are
(1) the one-time signing setup and (2) tethering + running the device.

---

## Scripts (in `mobile/scripts/`)

| Script | What it does | When |
|---|---|---|
| `device-setup.sh` | `npm install` → `expo prebuild` → `pod install` (generates `ios/BrickScan.xcworkspace`) | once per checkout / when native deps change |
| `device-build-install.sh` | `xcodebuild` for the tethered device → `devicectl install` + `launch` | every build/flash |
| `device-eas.sh` | cloud-signed build via EAS (no local signing) | alternative to the two above |

All three are idempotent and resolve repo paths themselves, so you can run them
from anywhere. `--help` on any of them prints its usage.

---

## One-time signing setup (you do this once)

Signing is **not** scripted — it needs your Apple Developer account and is a
2-minute click-through in Xcode. Two options:

### Option A — local Xcode signing (recommended; matches `device-build-install.sh`)

1. Run the native setup once:
   ```bash
   mobile/scripts/device-setup.sh
   ```
   This generates `mobile/ios/BrickScan.xcworkspace`.

2. Open it in Xcode:
   ```bash
   open mobile/ios/BrickScan.xcworkspace
   ```

3. Select the **BrickScan** target → **Signing & Capabilities** tab:
   - Tick **Automatically manage signing**.
   - Set **Team** to your Apple ID team (a free personal team works for
     development; add the Apple ID in Xcode → Settings → Accounts if it isn't
     listed).
   - Xcode auto-creates a development provisioning profile for the bundle id
     `com.edrickchang.brickscan`. If it complains the bundle id is taken, change
     it (here and in `app.json` → `expo.ios.bundleIdentifier`) to something
     unique like `com.<you>.brickscan`.

4. On the iPhone itself, the first install of a personal-team build needs you to
   **trust the developer**: Settings → General → VPN & Device Management → tap
   your developer cert → Trust.

That's it — `device-build-install.sh` passes `-allowProvisioningUpdates`, so
once the Team is set Xcode/`xcodebuild` keep the profile fresh automatically.

### Option B — EAS cloud signing (for `device-eas.sh`)

1. `npm i -g eas-cli && eas login` (the project's `eas.json` + `app.json`
   `projectId` are already set).
2. Register your device for internal/ad-hoc builds:
   ```bash
   eas device:create        # follow the link/QR on the phone to register its UDID
   ```
3. EAS manages the certs/profiles in the cloud the first time you build; just
   answer its prompts. Thereafter `device-eas.sh` is non-interactive.

---

## Per-run device loop

### 0. Prereqs each time
- iPhone plugged in over USB, **unlocked**, **"Trust This Computer"** accepted.
- **Developer Mode** on: iPhone Settings → Privacy & Security → Developer Mode → on (reboots once).
- Confirm the mac sees it:
  ```bash
  xcrun devicectl list devices        # your iPhone should be listed + "paired"
  ```

### 1. Start the backend + Metro (for a Debug/dev build)
The app auto-discovers the API host from Metro's `scriptURL` (see
`src/constants/config.ts`), so just start both on the mac:
```bash
# backend (from the repo that hosts it)
docker compose up -d            # or: backend/start_local.sh
# Metro, with the USB-IP watcher (survives unplug/replug)
cd mobile && npm run dev
```
(For a standalone **Release** build that doesn't need Metro, skip this and pass
`--release` in step 2 — but then set `EXPO_PUBLIC_API_URL` in `.env.local` to a
reachable host first, since there's no Metro `scriptURL` to infer it from.)

### 2. Build + install + launch
```bash
mobile/scripts/device-build-install.sh
# or target a specific device:
mobile/scripts/device-build-install.sh --device <UDID>
# or a standalone release build:
mobile/scripts/device-build-install.sh --release
```
First build is ~6–8 min; incremental builds ~30 s–2 min. The script finds the
device, builds, locates the `.app` from xcodebuild's own settings, then
`devicectl install`s and launches it.

### 3. Turn on telemetry and run a sweep
1. In the app: **Settings → enable the `liveScanTelemetry` flag**
   (`SETTINGS_KEYS.liveScanTelemetry`; off by default so it never costs anything
   in normal use).
2. Open **Continuous Scan** and sweep the camera over your bricks as usual.
3. Tap **Done**.

On Done (and on leaving the screen) the session is flushed to **both** sinks
(`src/ml/telemetrySinks.ts`):
- **File:** `<app document dir>/livescan_telemetry/<platform>-<ms>.json`
- **Backend (unless "local only" is set):** `POST /api/local-inventory/telemetry/livescan`,
  which journals it to `backend/data/livescan_telemetry/<session>.json`.

### 4. Pull the telemetry
- **From the backend** (easiest):
  ```bash
  curl http://<mac-host>:8000/api/local-inventory/telemetry/livescan/sessions
  cat backend/data/livescan_telemetry/<session>.json
  ```
- **From the device** (no network): the file is in the app's Documents container.
  Grab it via Xcode → Window → Devices and Simulators → BrickScan → "Download
  Container", or wire up `expo-sharing` on the file path
  (`telemetryFileUri(sessionId)`).

### 5. Diff against the inner-loop baseline
Run the inner loop on the Spark to (re)generate the offline baseline, then
compare the `aggregate` blocks (commit-rate, fused-top1, latency p50/p90) and
spot-check per-track `frames[].top5` drift:
```bash
ssh spark
cd /home/edrick/brickscan/ml
./venv/bin/python scripts/livescan_harness.py \
  --student     output/student_fastvit_sa24_20260529_025042/student.onnx \
  --gallery-dir training_data/real_photos_v3/train \
  --frames-dir  training_data/real_photos_v3/val \
  --color-artifact models/color_v1/color_model.npz \
  --out         output/livescan_telemetry.json \
  --frames-per-track 4 --max-pieces 60
```
The device numbers will differ (real camera, real lighting, live tracking) — the
point is the gap. Latency on-device should be far better than the Spark CPU
proxy once CoreML/ANE is wired; commit-rate and fused-top1 are the accuracy
signals to watch as you tag known pieces (`telemetry().setExpectedPart`).

---

## Tag a known piece for on-device accuracy (optional)

A live sweep has no ground truth, so the harness's `fused_correct` is null on
device. When you scan a **known** reference brick, you can tag its track so the
device telemetry recovers accuracy:
```ts
import { telemetry } from '@/ml/liveScanTelemetry';
telemetry().setExpectedPart(trackId, '3001');   // wherever you know the part
```
Tagged tracks populate `aggregate.fused_top1` / `n_tagged` exactly like the
inner loop.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ios/BrickScan.xcworkspace not found` | run `mobile/scripts/device-setup.sh` (it runs `pod install`) |
| `no tethered device found` | unlock phone, accept "Trust", enable Developer Mode; check `xcrun devicectl list devices` |
| `fmt/format-inl.h not found` / stale build | `mobile/scripts/device-setup.sh --clean` (nukes Pods + DerivedData) then rebuild |
| signing error in `xcodebuild` | open the workspace in Xcode once and set your Team (see One-time signing) |
| app launches but scan fails to reach API | ensure backend is up and Metro is running (`npm run dev`); the app infers the host from Metro |
| no telemetry file written | confirm the `liveScanTelemetry` flag is **on** and that you tapped **Done** (or left the screen) |
| backend POST didn't arrive | "local only" disables the POST (file is still written); also check the backend is reachable from the phone |

See also `mobile/RUN_IN_XCODE.md` for the manual Xcode ▶ path and deeper native
build notes.
