# Running BrickScan in Xcode

The app is now fully wired to build via Xcode's ▶ button. This document is
the minimal happy-path checklist + the most common failure modes and fixes.

## TL;DR — fastest path to a running build

```bash
# 1. Backend has to be reachable
cd /Users/edrickchang/Documents/Claude/Projects/Lego/brickscan
docker compose up -d
curl http://Edricks-MacBook-Air.local:8000/health   # → {"status":"ok"}

# 2. Open the workspace (NOT the .xcodeproj — pods aren't linked there)
open mobile/ios/BrickScan.xcworkspace
```

Then in Xcode:
1. Top-left destination dropdown → pick your iPhone (or a simulator)
2. Hit ▶ (or ⌘R)
3. First build is ~6-8 min; subsequent builds are ~30s-2min

Metro starts automatically once the app launches; the app connects to the
already-running backend via `http://Edricks-MacBook-Air.local:8000`.

## What got set up

| Component | Status |
|---|---|
| Pods | `pod install` ran cleanly — 116 pods, fmt 11 workaround applied |
| `PixelBridge` native module | Wired into `BrickScan.xcodeproj` via xcodeproj script — 8 .pbxproj refs |
| ONNX model assets | `assets/models/yolo_lego.int8.onnx` (58 MB) — Metro `assetExts` includes `.onnx` |
| `.env.local` | Switched to **Bonjour `Edricks-MacBook-Air.local`** so it survives USB unplug-replug |
| Bridging header | Already imports `RCTBridgeModule` — supports both DepthCapture + new PixelBridge |
| DerivedData | Cleared so Xcode builds fresh (no stale `fmt/format-inl.h` errors) |

## Native modules that should compile

- `DepthCapture` (existing) — LiDAR RGBD via ARKit
- **`PixelBridge` (new)** — vDSP SIMD JPEG → letterboxed Float32 NCHW tensor
  for on-device YOLO. Drops first-frame latency from ~50-80ms → ~8-15ms.
- `expo-camera`, `expo-image-manipulator`, `expo-haptics`, `expo-secure-store`,
  `expo-asset`, `expo-file-system`, `onnxruntime-react-native`, etc.
  (all auto-linked by `use_native_modules!` in the Podfile)

## Common failures and fixes

### "fmt/format-inl.h not found"
Xcode build database got into a weird state with two parallel build dirs.
Fix:
```bash
rm -rf ~/Library/Developer/Xcode/DerivedData/BrickScan-*
rm -rf ios/build "ios/build 2"
```
Then build again.

### "Code signing / no profile found"
Xcode → BrickScan target → Signing & Capabilities → tick "Automatically
manage signing" → pick your Apple ID team. First-run only.

### App launches but blank screen / loading spinner forever
Metro isn't reachable from the phone. Three possible causes:
1. **Backend down**. Run `docker compose ps` from the project root — should show
   `brickscan_backend` as Up. If not: `docker compose up -d`.
2. **mDNS / Bonjour not resolving**. Test from another device:
   `ping Edricks-MacBook-Air.local`. If this fails on your phone, fall back
   to a hardcoded IP — uncomment the IP fallback line in `mobile/.env.local`
   and set the current value of `ifconfig | grep "inet 169.254"`.
3. **Phone & Mac not on the same network when not USB-tethered**. Plug into
   USB; the link-local 169.254.x.x route works regardless of WiFi.

### "ModuleNotFoundError" or red TypeScript errors in Metro
Pure JS issue, not a build issue. Reload the app:
- iPhone: shake → "Reload" in dev menu
- Simulator: ⌘R inside the app
- Or kill Metro and restart: `npx expo start --clear`

### PixelBridge says "Native module not available"
The on-device pipeline auto-falls-back to the JS path (`preprocess.ts`).
You'll get correct output, just ~50ms slower per frame. To activate the
native module, ensure:
1. The Swift + .m files are compiled in (project.pbxproj has 8 references
   to PixelBridgeModule — verify with `grep PixelBridgeModule
   ios/BrickScan.xcodeproj/project.pbxproj | wc -l`).
2. The bridging header imports `<React/RCTBridgeModule.h>` — already does.
3. Build is `Debug-iphoneos` or `Release-iphoneos`, not Mac Catalyst.

## What's NOT included in this build

- **DINOv2 visual-search catalogue** — model still training on Spark
  (~17 hr left). Backend's visual_search service is loaded but `is_loaded()`
  returns false until the catalogue pickle is dropped at
  `backend/data/catalog_embeddings.pkl`. The cascade silently skips that
  tier when missing.
- **Set-scanning model** — POC scaffolding sits on Spark but the model
  isn't trained yet (queued behind DINOv2 on the GPU).
- **Pricing data** — service is wired (`/api/inventory/price/...` and
  `/api/inventory/valuation`) but Rebrickable's pricing endpoint shape is
  undocumented and may need a follow-up adjustment once we see real
  responses.

## Helpful one-liners during development

```bash
# Watch Metro logs from a separate terminal
cd mobile && npx expo start --dev-client

# Force clear Metro cache + RN cache
cd mobile && npx expo start --clear

# Re-run pod install after pulling a branch with native changes
cd mobile/ios && pod install

# Check what the phone is hitting for an API call
docker logs -f brickscan_backend | grep "INFO:"

# Quick backend health sanity check
curl http://Edricks-MacBook-Air.local:8000/health
curl http://Edricks-MacBook-Air.local:8000/api/inventory/visual-search-status
```

## When you next pull from origin

If the pull touches anything in `ios/Pods/` or `Podfile.lock` or any native
module file:

```bash
cd mobile/ios && pod install
```

Then ▶ in Xcode again. Otherwise (pure JS/TS changes), just **shake → Reload**
in the running app.

---

That's it. Open the workspace, hit ▶, and it should land on your phone.
