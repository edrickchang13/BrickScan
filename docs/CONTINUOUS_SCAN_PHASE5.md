# Continuous Scan — Phase 5 (started 2026-04-18: on-device YOLO)

## Status snapshot

**Stage 1a — int8 quantization of YOLOv8-L — ✅ shipped**
* `ml/export/quantize_yolo_int8.py` runs `onnxruntime.quantization.quantize_static`
  with a 128-image calibration set sampled from `yolo_dataset/val`.
* Detect head (`/model.22/*`) excluded from quant — otherwise the class-score
  sigmoid collapses to zero.
* Output: `backend/models/yolo_lego.int8.onnx` — **58.3 MB** (35% of the 166 MB fp32).
* Verification (`ml/export/verify_yolo_int8.py`, 32 val images):
  mean IoU@matched **0.964**, class agreement **92.9%**, count delta +13.3%.
  CPU latency: 241 ms int8 vs 449 ms fp32 on Apple Silicon.

**Stage 1b — YOLOv8-n distillation — 📝 script prepared, not launched**
* `ml/training/train_yolo_nano_distillation.py` — Ultralytics monkey-patch
  with KD-KL loss (student softmax vs teacher softmax with temperature).
* Required to hit the ≤10 MB install-delta acceptance criterion. 58 MB int8
  is below the 200 MB App Store cellular cap but above the delta target.
* Needs DGX Spark (Tailscale 100.124.143.19) and the YOLOv8-L `.pt` teacher
  checkpoint. Expected runtime ~6 h on a single B100. Do not launch without
  verifying teacher path.

**Stage 2 — on-device inference path — ✅ shipped**
* Deps: `onnxruntime-react-native@1.24.3`, `jpeg-js`, `expo-asset`.
* `mobile/src/ml/` — pure-TS modules:
  - `preprocess.ts` — letterbox + NCHW Float32 tensor build
  - `postprocess.ts` — YOLOv8 head decode (auto-detects pre-sigmoided exports)
    + greedy NMS + un-letterbox to normalised [0,1] bbox
  - `yoloDetector.ts` — wraps `onnxruntime-react-native` InferenceSession
  - `scanPipeline.ts` — end-to-end JPEG → RGBA → detect → `DetectedPiece[]`
* 16 Jest unit tests over preprocess/postprocess — all green.
* Model bundled at `mobile/assets/models/yolo_lego.int8.onnx`;
  `metro.config.js` adds `onnx` to assetExts; `app.json` lists it under
  `assetBundlePatterns`.

**Stage 3 — wired into continuous-scan — ✅ shipped**
* `SettingsScreen` toggle **"On-Device Detection"** (beta). When on, the
  screen preloads the detector on mount and `processOneFrame` runs local
  YOLO; transparent fallback to `apiClient.scanMultiPiece` on any failure.
* Current behaviour: on-device produces stub `DetectedPiece[]` with the
  YOLO 28-class label (`"2x4_blue"` → Rebrickable `3001` + blue hex) as the
  `primaryPrediction`. **The per-bbox backend classifier round-trip is
  not yet wired** — next iteration.

**Stage 4 — adaptive latency throttle — ✅ shipped**
* No custom thermal native module; instead the scan loop watches a
  5-sample rolling median latency.
* Stepped interval: **1200 ms → 2400 ms → 4800 ms** when median >500 ms;
  recovers when median <350 ms.
* `SettingsScreen` toggle **"High Performance Mode"** bypasses the throttle.

## Outstanding work before declaring done

1. **Native pixel bridge** — the JS `jpeg-js` decode adds ~50–80 ms per
   frame on iPhone 13. Native letterbox + tensor-pack should push
   first-frame latency below the 150 ms acceptance bar.
2. **Backend classifier round-trip per on-device bbox** — for now we use
   the 28-class YOLO label directly; Brickognize/MobileNetV3 refinement
   (Phase 2 path) is still the higher-accuracy option when online.
3. **Distillation launch** — needed to meet the ≤10 MB IPA delta criterion.
4. **On-device airplane-mode E2E test** on physical hardware.
5. **First `pod install`** after pulling — native module autolinks; builds
   from Xcode.

## Original plan (kept for reference) — deferred → in progress

---

# Continuous Scan — Phase 5 (deferred: true on-device inference)

## What was skipped

Phase 3 was originally scoped to include on-device YOLO inference via Core ML
/ onnxruntime-react-native. On closer inspection this is a multi-day native
project, not a 2-3 hour task, so it was deferred.

## Why it's nontrivial

1. **Model size.** Our trained YOLOv8-L ONNX is **167 MB**. App Store cellular
   download cap is 200 MB; shipping the model in-app would push BrickScan
   over that line for the first install.
2. **Native linking.** `onnxruntime-react-native` requires CocoaPods install,
   Xcode native build (~8 min), and breaks Expo's "change JS → hot-reload"
   dev loop.
3. **Pre/post-processing.** YOLO needs letterbox resize + normalization + NMS
   + coordinate decode. These are ~300 lines of RN-compatible JS that has to
   match the Python training pipeline exactly. One small mismatch silently
   tanks accuracy.
4. **Battery / thermals.** On-device YOLO at 640×640 on the iPhone 15 Pro Max
   Neural Engine is ~80–120 ms per frame. Sustained 4 fps runs the thermal
   sensor into "fair" within ~3 min, triggering iOS to throttle the GPU —
   latency spikes to 400 ms and stays there.

## Recommended plan when you commit to it

**Stage 1 — shrink the model (~1 day)**

* Distill YOLOv8-L down to YOLOv8-n (nano, ~6 MB). Train on the same Hex:Lego
  dataset but with YOLOv8-L as the teacher. Expect ~5 pts mAP50 drop —
  acceptable since on-device trades quality for latency + privacy.
* Alternative: int8-quantize YOLOv8-L to ~40 MB with onnxruntime's
  quantization toolkit. Smaller accuracy hit but still 4× the app-size cost
  of a distilled nano.

**Stage 2 — on-device inference path (~1 day)**

* Add `onnxruntime-react-native` dep → `pod install` → `npx expo run:ios`.
* Implement `OnDeviceDetector` module:
  - Load ONNX from bundled asset or first-launch CDN download.
  - Letterbox input → normalize → run session → NMS → return normalised
    bbox list matching backend `DetectedPiece[]` shape.
* Feature-flag it off by default; mobile falls back to the backend endpoint
  if the model fails to load.

**Stage 3 — wire into continuous-scan (~0.5 day)**

* Replace `apiClient.scanMultiPiece` in `processOneFrame` with a function
  that (a) runs detection on-device, (b) still round-trips to backend for
  the classifier (the classifier is small enough to run locally too but
  Brickognize + our Phase 2 MobileNetV3 are higher-accuracy and worth the
  ~300 ms trip).

**Stage 4 — thermal / battery protections (~0.5 day)**

* Subscribe to `expo-battery` + `thermalState` events; when state is "fair"
  or worse, drop fps from 4 → 2 → 1 automatically.
* Add a "high performance mode" setting that bypasses the automatic throttle.

## Acceptance criteria

* Full on-device scan works in airplane mode (no backend).
* First-frame latency under 150 ms on iPhone 13 and newer.
* Sustained 30 min of continuous scanning without thermal throttling kicking in.
* App install delta: ≤ 10 MB IPA over the pre-Phase-5 build.

## Current state (as of this commit)

* Phase 1 (MVP) + Phase 2 (per-bbox IoU tracking + overlay) + Phase 3a/c/d
  (Kalman smoothing, session persistence, dev perf telemetry) + Phase 4a/b/c
  (confirmation modal, drawer sort, polished labels) are all shipping.
* Inference still round-trips to backend via `/api/local-inventory/scan-multi`.
* Everything in Phase 5 is additive — none of the deferred work blocks the
  features already in prod.
