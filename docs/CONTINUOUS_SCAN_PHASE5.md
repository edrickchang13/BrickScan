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
