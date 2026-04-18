/**
 * End-to-end on-device scan pipeline for ContinuousScanScreen.
 *
 * Glues the quantized YOLOv8-L detector to the existing backend
 * DetectedPiece[] shape so the tracker/overlay don't need any changes.
 *
 * Strategy — Phase 5 Stage 2 POC:
 *   1. Take JPEG (already done by caller via expo-camera + ImageManipulator)
 *   2. Decode JPEG → RGBA in JS via `jpeg-js`. ~50–80ms per 720px frame;
 *      acceptable for MVP. Planned optimisation: native pixel bridge that
 *      exposes letterboxed NCHW Float32 buffer directly (Stage 5 in the doc).
 *   3. YoloDetector.detect() → RawDetection[]
 *   4. Build DetectedPiece[] stubs (YOLO 28-class used as primaryPrediction)
 *      OR, when `refineWithBackend=true`, round-trip each bbox crop to the
 *      backend classifier (Phase 5 doc Stage 3 behaviour). POC keeps the
 *      on-device label; backend refinement is a follow-up.
 */
import { Asset } from 'expo-asset';
import * as FileSystem from 'expo-file-system/legacy';
// eslint-disable-next-line @typescript-eslint/no-require-imports
const jpeg: { decode: (data: Uint8Array, opts?: { useTArray?: boolean }) => { data: Uint8Array; width: number; height: number } } = require('jpeg-js');

import { YoloDetector, rawDetectionToStubPiece } from './yoloDetector';
import type { DetectedPiece } from '@/types';

// Class labels baked in — must match the model's labels.json. Kept in sync
// manually; a future "self-describing ONNX" (metadata-props) could load these
// at runtime instead.
const YOLO_LEGO_LABELS: readonly string[] = [
  '1x1_black', '1x1_blue', '1x1_brown', '1x1_green', '1x1_pink', '1x1_red', '1x1_yellow',
  '1x2_green',
  '2x1_blue', '2x1_green', '2x1_pink', '2x1_red', '2x1_yellow',
  '2x2_blue', '2x2_green', '2x2_pink', '2x2_red', '2x2_yellow',
  '2x3_blue', '2x3_green', '2x3_pink', '2x3_red', '2x3_yellow',
  '2x4_blue', '2x4_green', '2x4_pink', '2x4_red', '2x4_yellow',
];

let detector: YoloDetector | null = null;
let loadPromise: Promise<boolean> | null = null;

/**
 * Resolve the quantized model asset to a local file URI usable by
 * onnxruntime-react-native. First call bundles the asset and copies it to
 * DocumentDirectory (so the underlying path survives OTA updates).
 */
async function resolveModelUri(): Promise<string> {
  // The model is ~58 MB — we ship it via Expo's asset pipeline. See
  // app.json → assetBundlePatterns.
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const modelModule = require('../../assets/models/yolo_lego.int8.onnx');
  const asset = Asset.fromModule(modelModule);
  if (!asset.downloaded) {
    await asset.downloadAsync();
  }
  if (!asset.localUri) {
    throw new Error('yolo_lego.int8.onnx failed to resolve a local URI');
  }
  return asset.localUri;
}

export async function ensureDetectorLoaded(): Promise<boolean> {
  if (detector?.isReady()) return true;
  if (loadPromise) return loadPromise;
  loadPromise = (async () => {
    const d = new YoloDetector(YOLO_LEGO_LABELS);
    const uri = await resolveModelUri();
    const ok = await d.load(uri);
    if (ok) {
      detector = d;
    }
    return ok;
  })();
  return loadPromise;
}

export async function disposeDetector(): Promise<void> {
  if (detector) {
    await detector.dispose();
    detector = null;
  }
  loadPromise = null;
}

export interface OnDeviceScanResult {
  pieces: DetectedPiece[];
  preprocessMs: number;
  inferenceMs: number;
  postprocessMs: number;
  decodeMs: number;
  totalMs: number;
}

/**
 * Run one on-device detection pass on a JPEG stored at `jpegUri`.
 * Returns a DetectedPiece[] shape that the Phase 2 tracker consumes verbatim.
 */
export async function runOnDeviceScan(jpegUri: string): Promise<OnDeviceScanResult> {
  if (!detector?.isReady()) {
    throw new Error('On-device detector not loaded; call ensureDetectorLoaded() first');
  }
  const tTotal = Date.now();

  // 1. Decode JPEG → RGBA
  const tDec = Date.now();
  const base64 = await FileSystem.readAsStringAsync(jpegUri, {
    encoding: FileSystem.EncodingType.Base64,
  });
  const jpegBytes = base64ToUint8Array(base64);
  const decoded = jpeg.decode(jpegBytes, { useTArray: true });
  const decodeMs = Date.now() - tDec;

  // 2. Run detector (handles its own letterbox + NCHW)
  const { detections, metrics } = await detector.detect(
    decoded.data,
    decoded.width,
    decoded.height,
  );

  // 3. Build DetectedPiece[] (on-device label → primaryPrediction stub)
  const pieces: DetectedPiece[] = detections.map((d, i) => {
    const stub = rawDetectionToStubPiece(d, i);
    return {
      pieceIndex: stub.pieceIndex,
      bbox: stub.bbox,
      predictions: stub.predictions,
      primaryPrediction: stub.primaryPrediction,
    };
  });

  return {
    pieces,
    preprocessMs: metrics.preprocessMs,
    inferenceMs: metrics.inferenceMs,
    postprocessMs: metrics.postprocessMs,
    decodeMs,
    totalMs: Date.now() - tTotal,
  };
}

function base64ToUint8Array(b64: string): Uint8Array {
  // RN doesn't have atob globally in all runtimes; use a minimal polyfill.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const g: any = globalThis;
  if (typeof g.atob === 'function') {
    const bin = g.atob(b64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    return arr;
  }
  // Fallback: manual base64 decode.
  const keyStr = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=';
  const input = b64.replace(/[^A-Za-z0-9+/=]/g, '');
  const out: number[] = [];
  for (let i = 0; i < input.length; ) {
    const e1 = keyStr.indexOf(input.charAt(i++));
    const e2 = keyStr.indexOf(input.charAt(i++));
    const e3 = keyStr.indexOf(input.charAt(i++));
    const e4 = keyStr.indexOf(input.charAt(i++));
    out.push((e1 << 2) | (e2 >> 4));
    if (e3 !== 64) out.push(((e2 & 15) << 4) | (e3 >> 2));
    if (e4 !== 64) out.push(((e3 & 3) << 6) | e4);
  }
  return new Uint8Array(out);
}
