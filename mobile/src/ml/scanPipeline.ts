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
import * as ImageManipulator from 'expo-image-manipulator';
// eslint-disable-next-line @typescript-eslint/no-require-imports
const jpeg: { decode: (data: Uint8Array, opts?: { useTArray?: boolean }) => { data: Uint8Array; width: number; height: number } } = require('jpeg-js');

import { YoloDetector, rawDetectionToStubPiece } from './yoloDetector';
import { apiClient } from '@/services/api';
import type { DetectedPiece, ScanPrediction } from '@/types';

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
  /** Wall-clock spent on backend bbox refinement when `refineWithBackend=true`. 0 otherwise. */
  refineMs: number;
  /** How many of the on-device bboxes were upgraded with backend predictions. */
  refinedCount: number;
  totalMs: number;
}

export interface OnDeviceScanOptions {
  /**
   * When true, each detected bbox is cropped from the original JPEG and POSTed
   * to the backend `/api/scan` cascade (Brickognize → Gemini → local). The
   * higher-accuracy backend prediction replaces the on-device 28-class YOLO
   * label as `primaryPrediction`. Adds ~300-500ms to the frame on USB; opt-in
   * because it negates the offline benefit of on-device detection.
   *
   * Phase 5 doc Stage 3 outstanding item #2.
   */
  refineWithBackend?: boolean;
  /**
   * Cap on parallel backend requests when refining. Defaults to 4 — enough to
   * pipeline most scenes without flooding the device's network stack.
   */
  refineConcurrency?: number;
  /**
   * Drop bboxes whose on-device confidence is below this before refining.
   * Avoids spending backend budget on near-noise detections.
   */
  refineMinConfidence?: number;
}

/**
 * Run one on-device detection pass on a JPEG stored at `jpegUri`.
 * Returns a DetectedPiece[] shape that the Phase 2 tracker consumes verbatim.
 */
export async function runOnDeviceScan(
  jpegUri: string,
  opts: OnDeviceScanOptions = {},
): Promise<OnDeviceScanResult> {
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
  let pieces: DetectedPiece[] = detections.map((d, i) => {
    const stub = rawDetectionToStubPiece(d, i);
    return {
      pieceIndex: stub.pieceIndex,
      bbox: stub.bbox,
      predictions: stub.predictions,
      primaryPrediction: stub.primaryPrediction,
    };
  });

  // 4. Optional Phase 5 Stage 3 fix — refine each bbox with the backend
  //    classifier. Each crop is sent through /api/scan; the backend's top
  //    prediction (Brickognize → Gemini → local cascade) replaces the
  //    on-device 28-class YOLO label.
  let refineMs = 0;
  let refinedCount = 0;
  if (opts.refineWithBackend && pieces.length > 0) {
    const tRef = Date.now();
    const minConf = opts.refineMinConfidence ?? 0.30;
    const concurrency = opts.refineConcurrency ?? 4;
    pieces = await refinePiecesWithBackend(
      pieces, jpegUri, decoded.width, decoded.height,
      { minConfidence: minConf, concurrency },
    );
    refineMs = Date.now() - tRef;
    refinedCount = pieces.filter(p => p.primaryPrediction.source === 'backend_refined').length;
  }

  return {
    pieces,
    preprocessMs: metrics.preprocessMs,
    inferenceMs: metrics.inferenceMs,
    postprocessMs: metrics.postprocessMs,
    decodeMs,
    refineMs,
    refinedCount,
    totalMs: Date.now() - tTotal,
  };
}

interface RefineOpts { minConfidence: number; concurrency: number; }

/**
 * Crop each piece's bbox out of the source JPEG and POST it to /api/scan.
 * Replaces `primaryPrediction` with the backend's top result, marking the
 * source as `backend_refined` so the UI can distinguish refined vs raw bboxes.
 *
 * Refinement is best-effort — any single bbox that fails to crop, encode, or
 * post falls back to the on-device stub silently. The whole pipeline never
 * fails because of one bad crop.
 */
async function refinePiecesWithBackend(
  pieces: DetectedPiece[],
  jpegUri: string,
  imgWidth: number,
  imgHeight: number,
  opts: RefineOpts,
): Promise<DetectedPiece[]> {
  const refineTask = async (piece: DetectedPiece): Promise<DetectedPiece> => {
    const stubConf = piece.primaryPrediction?.confidence ?? 0;
    if (stubConf < opts.minConfidence || !piece.bbox || piece.bbox.length !== 4) {
      return piece;
    }
    try {
      const [nx1, ny1, nx2, ny2] = piece.bbox;
      const x = Math.max(0, Math.round(nx1 * imgWidth));
      const y = Math.max(0, Math.round(ny1 * imgHeight));
      const w = Math.max(1, Math.round((nx2 - nx1) * imgWidth));
      const h = Math.max(1, Math.round((ny2 - ny1) * imgHeight));
      // Skip degenerate crops (slivers don't classify well anyway)
      if (w < 24 || h < 24) return piece;

      const cropped = await ImageManipulator.manipulateAsync(
        jpegUri,
        [{ crop: { originX: x, originY: y, width: w, height: h } }],
        { compress: 0.7, format: ImageManipulator.SaveFormat.JPEG, base64: true },
      );
      const b64 = cropped.base64;
      if (!b64) return piece;

      // Backend cascade: Brickognize → Gemini → local. Returns ScanResult.
      const result = await apiClient.scanImage(b64);
      const top = result?.predictions?.[0];
      if (!top) return piece;

      const refined: ScanPrediction = {
        partNum: top.partNum,
        partName: top.partName,
        colorId: String(top.colorId ?? piece.primaryPrediction?.colorId ?? ''),
        colorName: top.colorName ?? piece.primaryPrediction?.colorName ?? '',
        colorHex: top.colorHex ?? piece.primaryPrediction?.colorHex ?? '',
        confidence: top.confidence ?? stubConf,
        imageUrl: top.imageUrl,
        source: 'backend_refined',
      };
      return {
        ...piece,
        predictions: [refined, ...piece.predictions.slice(0, 4)],
        primaryPrediction: refined,
      };
    } catch {
      // Silent fall-back — the on-device stub is still in `piece` unchanged
      return piece;
    }
  };

  // Bounded-parallelism worker pool — keeps the device's network stack happy
  const out: DetectedPiece[] = new Array(pieces.length);
  let cursor = 0;
  const worker = async () => {
    while (true) {
      const i = cursor++;
      if (i >= pieces.length) return;
      out[i] = await refineTask(pieces[i]);
    }
  };
  await Promise.all(Array.from({ length: opts.concurrency }, () => worker()));
  return out;
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
