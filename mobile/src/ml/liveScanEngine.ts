/**
 * On-device live-scan ML engine — the orchestration glue that ties the four
 * Phase 2 modules together for ContinuousScanScreen:
 *
 *   per tracked piece (per frame):
 *     crop bbox from frame JPEG → RGBA
 *       → embeddingRetrieval.embed      (student → L2-norm embedding)
 *       → partIndex.search (top-1 = maxSim)
 *       → trackFusion.updateTrack(id, emb, maxSim)
 *     crop resized to 128×128 RGBA
 *       → colorClassifier.classify       (LDA-kNN colour id)
 *     trackFusion.fusedTopK(id, searchFn) → fused top-k
 *     gate commit on trackFusion.isCommitted(id, searchFn)
 *
 * Mirrors the structure of `scanPipeline.ts` (the YOLO glue): module-level
 * singletons, a graceful-degradation load path, and a JPEG→RGBA crop helper
 * built on ImageManipulator + jpeg-js (the same decode approach the detector
 * pipeline uses). The screen stays declarative and this stays unit-testable.
 *
 * ───────────────────────────────────────────────────────────────────────────
 * REAL MODEL / REAL GALLERY DROP-IN (no code change required)
 * ───────────────────────────────────────────────────────────────────────────
 * Today the engine runs against the SYNTHETIC stub gallery (`PartIndex.synthetic`)
 * and NO embedding model is bundled, so `EmbeddingModel.embed()` returns null and
 * the retrieval/fusion path no-ops gracefully (colour classification still runs).
 * When Phase 1 ships the distilled student `.onnx` and the real gallery index
 * JSON, wire them in via `configureLiveScanEngine({...})` (see TODO markers
 * below) — nothing else in this file or the screen needs to change.
 */
import { Asset } from 'expo-asset';
import * as FileSystem from 'expo-file-system/legacy';
import * as ImageManipulator from 'expo-image-manipulator';
// eslint-disable-next-line @typescript-eslint/no-require-imports
const jpeg: { decode: (data: Uint8Array, opts?: { useTArray?: boolean }) => { data: Uint8Array; width: number; height: number } } = require('jpeg-js');

import { EmbeddingModel } from './embeddingRetrieval';
import { PartIndex, type PartIndexData, type PartMatch } from './partIndex';
import { colorClassifier } from './colorClassifier';
import { TrackFusion, type FusedMatch, type SearchFn } from '@/utils/trackFusion';
import type { Bbox } from '@/utils/bboxTracker';

// ───────────────────────────────────────────────────────────────────────────
// Configuration — the ONLY thing that changes when the real model/index land.
// ───────────────────────────────────────────────────────────────────────────

export interface LiveScanEngineConfig {
  /**
   * Loader for the distilled student embedding model, returning a local file
   * URI for `EmbeddingModel.load()` (e.g. a resolved Expo Asset, like
   * `scanPipeline.resolveModelUri`). `null` ⇒ no model bundled yet ⇒ retrieval
   * is skipped and `embed()` returns null gracefully.
   *
   * TODO(phase1-student): replace `null` with a resolver for the real
   * `assets/models/<student>.onnx`, e.g.:
   *
   *   resolveEmbeddingModelUri: async () => {
   *     const m = require('../../assets/models/brick_student.onnx');
   *     const a = Asset.fromModule(m);
   *     if (!a.downloaded) await a.downloadAsync();
   *     if (!a.localUri) throw new Error('student model failed to resolve');
   *     return a.localUri;
   *   }
   */
  resolveEmbeddingModelUri: (() => Promise<string>) | null;
  /**
   * Source for the gallery retrieval index. Returns a `PartIndexData` to feed
   * `PartIndex.fromData`, or the literal string `'synthetic'` to use the
   * built-in stub (`PartIndex.synthetic()`).
   *
   * TODO(phase1-gallery): replace `'synthetic'` with the real bundled gallery,
   * e.g. (Metro inlines JSON via resolveJsonModule):
   *
   *   loadPartIndexData: async () =>
   *     require('../../assets/models/gallery_index.json') as PartIndexData
   */
  loadPartIndexData: (() => Promise<PartIndexData | 'synthetic'>) | 'synthetic';
}

/**
 * Default config: NO student model + the synthetic stub gallery. This is the
 * "ready to run on device, identifies nothing real yet" state — replace both
 * fields when Phase 1 artifacts land (see the TODO markers above).
 */
export const DEFAULT_LIVE_SCAN_CONFIG: LiveScanEngineConfig = {
  // Phase 1 student: FastViT-SA24, 768-d, 90.1% single-frame / 95.6% fused (N=4).
  resolveEmbeddingModelUri: async () => {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const m = require('../../assets/models/student.onnx');
    const a = Asset.fromModule(m);
    if (!a.downloaded) await a.downloadAsync();
    if (!a.localUri) throw new Error('student model failed to resolve');
    return a.localUri;
  },
  // Phase 1 gallery: 12.5k student-embedded exemplars, int8 global-scale (1/127).
  loadPartIndexData: async () =>
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    require('../../assets/models/gallery_index.json') as PartIndexData,
};

let config: LiveScanEngineConfig = DEFAULT_LIVE_SCAN_CONFIG;

/**
 * Override the model/index sources before `ensureLiveScanLoaded()`. Call once
 * at startup if the real Phase 1 artifacts are available. Resets any loaded
 * state so the next `ensureLiveScanLoaded()` picks up the new sources.
 */
export function configureLiveScanEngine(next: Partial<LiveScanEngineConfig>): void {
  config = { ...config, ...next };
  // Force a reload against the new sources.
  void disposeLiveScanEngine();
}

// ───────────────────────────────────────────────────────────────────────────
// Singletons — one embedding model + one gallery index + one fusion accumulator
// shared across all live tracks (TrackFusion is keyed by the tracker's id).
// ───────────────────────────────────────────────────────────────────────────

const embedder = new EmbeddingModel();
const fusion = new TrackFusion();

let partIndex: PartIndex | null = null;
let loadPromise: Promise<boolean> | null = null;
let embedderReady = false;

/** Tracks already counted into inventory — commit fires the add exactly once. */
const committedTrackIds = new Set<string>();

/**
 * Bound k-NN search over the loaded gallery. Used for both `fusedTopK` and the
 * commit-margin check inside `isCommitted`. Returns [] until the index loads.
 */
export const searchFn: SearchFn = (vec: Float32Array, k: number): FusedMatch[] => {
  if (!partIndex) return [];
  return partIndex.search(vec, k) as PartMatch[];
};

/**
 * Resolve + load the gallery index and (if configured) the student model.
 * Idempotent and concurrency-safe (mirrors `scanPipeline.ensureDetectorLoaded`).
 * Returns true once the gallery is ready; the embedding model is best-effort
 * (its absence just means retrieval no-ops via `embed()` → null).
 */
export async function ensureLiveScanLoaded(): Promise<boolean> {
  if (partIndex) return true;
  if (loadPromise) return loadPromise;
  loadPromise = (async () => {
    // 1. Gallery index (synthetic stub today; real JSON drops in via config).
    try {
      const src = config.loadPartIndexData;
      if (src === 'synthetic') {
        partIndex = PartIndex.synthetic();
      } else {
        const data = await src();
        partIndex = data === 'synthetic' ? PartIndex.synthetic() : PartIndex.fromData(data);
      }
    } catch (err) {
      console.warn('[liveScanEngine] gallery index load failed:', err);
      partIndex = null;
      return false;
    }

    // 2. Embedding model — optional. When unconfigured we leave it unloaded and
    //    `embed()` returns null (graceful: retrieval/fusion are skipped).
    if (config.resolveEmbeddingModelUri) {
      try {
        const uri = await config.resolveEmbeddingModelUri();
        embedderReady = await embedder.load(uri);
        if (!embedderReady) {
          console.warn('[liveScanEngine] embedding model failed to load; retrieval disabled.');
        }
      } catch (err) {
        console.warn('[liveScanEngine] embedding model resolve failed:', err);
        embedderReady = false;
      }
    } else {
      embedderReady = false;
    }

    return true;
  })();
  return loadPromise;
}

/** True once the embedding student is loaded (retrieval/fusion are live). */
export function isEmbeddingReady(): boolean {
  return embedderReady && embedder.isReady();
}

/** Dispose ML state (e.g. on leaving the scan screen). Clears fusion + commits. */
export async function disposeLiveScanEngine(): Promise<void> {
  await embedder.dispose();
  embedderReady = false;
  partIndex = null;
  loadPromise = null;
  fusion.reset();
  committedTrackIds.clear();
}

// ───────────────────────────────────────────────────────────────────────────
// Per-track frame processing
// ───────────────────────────────────────────────────────────────────────────

/** Crop size fed to the colour classifier — it REQUIRES 128×128 RGBA. */
const COLOR_CROP_SIZE = 128;
/**
 * Minimum bbox edge (px, source-image space) worth processing. Slivers don't
 * embed or colour-classify usefully and waste a crop+decode. Mirrors the
 * `w<24||h<24` guard in scanPipeline's backend refinement.
 */
const MIN_CROP_PX = 24;

export interface TrackFrameResult {
  /** Fused top-1 part number, or '' if retrieval is unavailable / no match. */
  fusedPartNum: string;
  /** Fused top-1 cosine score, or 0. */
  fusedScore: number;
  /** Full fused top-k (best first); [] when retrieval is unavailable. */
  fused: FusedMatch[];
  /** Best colour id from the colour classifier, or '' if none segmented. */
  colorId: string;
  colorName: string;
  /** True once the track is confidently committed (stability + margin met). */
  committed: boolean;
}

/**
 * Process one tracked piece for the current frame.
 *
 * Crops the track's bbox out of the frame JPEG (once, at full crop resolution
 * for the embedder; a 128×128 copy for colour), runs embedding→retrieval→fusion
 * and colour classification, then reports the fused top-1 + colour + commit
 * state. Best-effort: a crop/decode/embed failure degrades to whatever did
 * succeed (e.g. colour-only, or an empty result) and never throws into the
 * frame loop.
 *
 * @param trackId    stable BrickTrack.id
 * @param bbox       normalised [x1,y1,x2,y2] in [0,1] image space
 * @param jpegUri    file URI of the current frame JPEG
 * @param imgWidth   frame width (px) — to denormalise the bbox
 * @param imgHeight  frame height (px)
 */
export async function processTrackFrame(
  trackId: string,
  bbox: Bbox,
  jpegUri: string,
  imgWidth: number,
  imgHeight: number,
): Promise<TrackFrameResult> {
  const empty: TrackFrameResult = {
    fusedPartNum: '', fusedScore: 0, fused: [], colorId: '', colorName: '', committed: false,
  };

  // Denormalise + clamp the crop rect to the frame.
  const [nx1, ny1, nx2, ny2] = bbox;
  const x = Math.max(0, Math.round(nx1 * imgWidth));
  const y = Math.max(0, Math.round(ny1 * imgHeight));
  const w = Math.max(1, Math.round((nx2 - nx1) * imgWidth));
  const h = Math.max(1, Math.round((ny2 - ny1) * imgHeight));
  if (w < MIN_CROP_PX || h < MIN_CROP_PX) return empty;
  // Keep the crop inside the image bounds (ImageManipulator clamps, but be safe).
  const cw = Math.min(w, imgWidth - x);
  const ch = Math.min(h, imgHeight - y);
  if (cw < MIN_CROP_PX || ch < MIN_CROP_PX) return empty;

  // 1. Embedding + retrieval + fusion (skipped gracefully if no model loaded).
  if (isEmbeddingReady()) {
    try {
      const crop = await cropToRgba(jpegUri, { x, y, width: cw, height: ch });
      if (crop) {
        const emb = await embedder.embed(crop.rgba, crop.width, crop.height);
        if (emb) {
          // top-1 single-frame similarity = this frame's maxSim for the recipe.
          const top1 = searchFn(emb, 1);
          const maxSim = top1.length > 0 ? top1[0].score : 0;
          fusion.updateTrack(trackId, emb, maxSim);
        }
      }
    } catch (err) {
      console.warn('[liveScanEngine] embed/retrieval failed for', trackId, err);
    }
  }

  // 2. Colour classification — crop MUST be resized to 128×128 RGBA.
  let colorId = '';
  let colorName = '';
  try {
    const colorCrop = await cropToRgba(
      jpegUri,
      { x, y, width: cw, height: ch },
      COLOR_CROP_SIZE,
    );
    if (colorCrop) {
      const res = colorClassifier.classify(
        colorCrop.rgba, colorCrop.width, colorCrop.height,
      );
      colorId = res.colorId;
      colorName = res.name;
    }
  } catch (err) {
    console.warn('[liveScanEngine] colour classify failed for', trackId, err);
  }

  // 3. Fused retrieval + commit gate (no-op when no frames were folded in).
  const fused = fusion.fusedTopK(trackId, searchFn);
  const committed = fusion.isCommitted(trackId, searchFn);
  return {
    fusedPartNum: fused.length > 0 ? fused[0].partNum : '',
    fusedScore: fused.length > 0 ? fused[0].score : 0,
    fused,
    colorId,
    colorName,
    committed,
  };
}

/** Has this track already been counted into inventory? */
export function isTrackCounted(trackId: string): boolean {
  return committedTrackIds.has(trackId);
}

/** Mark a track as counted (call right after a successful inventory add). */
export function markTrackCounted(trackId: string): void {
  committedTrackIds.add(trackId);
}

/** Drop a track's fusion + counted state when the tracker garbage-collects it. */
export function forgetTrack(trackId: string): void {
  fusion.forget(trackId);
  committedTrackIds.delete(trackId);
}

// ───────────────────────────────────────────────────────────────────────────
// JPEG → RGBA crop helper (ImageManipulator crop[/resize] + jpeg-js decode)
// ───────────────────────────────────────────────────────────────────────────

interface CropRect { x: number; y: number; width: number; height: number; }
interface RgbaCrop { rgba: Uint8Array; width: number; height: number; }

/**
 * Crop a rectangle out of a JPEG and decode it to RGBA. When `resizeSquare` is
 * given, the crop is additionally resized to `resizeSquare`×`resizeSquare`
 * (the colour classifier's required 128×128). Returns null on any failure so
 * callers can degrade gracefully.
 */
async function cropToRgba(
  jpegUri: string,
  rect: CropRect,
  resizeSquare?: number,
): Promise<RgbaCrop | null> {
  try {
    const actions: ImageManipulator.Action[] = [
      { crop: { originX: rect.x, originY: rect.y, width: rect.width, height: rect.height } },
    ];
    if (resizeSquare) {
      actions.push({ resize: { width: resizeSquare, height: resizeSquare } });
    }
    const out = await ImageManipulator.manipulateAsync(jpegUri, actions, {
      compress: 1,
      format: ImageManipulator.SaveFormat.JPEG,
    });
    const b64 = await FileSystem.readAsStringAsync(out.uri, {
      encoding: FileSystem.EncodingType.Base64,
    });
    const decoded = jpeg.decode(base64ToUint8Array(b64), { useTArray: true });
    return { rgba: decoded.data, width: decoded.width, height: decoded.height };
  } catch (err) {
    console.warn('[liveScanEngine] cropToRgba failed:', err);
    return null;
  }
}

/** base64 → Uint8Array (atob fast-path + manual fallback). Matches scanPipeline. */
function base64ToUint8Array(b64: string): Uint8Array {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const g: any = globalThis;
  if (typeof g.atob === 'function') {
    const bin = g.atob(b64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    return arr;
  }
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

// Keep `Asset` referenced so the import is retained for the documented TODO
// resolver examples (which use Asset.fromModule). Tree-shaking-safe no-op.
void Asset;
