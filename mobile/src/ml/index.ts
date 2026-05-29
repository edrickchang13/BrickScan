export { YoloDetector, rawDetectionToStubPiece } from './yoloDetector';
export {
  computeLetterbox,
  hwcRgbToNchwFloat32,
  letterboxRgba,
  YOLO_INPUT_SIZE,
} from './preprocess';
export type { LetterboxInfo } from './preprocess';
export { decodeYoloOutput } from './postprocess';
export {
  DEFAULT_DETECTOR_OPTS,
} from './types';
export type {
  DetectorOptions,
  DetectorMetrics,
  DetectResult,
  NormalizedBbox,
  RawDetection,
} from './types';

// ── Phase 2 on-device retrieval + colour + live-scan orchestration ──────────
export {
  EmbeddingModel,
  l2NormalizeInPlace,
  EMBED_INPUT_SIZE,
  DEFAULT_EMBEDDING_OPTS,
} from './embeddingRetrieval';
export type { EmbeddingOptions } from './embeddingRetrieval';
export {
  PartIndex,
  buildIndexData,
  int8ToBase64,
  base64ToInt8,
  DEFAULT_INT8_SCALE,
  SYNTHETIC_STUB_INDEX,
} from './partIndex';
export type { PartMatch, PartIndexData } from './partIndex';
export {
  ColorClassifier,
  colorClassifier,
  srgbToLab,
  extractFeature,
  percentileLinear,
  median,
  COLOR_ASSET_VERSION,
} from './colorClassifier';
export type {
  ColorPrediction,
  ColorClassifyResult,
} from './colorClassifier';
export {
  ensureLiveScanLoaded,
  processTrackFrame,
  forgetTrack,
  isTrackCounted,
  markTrackCounted,
  isEmbeddingReady,
  disposeLiveScanEngine,
  configureLiveScanEngine,
  searchFn,
  DEFAULT_LIVE_SCAN_CONFIG,
} from './liveScanEngine';
export type { LiveScanEngineConfig, TrackFrameResult } from './liveScanEngine';
