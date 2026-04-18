/**
 * On-device YOLO detector types.
 *
 * Output from the on-device YOLOv8 detector — one entry per surviving NMS
 * detection. Kept separate from `DetectedPiece` because the on-device model
 * produces a 28-class `{size}_{color}` label that we combine with the backend
 * classifier downstream, rather than replacing it.
 */

export type NormalizedBbox = [number, number, number, number]; // x1,y1,x2,y2 in [0,1]

export interface RawDetection {
  bbox: NormalizedBbox;
  classIdx: number;
  /** YOLO 28-class label, e.g. "2x4_blue". May be empty string if labels unavailable. */
  className: string;
  /** Post-sigmoid class confidence in [0,1]. */
  score: number;
}

export interface DetectorMetrics {
  /** Image preprocess time (ms). */
  preprocessMs: number;
  /** Pure ORT session.run time (ms). */
  inferenceMs: number;
  /** Decode + NMS time (ms). */
  postprocessMs: number;
  /** Total wall-clock detect() time (ms). */
  totalMs: number;
  /** Number of raw candidates before NMS. */
  preNmsCount: number;
  /** Number of detections returned. */
  postNmsCount: number;
}

export interface DetectResult {
  detections: RawDetection[];
  metrics: DetectorMetrics;
}

export interface DetectorOptions {
  /** Minimum class confidence to keep a detection (post-sigmoid). */
  scoreThreshold: number;
  /** IoU threshold for NMS suppression. */
  iouThreshold: number;
  /** Max detections returned after NMS. */
  maxDetections: number;
}

export const DEFAULT_DETECTOR_OPTS: DetectorOptions = {
  scoreThreshold: 0.25,
  iouThreshold: 0.45,
  maxDetections: 100,
};
