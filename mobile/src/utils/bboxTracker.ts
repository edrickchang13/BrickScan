/**
 * Simple IoU-based multi-object tracker for the continuous-scan pipeline.
 *
 * The backend returns `DetectedPiece[]` per frame — each piece has a normalised
 * bbox [x1, y1, x2, y2] in [0,1] image space plus a stack of `predictions`.
 * This tracker matches each fresh detection to a running track by highest
 * IoU overlap with the track's last-known bbox; falls through to spawning a
 * new track if no match exceeds MATCH_IOU_THRESHOLD.
 *
 * For each track we maintain:
 *   - vote history (per-frame top-1 part_num + confidence)
 *   - exponential-moving-average confidence
 *   - consecutive-agreement counter (how many recent frames agreed on the top)
 *   - lock state (fires at N consecutive agreements + confidence threshold)
 *
 * The tracker is deliberately simple — no Kalman filter, no appearance model.
 * For a mostly-static phone pointing at a pile of bricks that works fine.
 * Upgrade to DeepSORT-style embedding matching when the use case evolves.
 */

import type { DetectedPiece } from '@/types';

export type Bbox = [number, number, number, number]; // x1, y1, x2, y2

export interface BrickTrack {
  /** Stable local ID (first-seen monotonic counter). */
  id: string;
  /** Current best-guess part_num (highest fused confidence). */
  partNum: string;
  partName: string;
  colorName?: string;
  colorHex?: string;
  /** Most recent bounding box (normalised image-space). */
  bbox: Bbox;
  /** Raw predictions seen for this track, most recent first. */
  history: { partNum: string; confidence: number; ts: number }[];
  /** EMA of top-1 confidence across frames. */
  fusedConfidence: number;
  /** Number of consecutive frames where the top-1 part_num matched `partNum`. */
  consecutiveAgreements: number;
  firstSeenAt: number;
  lastSeenAt: number;
  lockedAt: number | null;
  /** Monotonic counter — bumped each time track is updated. Used by UI keys. */
  version: number;
}

export interface TrackerOptions {
  /** IoU threshold above which two boxes are considered the same piece. */
  matchIouThreshold: number;
  /** Consecutive agreements required to promote a track to "locked". */
  lockAgreementCount: number;
  /** Fused confidence required at lock time. */
  lockConfidence: number;
  /** EMA smoothing coefficient (closer to 1 = less smoothing). */
  emaAlpha: number;
  /** Track TTL (ms) with no sighting — then garbage-collected (pre-lock only). */
  trackTimeoutMs: number;
}

export const DEFAULT_TRACKER_OPTS: TrackerOptions = {
  matchIouThreshold: 0.30,
  lockAgreementCount: 3,
  lockConfidence: 0.85,
  emaAlpha: 0.40,
  trackTimeoutMs: 15_000,
};

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

function iou(a: Bbox, b: Bbox): number {
  const [ax1, ay1, ax2, ay2] = a;
  const [bx1, by1, bx2, by2] = b;
  const ix1 = Math.max(ax1, bx1);
  const iy1 = Math.max(ay1, by1);
  const ix2 = Math.min(ax2, bx2);
  const iy2 = Math.min(ay2, by2);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  if (inter === 0) return 0;
  const areaA = Math.max(0, ax2 - ax1) * Math.max(0, ay2 - ay1);
  const areaB = Math.max(0, bx2 - bx1) * Math.max(0, by2 - by1);
  const union = areaA + areaB - inter;
  return union > 0 ? inter / union : 0;
}

// ---------------------------------------------------------------------------
// Update — call per frame with the latest detections
// ---------------------------------------------------------------------------

let _nextTrackId = 1;
function mintId() {
  return `t${_nextTrackId++}_${Date.now().toString(36)}`;
}

export interface UpdateResult {
  tracks: BrickTrack[];
  /** IDs of tracks that crossed the lock threshold during THIS update. */
  newlyLocked: string[];
}

export function updateTracks(
  previous: BrickTrack[],
  detections: DetectedPiece[],
  now: number = Date.now(),
  opts: TrackerOptions = DEFAULT_TRACKER_OPTS,
): UpdateResult {
  // Step 1: greedy IoU matching, best pairs first
  const tracksMutable = previous.map(t => ({ ...t, _matched: false as boolean }));
  const detectionsMutable = detections
    .filter(d => d.bbox && d.bbox.length === 4)
    .map(d => ({ ...d, _assigned: false as boolean }));

  const matchCandidates: { trackIdx: number; detIdx: number; iou: number }[] = [];
  for (let ti = 0; ti < tracksMutable.length; ti++) {
    for (let di = 0; di < detectionsMutable.length; di++) {
      const det = detectionsMutable[di];
      if (!det.bbox || det.bbox.length !== 4) continue;
      const score = iou(
        tracksMutable[ti].bbox,
        det.bbox as Bbox,
      );
      if (score >= opts.matchIouThreshold) {
        matchCandidates.push({ trackIdx: ti, detIdx: di, iou: score });
      }
    }
  }
  matchCandidates.sort((a, b) => b.iou - a.iou);

  for (const c of matchCandidates) {
    const t = tracksMutable[c.trackIdx];
    const d = detectionsMutable[c.detIdx];
    if (t._matched || d._assigned) continue;
    t._matched = true;
    d._assigned = true;
  }

  // Step 2: update matched tracks with fresh predictions
  const newlyLocked: string[] = [];
  const out: BrickTrack[] = [];

  for (let ti = 0; ti < tracksMutable.length; ti++) {
    const t = tracksMutable[ti];
    // Find the assignment, if any
    const match = matchCandidates.find(c => c.trackIdx === ti && tracksMutable[c.trackIdx]._matched);
    const det = match ? detectionsMutable[match.detIdx] : null;

    if (det && det._assigned) {
      const primary = det.primaryPrediction;
      const conf = primary?.confidence ?? 0;
      const topPart = primary?.partNum ?? '';
      const sameTopAsBefore = topPart && topPart === t.partNum;
      const consecutive = sameTopAsBefore ? t.consecutiveAgreements + 1 : 1;
      const fused = opts.emaAlpha * conf + (1 - opts.emaAlpha) * t.fusedConfidence;
      const shouldLock = t.lockedAt === null
        && consecutive >= opts.lockAgreementCount
        && fused >= opts.lockConfidence;
      const updated: BrickTrack = {
        ...t,
        partNum: topPart || t.partNum,
        partName: primary?.partName || t.partName,
        colorName: primary?.colorName || t.colorName,
        colorHex: primary?.colorHex || t.colorHex,
        bbox: det.bbox as Bbox,
        history: [
          { partNum: topPart, confidence: conf, ts: now },
          ...t.history.slice(0, 14),
        ],
        fusedConfidence: fused,
        consecutiveAgreements: consecutive,
        lastSeenAt: now,
        lockedAt: shouldLock ? now : t.lockedAt,
        version: t.version + 1,
      };
      delete (updated as any)._matched;
      if (shouldLock) newlyLocked.push(updated.id);
      out.push(updated);
    } else {
      // Track wasn't re-detected this frame
      const decayed: BrickTrack = {
        ...t,
        fusedConfidence: t.fusedConfidence * 0.92,
        consecutiveAgreements: 0,
        version: t.version + 1,
      };
      delete (decayed as any)._matched;
      // Drop pre-lock tracks that have aged out
      if (decayed.lockedAt === null && now - decayed.lastSeenAt > opts.trackTimeoutMs) {
        continue;
      }
      out.push(decayed);
    }
  }

  // Step 3: spawn tracks for unassigned detections
  for (const det of detectionsMutable) {
    if (det._assigned) continue;
    if (!det.bbox || det.bbox.length !== 4) continue;
    const primary = det.primaryPrediction;
    if (!primary?.partNum) continue;
    out.push({
      id: mintId(),
      partNum: primary.partNum,
      partName: primary.partName ?? primary.partNum,
      colorName: primary.colorName,
      colorHex: primary.colorHex,
      bbox: det.bbox as Bbox,
      history: [{
        partNum: primary.partNum,
        confidence: primary.confidence ?? 0,
        ts: now,
      }],
      fusedConfidence: primary.confidence ?? 0,
      consecutiveAgreements: 1,
      firstSeenAt: now,
      lastSeenAt: now,
      lockedAt: null,
      version: 1,
    });
  }

  return { tracks: out, newlyLocked };
}
