/**
 * Unit tests for the per-bbox IoU tracker that powers ContinuousScanScreen.
 *
 * The tracker is a pure function — no React, no camera, no async — so we
 * can blast it with deterministic inputs and verify exact state transitions.
 *
 * Coverage targets:
 *   - IoU matching assigns the same track ID across frames when bbox stable
 *   - A new (non-overlapping) bbox creates a new track
 *   - A pre-lock track expires after `trackTimeoutMs` of no sightings
 *   - A locked track survives indefinite no-sightings
 *   - Lock fires after N consecutive same-top-1 frames at fused conf ≥ threshold
 *   - Lock does NOT fire when the top class flips between frames
 *   - newlyLocked carries IDs only for the locks that crossed the threshold
 *     during the current update (not for already-locked tracks)
 */
import { updateTracks, DEFAULT_TRACKER_OPTS } from '@/utils/bboxTracker';
import type { DetectedPiece } from '@/types';

function piece(
  bbox: [number, number, number, number],
  partNum: string,
  confidence: number,
  pieceIndex = 0,
): DetectedPiece {
  const primary = {
    partNum, partName: partNum, colorId: '',
    colorName: '', colorHex: '', confidence,
  };
  return {
    pieceIndex,
    bbox,
    predictions: [primary],
    primaryPrediction: primary,
  };
}

describe('updateTracks', () => {
  it('creates a new track for an unmatched detection', () => {
    const t0 = 1_000_000;
    const { tracks, newlyLocked } = updateTracks([], [piece([0.1, 0.1, 0.3, 0.3], '3001', 0.9)], t0);
    expect(tracks).toHaveLength(1);
    expect(tracks[0].partNum).toBe('3001');
    expect(tracks[0].consecutiveAgreements).toBe(1);
    expect(tracks[0].lockedAt).toBeNull();
    expect(newlyLocked).toEqual([]);
  });

  it('extends the existing track when bbox overlaps', () => {
    const t0 = 1_000_000;
    const { tracks: t1 } = updateTracks([], [piece([0.1, 0.1, 0.3, 0.3], '3001', 0.9)], t0);
    // Slight bbox jitter — IoU well above the 0.30 threshold
    const { tracks: t2 } = updateTracks(t1, [piece([0.11, 0.11, 0.31, 0.31], '3001', 0.9)], t0 + 1000);
    expect(t2).toHaveLength(1);
    expect(t2[0].id).toBe(t1[0].id);                      // same track ID
    expect(t2[0].consecutiveAgreements).toBe(2);
  });

  it('spawns a new track for a non-overlapping bbox', () => {
    const t0 = 1_000_000;
    const { tracks: t1 } = updateTracks([], [piece([0.0, 0.0, 0.2, 0.2], '3001', 0.9)], t0);
    const { tracks: t2 } = updateTracks(t1, [piece([0.6, 0.6, 0.9, 0.9], '3022', 0.8)], t0 + 1000);
    expect(t2).toHaveLength(2);
    expect(new Set(t2.map(t => t.partNum))).toEqual(new Set(['3001', '3022']));
  });

  it('locks after lockAgreementCount consecutive same-class frames', () => {
    const t0 = 1_000_000;
    let prev: any[] = [];
    let newlyLocked: string[] = [];
    for (let i = 0; i < DEFAULT_TRACKER_OPTS.lockAgreementCount; i++) {
      const r = updateTracks(prev, [piece([0.1, 0.1, 0.3, 0.3], '3001', 0.95)], t0 + i * 1000);
      prev = r.tracks;
      newlyLocked = r.newlyLocked;
    }
    expect(prev).toHaveLength(1);
    expect(prev[0].lockedAt).not.toBeNull();
    expect(newlyLocked).toEqual([prev[0].id]);
  });

  it('does not lock when the top-1 class flips between frames', () => {
    const t0 = 1_000_000;
    let prev: any[] = [];
    const sequence = ['3001', '3022', '3001', '3022'];   // alternating
    for (let i = 0; i < sequence.length; i++) {
      const r = updateTracks(prev, [piece([0.1, 0.1, 0.3, 0.3], sequence[i], 0.95)], t0 + i * 1000);
      prev = r.tracks;
    }
    expect(prev).toHaveLength(1);
    expect(prev[0].lockedAt).toBeNull();
    expect(prev[0].consecutiveAgreements).toBe(1);   // resets on each flip
  });

  it('newlyLocked is empty when re-passing an already-locked track', () => {
    const t0 = 1_000_000;
    let prev: any[] = [];
    for (let i = 0; i < DEFAULT_TRACKER_OPTS.lockAgreementCount; i++) {
      prev = updateTracks(prev, [piece([0.1, 0.1, 0.3, 0.3], '3001', 0.95)], t0 + i * 1000).tracks;
    }
    expect(prev[0].lockedAt).not.toBeNull();
    const r = updateTracks(prev, [piece([0.1, 0.1, 0.3, 0.3], '3001', 0.95)], t0 + 10_000);
    expect(r.newlyLocked).toEqual([]);
  });

  it('expires a pre-lock track after trackTimeoutMs of no sightings', () => {
    const t0 = 1_000_000;
    const { tracks: t1 } = updateTracks([], [piece([0.1, 0.1, 0.3, 0.3], '3001', 0.5)], t0);
    expect(t1).toHaveLength(1);
    // No detections in the next pass, way past TTL
    const tFar = t0 + DEFAULT_TRACKER_OPTS.trackTimeoutMs + 1000;
    const { tracks: t2 } = updateTracks(t1, [], tFar);
    expect(t2).toHaveLength(0);
  });

  it('keeps a locked track even after long absence', () => {
    const t0 = 1_000_000;
    let prev: any[] = [];
    for (let i = 0; i < DEFAULT_TRACKER_OPTS.lockAgreementCount; i++) {
      prev = updateTracks(prev, [piece([0.1, 0.1, 0.3, 0.3], '3001', 0.95)], t0 + i * 1000).tracks;
    }
    const lockedId = prev[0].id;
    const tFar = t0 + DEFAULT_TRACKER_OPTS.trackTimeoutMs * 5;
    const { tracks: t2 } = updateTracks(prev, [], tFar);
    expect(t2).toHaveLength(1);
    expect(t2[0].id).toBe(lockedId);
    expect(t2[0].lockedAt).not.toBeNull();
  });

  it('handles two simultaneous bricks of the same part_num as separate tracks', () => {
    const t0 = 1_000_000;
    const detections = [
      piece([0.0, 0.0, 0.2, 0.2], '3001', 0.9, 0),
      piece([0.7, 0.7, 0.9, 0.9], '3001', 0.9, 1),    // far away
    ];
    const { tracks } = updateTracks([], detections, t0);
    expect(tracks).toHaveLength(2);
    expect(tracks[0].id).not.toBe(tracks[1].id);
    expect(tracks.every(t => t.partNum === '3001')).toBe(true);
  });
});
