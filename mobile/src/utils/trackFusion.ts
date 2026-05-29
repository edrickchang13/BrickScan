/**
 * Per-track multi-view embedding fusion for the continuous-scan pipeline.
 *
 * Phase 0 proved the core thesis: fusing several views of the SAME piece (as a
 * live camera sweep produces) lifts retrieval top-1 from ~88% to ~95% over any
 * single frame. This module is the on-device realisation of that recipe.
 *
 * The fusion recipe (validated in ml/scripts/fusion_eval.py, "confweighted"):
 *   1. Each frame contributes an L2-normalised embedding plus that frame's own
 *      single-frame max similarity to the gallery (maxSim).
 *   2. Across a track's frames, weight each frame by softmax(maxSim * 20).
 *   3. Fused vector = weighted sum of the per-frame embeddings, re-L2-normalised.
 *   4. Re-run k-NN on the fused vector -> fused top-k.
 *
 * Why keep raw frames instead of only a running centroid:
 *   The softmax weights are normalised across the WHOLE set of retained frames
 *   and shift every time a new frame arrives — a confident later view should be
 *   able to dominate earlier noisy ones. A single running centroid can't
 *   reproduce that without re-deriving the weights, so we retain the recent
 *   per-frame (embedding, maxSim) records (capped) and recompute the fused
 *   vector on demand. This keeps parity with the offline eval exactly.
 *
 * Commit logic (isCommitted): the fused top-1 partNum must be stable across the
 * last N updates AND the fused top-1 margin (score gap to #2) must exceed tau.
 * That mirrors the IoU tracker's lock semantics (consecutive agreement + a
 * confidence bar) but operates on the FUSED result rather than per-frame votes.
 *
 * Deliberately pure: no React, no native, no `@/types` import. Everything here
 * is plain arrays/maps so it unit-tests cleanly and runs inside a worklet.
 */

/** A single k-NN hit on the fused vector. `score` is cosine similarity in [-1,1]. */
export interface FusedMatch {
  partNum: string;
  score: number;
}

/**
 * k-NN search over the gallery. Given a fused, L2-normalised query vector,
 * returns matches sorted by descending score (best first). The caller wires
 * this to the real on-device index; tests pass a synthetic gallery.
 */
export type SearchFn = (fused: Float32Array, k: number) => FusedMatch[];

export interface FusionOptions {
  /** Softmax temperature multiplier on per-frame maxSim. Matches fusion_eval.py (20). */
  softmaxScale: number;
  /** Max per-frame records retained per track (ring buffer). Bounds memory in long sweeps. */
  maxFrames: number;
  /** k for fused k-NN. Top-1 is used for commit; #2 is needed for the margin. */
  topK: number;
  /** Consecutive fused-top-1 agreements required before a track may commit (N). */
  commitStability: number;
  /** Minimum fused top-1 margin (score#1 - score#2) required to commit (tau). */
  commitMargin: number;
}

export const DEFAULT_FUSION_OPTS: FusionOptions = {
  softmaxScale: 20.0,
  maxFrames: 32,
  topK: 5,
  commitStability: 4,
  commitMargin: 0.05,
};

interface FrameRecord {
  /** L2-normalised per-frame embedding. */
  embedding: Float32Array;
  /** That frame's single-frame max similarity to the gallery. */
  maxSim: number;
}

interface TrackState {
  frames: FrameRecord[];
  /** Recent fused top-1 partNums, most recent last. Length capped at commitStability. */
  recentTop1: string[];
  /** Cached fused vector; invalidated (null) on each updateTrack. */
  fusedCache: Float32Array | null;
}

// ---------------------------------------------------------------------------
// Vector helpers (pure, allocation-light)
// ---------------------------------------------------------------------------

/** L2-normalise in place; a zero vector is left untouched (norm 0 -> no divide). */
function l2NormalizeInPlace(v: Float32Array): Float32Array {
  let sumSq = 0;
  for (let i = 0; i < v.length; i++) sumSq += v[i] * v[i];
  const norm = Math.sqrt(sumSq);
  if (norm > 0) {
    for (let i = 0; i < v.length; i++) v[i] /= norm;
  }
  return v;
}

/**
 * Numerically-stable softmax over the frames' maxSim*scale. Returns weights
 * that sum to 1, in frame order. Matches torch.softmax(maxsim * scale).
 */
function softmaxWeights(frames: FrameRecord[], scale: number): number[] {
  const logits = frames.map(f => f.maxSim * scale);
  let max = -Infinity;
  for (const l of logits) if (l > max) max = l;
  let sum = 0;
  const exps = logits.map(l => {
    const e = Math.exp(l - max);
    sum += e;
    return e;
  });
  // sum > 0 always (at least one e == 1 from the max element).
  return exps.map(e => e / sum);
}

/**
 * Confidence-weighted pool of the retained per-frame embeddings, re-L2-normalised.
 * This is the fused query vector. Returns null if the track has no frames.
 */
function computeFused(state: TrackState, scale: number): Float32Array | null {
  const { frames } = state;
  if (frames.length === 0) return null;
  if (frames.length === 1) {
    // Single view — fused == that frame (already L2-normalised). Copy so the
    // caller can't mutate our stored embedding through the returned vector.
    return Float32Array.from(frames[0].embedding);
  }
  const weights = softmaxWeights(frames, scale);
  const dim = frames[0].embedding.length;
  const acc = new Float32Array(dim);
  for (let fi = 0; fi < frames.length; fi++) {
    const emb = frames[fi].embedding;
    const w = weights[fi];
    for (let d = 0; d < dim; d++) acc[d] += emb[d] * w;
  }
  return l2NormalizeInPlace(acc);
}

// ---------------------------------------------------------------------------
// TrackFusion — the per-track accumulator
// ---------------------------------------------------------------------------

/**
 * Accumulates per-frame embeddings per track and exposes fused retrieval +
 * commit state. One instance is shared across all live tracks (keyed by the
 * tracker's stable track id). Pure with respect to its own state — no globals,
 * so multiple instances (or a fresh one per scan session) are independent.
 */
export class TrackFusion {
  private readonly opts: FusionOptions;
  private readonly tracks = new Map<string, TrackState>();

  constructor(opts: Partial<FusionOptions> = {}) {
    this.opts = { ...DEFAULT_FUSION_OPTS, ...opts };
  }

  /**
   * Fold one frame's view of a track into its fused estimate.
   *
   * @param trackId   stable id from bboxTracker (BrickTrack.id)
   * @param embedding per-frame embedding; expected L2-normalised. We normalise
   *                  a copy defensively so callers can't corrupt the pool with
   *                  an un-normalised vector and we never retain the caller's
   *                  buffer (which may be a reused scratch Float32Array).
   * @param maxSim    that frame's single-frame max similarity to the gallery
   */
  updateTrack(trackId: string, embedding: Float32Array, maxSim: number): void {
    let state = this.tracks.get(trackId);
    if (!state) {
      state = { frames: [], recentTop1: [], fusedCache: null };
      this.tracks.set(trackId, state);
    }
    const stored = l2NormalizeInPlace(Float32Array.from(embedding));
    state.frames.push({ embedding: stored, maxSim });
    // Bound memory: keep the most recent maxFrames views (ring-buffer trim).
    if (state.frames.length > this.opts.maxFrames) {
      state.frames.splice(0, state.frames.length - this.opts.maxFrames);
    }
    state.fusedCache = null; // invalidate — next fusedTopK recomputes
  }

  /** Current fused (confidence-weighted, L2-normalised) vector, or null if unseen. */
  getFusedEmbedding(trackId: string): Float32Array | null {
    const state = this.tracks.get(trackId);
    if (!state) return null;
    if (state.fusedCache === null) {
      state.fusedCache = computeFused(state, this.opts.softmaxScale);
    }
    return state.fusedCache;
  }

  /**
   * Run k-NN on the track's fused vector and return the top-k matches
   * (best first). Side effect: records the fused top-1 partNum into the
   * track's stability history, which `isCommitted` reads. Returns [] for an
   * unknown / empty track.
   */
  fusedTopK(trackId: string, searchFn: SearchFn): FusedMatch[] {
    const state = this.tracks.get(trackId);
    if (!state) return [];
    const fused = this.getFusedEmbedding(trackId);
    if (!fused) return [];
    const matches = searchFn(fused, this.opts.topK);
    const top1 = matches.length > 0 ? matches[0].partNum : '';
    state.recentTop1.push(top1);
    if (state.recentTop1.length > this.opts.commitStability) {
      state.recentTop1.splice(0, state.recentTop1.length - this.opts.commitStability);
    }
    return matches;
  }

  /**
   * True once the track is confidently identified: the fused top-1 partNum has
   * been identical across the last N (`commitStability`) `fusedTopK` calls AND
   * the most recent fused top-1 margin (score#1 - score#2) exceeds `commitMargin`.
   *
   * Requires `fusedTopK` to have been called at least `commitStability` times —
   * the margin is recomputed here from a fresh search so it reflects the latest
   * fused vector. A track with only one match (#2 absent) treats the margin as
   * the top-1 score itself (gap against an implicit zero), which is the correct
   * behaviour for a gallery hit with no runner-up.
   */
  isCommitted(trackId: string, searchFn: SearchFn): boolean {
    const state = this.tracks.get(trackId);
    if (!state) return false;
    const { commitStability, commitMargin } = this.opts;

    // Stability: need a full window of identical, non-empty top-1s.
    if (state.recentTop1.length < commitStability) return false;
    const newest = state.recentTop1[state.recentTop1.length - 1];
    if (!newest) return false;
    for (const p of state.recentTop1) {
      if (p !== newest) return false;
    }

    // Margin: gap between fused #1 and #2 on the current fused vector.
    const fused = this.getFusedEmbedding(trackId);
    if (!fused) return false;
    const matches = searchFn(fused, this.opts.topK);
    if (matches.length === 0 || matches[0].partNum !== newest) return false;
    const margin = matches[0].score - (matches.length > 1 ? matches[1].score : 0);
    return margin > commitMargin;
  }

  /** Number of frames currently pooled for a track (for diagnostics/tests). */
  frameCount(trackId: string): number {
    return this.tracks.get(trackId)?.frames.length ?? 0;
  }

  /** Drop a track's accumulated state (e.g. when the tracker garbage-collects it). */
  forget(trackId: string): void {
    this.tracks.delete(trackId);
  }

  /** Drop all tracks (e.g. on leaving the scan screen). */
  reset(): void {
    this.tracks.clear();
  }
}
