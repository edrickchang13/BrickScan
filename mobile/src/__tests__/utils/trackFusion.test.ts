/**
 * Unit tests for the per-track multi-view embedding fusion core.
 *
 * Like the tracker, TrackFusion is a pure data structure — no React, camera, or
 * native deps — so we drive it with synthetic embeddings and a synthetic gallery
 * and assert exact behaviour. The two headline properties we prove:
 *
 *   (a) FUSION BEATS A NOISY SINGLE FRAME. With a gallery of two look-alike
 *       parts, individual frames frequently mis-rank the true part (their
 *       single-frame top-1 is wrong), yet the confidence-weighted fused vector
 *       recovers the correct top-1. This is the Phase 0 88%->95% result in
 *       miniature.
 *
 *   (b) isCommitted FLIPS TRUE ONLY AFTER STABILITY + MARGIN. It stays false
 *       while the fused top-1 is still churning, false when stable-but-narrow
 *       (margin below tau), and true once both the N-window agreement and the
 *       margin bar are met.
 *
 * Everything is seeded/deterministic.
 */
import {
  TrackFusion,
  DEFAULT_FUSION_OPTS,
  type FusedMatch,
  type SearchFn,
} from '@/utils/trackFusion';

// ---------------------------------------------------------------------------
// Synthetic embedding + gallery helpers
// ---------------------------------------------------------------------------

const DIM = 8;

/** Deterministic LCG so tests are reproducible without a seed library. */
function makeRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

function l2(v: number[]): Float32Array {
  const norm = Math.sqrt(v.reduce((a, x) => a + x * x, 0)) || 1;
  return Float32Array.from(v.map(x => x / norm));
}

function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** Two unit "prototype" embeddings for two distinct parts, plus a look-alike pair. */
const PROTO: Record<string, Float32Array> = {
  // Well-separated parts (orthogonal-ish).
  '3001': l2([1, 0, 0, 0, 0, 0, 0, 0]),
  '3002': l2([0, 1, 0, 0, 0, 0, 0, 0]),
  // Look-alike pair: very close in embedding space (small angular gap).
  '3003': l2([0, 0, 1, 0.00, 0, 0, 0, 0]),
  '3004': l2([0, 0, 1, 0.18, 0, 0, 0, 0]),
};

/** Gallery k-NN: cosine against every prototype, top-k by score. */
function gallerySearch(fused: Float32Array, k: number): FusedMatch[] {
  const hits: FusedMatch[] = Object.entries(PROTO).map(([partNum, proto]) => ({
    partNum,
    score: dot(fused, proto),
  }));
  hits.sort((a, b) => b.score - a.score);
  return hits.slice(0, k);
}

/**
 * One noisy frame view of `truthPart`: prototype + Gaussian-ish noise, then
 * L2-normalised. `noise` controls how badly individual frames are corrupted.
 */
function noisyFrame(truthPart: string, rng: () => number, noise: number): Float32Array {
  const proto = PROTO[truthPart];
  const v = new Array(DIM);
  for (let i = 0; i < DIM; i++) {
    // Box-Muller-ish: sum of two uniforms centred at 0.
    const g = (rng() + rng() - 1) * noise;
    v[i] = proto[i] + g;
  }
  return l2(v);
}

/** Single-frame top-1 partNum for a raw (already L2-normalised) frame embedding. */
function singleFrameTop1(frame: Float32Array): string {
  return gallerySearch(frame, 1)[0].partNum;
}

// ---------------------------------------------------------------------------

describe('TrackFusion', () => {
  describe('updateTrack / fused vector', () => {
    it('starts with no frames and yields no matches for an unknown track', () => {
      const tf = new TrackFusion();
      expect(tf.frameCount('nope')).toBe(0);
      expect(tf.getFusedEmbedding('nope')).toBeNull();
      expect(tf.fusedTopK('nope', gallerySearch)).toEqual([]);
      expect(tf.isCommitted('nope', gallerySearch)).toBe(false);
    });

    it('fused vector stays L2-normalised across many frames', () => {
      const tf = new TrackFusion();
      const rng = makeRng(7);
      for (let i = 0; i < 12; i++) {
        tf.updateTrack('t1', noisyFrame('3001', rng, 0.4), 0.6 + rng() * 0.2);
      }
      const fused = tf.getFusedEmbedding('t1')!;
      expect(fused).not.toBeNull();
      const norm = Math.sqrt(dot(fused, fused));
      expect(norm).toBeCloseTo(1, 5);
    });

    it('normalises a non-unit input embedding defensively (and copies it)', () => {
      const tf = new TrackFusion();
      const raw = Float32Array.from([3, 0, 0, 0, 0, 0, 0, 0]); // norm 3, not 1
      tf.updateTrack('t1', raw, 0.9);
      // Stored copy is normalised: fused of a single frame is the unit vector.
      const fused = tf.getFusedEmbedding('t1')!;
      expect(Math.sqrt(dot(fused, fused))).toBeCloseTo(1, 6);
      // Mutating the caller's buffer afterwards must not affect stored state.
      raw[1] = 99;
      expect(tf.getFusedEmbedding('t1')![1]).toBe(0);
    });

    it('caps retained frames at maxFrames (ring buffer)', () => {
      const tf = new TrackFusion({ maxFrames: 4 });
      const rng = makeRng(1);
      for (let i = 0; i < 10; i++) {
        tf.updateTrack('t1', noisyFrame('3001', rng, 0.3), 0.7);
      }
      expect(tf.frameCount('t1')).toBe(4);
    });
  });

  describe('(a) fusion beats a noisy single frame', () => {
    it('fused top-1 is correct even when most single frames mis-rank the part', () => {
      const tf = new TrackFusion();
      const rng = makeRng(42);
      const TRUTH = '3003'; // the look-alike that frames confuse with 3004
      const N = 16;

      let wrongSingleFrames = 0;
      for (let i = 0; i < N; i++) {
        // Heavy per-frame noise so single frames often flip 3003<->3004.
        const frame = noisyFrame(TRUTH, rng, 0.6);
        if (singleFrameTop1(frame) !== TRUTH) wrongSingleFrames++;
        // maxSim = the frame's own best gallery score (what the recipe weights by).
        const maxSim = gallerySearch(frame, 1)[0].score;
        tf.updateTrack('track-A', frame, maxSim);
      }

      // Precondition for the test to be meaningful: single frames really are noisy.
      expect(wrongSingleFrames).toBeGreaterThan(0);

      const fusedTop = tf.fusedTopK('track-A', gallerySearch);
      expect(fusedTop[0].partNum).toBe(TRUTH);
      // And the fused #1 actually outscores the look-alike #2.
      expect(fusedTop[0].score).toBeGreaterThan(fusedTop[1].score);
    });

    it('confidence weighting lets a few high-maxSim views overrule many low-maxSim wrong ones', () => {
      const tf = new TrackFusion();
      const rng = makeRng(99);

      // 10 LOW-confidence frames pointing (noisily) at the WRONG look-alike 3004,
      // then 4 HIGH-confidence clean frames of the TRUE part 3003. Plain mean
      // pooling would be dragged toward 3004; softmax(maxSim*20) should let the
      // confident minority win.
      for (let i = 0; i < 10; i++) {
        const frame = noisyFrame('3004', rng, 0.5);
        tf.updateTrack('track-B', frame, 0.30); // deliberately low confidence
      }
      for (let i = 0; i < 4; i++) {
        const frame = noisyFrame('3003', rng, 0.05); // near-clean
        tf.updateTrack('track-B', frame, 0.95); // high confidence
      }

      const fusedTop = tf.fusedTopK('track-B', gallerySearch);
      expect(fusedTop[0].partNum).toBe('3003');
    });
  });

  describe('(b) isCommitted flips true only after stability + margin', () => {
    // Use well-separated parts so once stable, the margin clears tau easily.
    const opts = { commitStability: 4, commitMargin: 0.05, topK: 5 };

    it('stays false until the stability window is filled', () => {
      const tf = new TrackFusion(opts);
      const rng = makeRng(3);
      // Three confident, consistent frames + searches: one short of N=4.
      for (let i = 0; i < 3; i++) {
        tf.updateTrack('t1', noisyFrame('3001', rng, 0.05), 0.95);
        tf.fusedTopK('t1', gallerySearch);
      }
      expect(tf.isCommitted('t1', gallerySearch)).toBe(false);

      // Fourth consistent search completes the window -> commit.
      tf.updateTrack('t1', noisyFrame('3001', rng, 0.05), 0.95);
      tf.fusedTopK('t1', gallerySearch);
      expect(tf.isCommitted('t1', gallerySearch)).toBe(true);
    });

    it('does NOT commit while the fused top-1 is still churning', () => {
      const tf = new TrackFusion(opts);
      const rng = makeRng(5);
      // Alternate which part dominates each frame so fused top-1 keeps flipping.
      // Big swings (high weight on the alternating part) move the fused vector
      // across the 3001/3002 boundary frame to frame.
      const seq = ['3001', '3002', '3001', '3002', '3001', '3002'];
      const top1s: string[] = [];
      for (let i = 0; i < seq.length; i++) {
        // Inject a strong pull toward seq[i] with high confidence weight.
        tf.updateTrack('t1', noisyFrame(seq[i], rng, 0.02), 0.99);
        // Also clear older frames' influence by capping the window small.
        const top = tf.fusedTopK('t1', gallerySearch);
        top1s.push(top[0].partNum);
        expect(tf.isCommitted('t1', gallerySearch)).toBe(false);
      }
      // Sanity: the fused top-1 genuinely changed across the run (it churned).
      expect(new Set(top1s).size).toBeGreaterThan(1);
    });

    it('does NOT commit when stable but the margin is below tau (look-alikes)', () => {
      // Demand a large margin the near-identical look-alikes can't meet.
      const tf = new TrackFusion({ commitStability: 4, commitMargin: 0.5, topK: 5 });
      const rng = makeRng(11);
      for (let i = 0; i < 6; i++) {
        // Clean, consistent views of 3003 -> stable top-1 == 3003 ...
        tf.updateTrack('t1', noisyFrame('3003', rng, 0.03), 0.9);
        tf.fusedTopK('t1', gallerySearch);
      }
      // Stability is satisfied (top-1 is consistently 3003) ...
      // but 3003 vs the look-alike 3004 margin is tiny, well under 0.5.
      const top = gallerySearch(tf.getFusedEmbedding('t1')!, 5);
      expect(top[0].partNum).toBe('3003');
      expect(top[0].score - top[1].score).toBeLessThan(0.5);
      expect(tf.isCommitted('t1', gallerySearch)).toBe(false);
    });

    it('commits once both stability AND margin are satisfied', () => {
      const tf = new TrackFusion(opts);
      const rng = makeRng(13);
      let committedAt = -1;
      for (let i = 0; i < 6; i++) {
        tf.updateTrack('t1', noisyFrame('3001', rng, 0.05), 0.95);
        tf.fusedTopK('t1', gallerySearch);
        if (committedAt < 0 && tf.isCommitted('t1', gallerySearch)) committedAt = i;
      }
      // Commit must occur, and not before the N-window is filled (index N-1).
      expect(committedAt).toBe(opts.commitStability - 1);
      expect(tf.isCommitted('t1', gallerySearch)).toBe(true);
    });

    it('forget() / reset() clear committed state', () => {
      const tf = new TrackFusion(opts);
      const rng = makeRng(17);
      for (let i = 0; i < 5; i++) {
        tf.updateTrack('t1', noisyFrame('3001', rng, 0.05), 0.95);
        tf.fusedTopK('t1', gallerySearch);
      }
      expect(tf.isCommitted('t1', gallerySearch)).toBe(true);
      tf.forget('t1');
      expect(tf.frameCount('t1')).toBe(0);
      expect(tf.isCommitted('t1', gallerySearch)).toBe(false);
    });
  });

  describe('defaults', () => {
    it('exposes sane defaults matching the fusion recipe', () => {
      expect(DEFAULT_FUSION_OPTS.softmaxScale).toBe(20.0); // fusion_eval.py temperature
      expect(DEFAULT_FUSION_OPTS.commitStability).toBe(4);
      expect(DEFAULT_FUSION_OPTS.commitMargin).toBeGreaterThan(0);
    });
  });
});
