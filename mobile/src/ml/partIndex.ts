/**
 * Bundled gallery index + cosine k-NN retrieval.
 *
 * Phase 2 interface contract (fixed):
 *   L2-normalized float query embedding (from `embeddingRetrieval`)
 *     ->  cosine k-NN over the bundled gallery
 *     ->  top-k { partNum, score }
 *
 * Index format & quantization mirror `ml/scripts/ondevice_index.py`:
 *   - Gallery vectors are stored INT8 with a SINGLE GLOBAL scale (1/127), not
 *     per-vector. For L2-normalized embeddings every component is already in
 *     [-1, 1], so a global scale is near-optimal AND preserves cosine ranking;
 *     per-vector scaling broke ranking and tanked top-1 in Phase 1 testing
 *     (ONDEVICE_NOTES #1).
 *   - Retrieval is ASYMMETRIC: the query stays FLOAT (fresh from the model) and
 *     is scored against the dequantized int8 gallery (ONDEVICE_NOTES #2). This
 *     was the best int8 variant and avoids round-tripping the query.
 *   - Since the query is L2-normalized and the gallery's global scale is a
 *     positive constant, `score = query · (gallery_q * scale)` is monotonic in
 *     the true cosine, so ranking is exact and `score` is the cosine estimate.
 *
 * Bundled wire format (`PartIndexData`) is plain JSON so it ships through
 * Expo's asset/`resolveJsonModule` pipeline. Int8 vectors are base64-packed to
 * keep the file ~4x smaller than a float JSON array and avoid a giant number
 * literal list. A tiny synthetic stub (`SYNTHETIC_STUB_INDEX`) is included so
 * this module is testable now; the real gallery drops in once Phase 1 lands.
 */

/** One retrieval hit: a catalog part number and its cosine similarity in [-1,1]. */
export interface PartMatch {
  partNum: string;
  score: number;
}

/**
 * Serialized, bundle-ready index. `vectors` is base64 of a row-major int8
 * [count, dim] matrix; `partNums[i]` is the catalog part for gallery row i.
 * `scale` is the single global dequant factor (q * scale ≈ original float).
 */
export interface PartIndexData {
  /** Format version so the loader can reject incompatible future layouts. */
  version: 1;
  /** Embedding dimensionality D (must match the student model's output). */
  dim: number;
  /** Number of gallery vectors (rows). Equals `partNums.length`. */
  count: number;
  /** Single global int8 dequant scale, typically 1/127. */
  scale: number;
  /** base64 of the row-major int8 [count*dim] gallery matrix. */
  vectors: string;
  /** Per-row catalog part numbers, length === count. */
  partNums: string[];
}

/** Default global int8 scale used by the Phase 1 exporter (1/127). */
export const DEFAULT_INT8_SCALE = 1 / 127;

export class PartIndex {
  private readonly dim: number;
  private readonly count: number;
  private readonly scale: number;
  /** Row-major dequantized gallery, [count*dim] float32. */
  private readonly gallery: Float32Array;
  private readonly partNums: string[];

  private constructor(
    dim: number,
    count: number,
    scale: number,
    gallery: Float32Array,
    partNums: string[],
  ) {
    this.dim = dim;
    this.count = count;
    this.scale = scale;
    this.gallery = gallery;
    this.partNums = partNums;
  }

  /**
   * Build an index from the bundled JSON. Decodes the base64 int8 matrix and
   * dequantizes it once (q * scale) into a float gallery so each `search` is a
   * plain float dot product — the realistic asymmetric on-device path.
   *
   * Throws on a malformed/incompatible payload so a bad bundle fails loudly at
   * load time rather than silently returning wrong neighbours.
   */
  static fromData(data: PartIndexData): PartIndex {
    if (data.version !== 1) {
      throw new Error(`PartIndex: unsupported index version ${data.version}`);
    }
    if (data.partNums.length !== data.count) {
      throw new Error(
        `PartIndex: partNums length ${data.partNums.length} != count ${data.count}`,
      );
    }
    const q = base64ToInt8(data.vectors);
    const expected = data.count * data.dim;
    if (q.length !== expected) {
      throw new Error(
        `PartIndex: decoded ${q.length} int8 values, expected count*dim=${expected}`,
      );
    }
    // Dequantize once: gallery[i] = q[i] * scale. Cheap and lets every query be
    // a float dot product (asymmetric path).
    const gallery = new Float32Array(q.length);
    const scale = data.scale;
    for (let i = 0; i < q.length; i++) gallery[i] = q[i] * scale;
    return new PartIndex(data.dim, data.count, scale, gallery, data.partNums.slice());
  }

  /** Convenience: load the bundled synthetic stub. Replaced by the real gallery. */
  static synthetic(): PartIndex {
    return PartIndex.fromData(SYNTHETIC_STUB_INDEX);
  }

  dimension(): number {
    return this.dim;
  }

  size(): number {
    return this.count;
  }

  /**
   * Cosine k-NN. `vec` must be the L2-normalized FLOAT query embedding from the
   * student model with the same dimensionality as the index. Returns the top-k
   * matches sorted by descending score (cosine similarity).
   *
   * Implementation note: a full scan with per-row dot products. The gallery is
   * row-major so each row is contiguous (cache-friendly), and a bounded
   * insertion into a small top-k buffer beats sorting all `count` scores when
   * k << count — which is the norm for catalog retrieval.
   */
  search(vec: Float32Array, k: number): PartMatch[] {
    if (vec.length !== this.dim) {
      throw new Error(
        `PartIndex.search: query dim ${vec.length} != index dim ${this.dim}`,
      );
    }
    const kk = Math.max(1, Math.min(k, this.count));
    if (this.count === 0) return [];

    // Top-k via a tiny ascending-by-score buffer: buf[0] is the current
    // weakest kept hit. We only touch the heap-ish array when a score beats it.
    const idxBuf = new Int32Array(kk).fill(-1);
    const scoreBuf = new Float32Array(kk).fill(-Infinity);
    let filled = 0;

    const { gallery, dim } = this;
    for (let row = 0; row < this.count; row++) {
      const base = row * dim;
      let dot = 0;
      for (let d = 0; d < dim; d++) dot += vec[d] * gallery[base + d];

      if (filled < kk) {
        insertSorted(scoreBuf, idxBuf, filled, dot, row);
        filled++;
      } else if (dot > scoreBuf[0]) {
        // Drop the weakest (index 0) and insert the new score in order.
        replaceWeakest(scoreBuf, idxBuf, kk, dot, row);
      }
    }

    // Buffer is ascending; emit descending (best first).
    const out: PartMatch[] = new Array(filled);
    for (let i = 0; i < filled; i++) {
      const src = filled - 1 - i;
      out[i] = { partNum: this.partNums[idxBuf[src]], score: scoreBuf[src] };
    }
    return out;
  }
}

// ── int8 packing helpers ──────────────────────────────────────────────────────

/**
 * Quantize L2-normalized float vectors to int8 with a single GLOBAL scale and
 * return the bundle-ready JSON. This is the TS mirror of
 * `ondevice_index.quantize_int8(mode="global")`; useful for generating a real
 * `PartIndexData` from float gallery embeddings inside the app or a JS tool.
 *
 * `vectors` is a row-major [count*dim] float buffer (each `dim`-slice an
 * embedding). The scale is fixed global (1/127) on purpose — see the file
 * header and ONDEVICE_NOTES #1.
 */
export function buildIndexData(
  vectors: Float32Array,
  partNums: string[],
  dim: number,
  scale: number = DEFAULT_INT8_SCALE,
): PartIndexData {
  const count = partNums.length;
  if (vectors.length !== count * dim) {
    throw new Error(
      `buildIndexData: vectors length ${vectors.length} != count*dim=${count * dim}`,
    );
  }
  const q = new Int8Array(vectors.length);
  for (let i = 0; i < vectors.length; i++) {
    // round-to-nearest, clamp to [-127,127] (matches numpy proof).
    const r = Math.round(vectors[i] / scale);
    q[i] = r < -127 ? -127 : r > 127 ? 127 : r;
  }
  return {
    version: 1,
    dim,
    count,
    scale,
    vectors: int8ToBase64(q),
    partNums: partNums.slice(),
  };
}

/** Encode an int8 array as base64 (two's-complement bytes). */
export function int8ToBase64(q: Int8Array): string {
  const bytes = new Uint8Array(q.buffer, q.byteOffset, q.byteLength);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const g: any = globalThis;
  if (typeof g.btoa === 'function') {
    let bin = '';
    // Chunk to avoid blowing the call stack on String.fromCharCode(...big).
    const CHUNK = 0x8000;
    for (let i = 0; i < bytes.length; i += CHUNK) {
      bin += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
    }
    return g.btoa(bin);
  }
  // Node / RN fallback.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const B: any = (g as any).Buffer;
  if (B) return B.from(bytes).toString('base64');
  throw new Error('int8ToBase64: no btoa or Buffer available');
}

/** Decode base64 → int8 array (reinterpreting bytes as two's-complement). */
export function base64ToInt8(b64: string): Int8Array {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const g: any = globalThis;
  let bytes: Uint8Array;
  if (typeof g.atob === 'function') {
    const bin = g.atob(b64);
    bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  } else {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const B: any = (g as any).Buffer;
    if (!B) throw new Error('base64ToInt8: no atob or Buffer available');
    const buf = B.from(b64, 'base64');
    bytes = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  }
  return new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
}

// ── top-k buffer primitives (ascending by score; buf[0] is weakest) ───────────

/** Insert (score,idx) into the first `filled` slots, keeping ascending order. */
function insertSorted(
  scores: Float32Array,
  idxs: Int32Array,
  filled: number,
  score: number,
  idx: number,
): void {
  let p = filled;
  while (p > 0 && scores[p - 1] > score) {
    scores[p] = scores[p - 1];
    idxs[p] = idxs[p - 1];
    p--;
  }
  scores[p] = score;
  idxs[p] = idx;
}

/** Drop slot 0 (weakest) and insert (score,idx) into the full `k`-slot buffer. */
function replaceWeakest(
  scores: Float32Array,
  idxs: Int32Array,
  k: number,
  score: number,
  idx: number,
): void {
  // Shift everything left until we find the insertion point, then place.
  let p = 0;
  while (p + 1 < k && scores[p + 1] < score) {
    scores[p] = scores[p + 1];
    idxs[p] = idxs[p + 1];
    p++;
  }
  scores[p] = score;
  idxs[p] = idx;
}

// ── synthetic stub index ──────────────────────────────────────────────────────

/**
 * A tiny, deterministic synthetic index so `PartIndex.search` is testable
 * before the real Phase 1 gallery exists. Six 4-D unit vectors over three
 * "parts": each part has two near-duplicate exemplars clustered around a
 * distinct axis, so a query near one axis retrieves that part's exemplars
 * first. Built via `buildIndexData` so the on-disk format is exercised too.
 *
 * Vectors (pre-quantization, L2-normalized):
 *   3001 (axis 0): [1,0,0,0]               and [0.980,0.198,0,0]
 *   3002 (axis 1): [0,1,0,0]               and [0,0.980,0.198,0]
 *   3003 (axis 2): [0,0,1,0]               and [0,0,0.980,0.198]
 */
export const SYNTHETIC_STUB_INDEX: PartIndexData = buildIndexData(
  new Float32Array([
    1.0, 0.0, 0.0, 0.0,
    0.9801, 0.1981, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.9801, 0.1981, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.9801, 0.1981,
  ]),
  ['3001', '3001', '3002', '3002', '3003', '3003'],
  4,
);
