import {
  PartIndex,
  buildIndexData,
  int8ToBase64,
  base64ToInt8,
  SYNTHETIC_STUB_INDEX,
  DEFAULT_INT8_SCALE,
  type PartIndexData,
} from '../../ml/partIndex';

/** L2-normalize a plain array into a Float32Array (test helper). */
function unit(v: number[]): Float32Array {
  const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
  return Float32Array.from(v.map((x) => x / n));
}

describe('ml/partIndex', () => {
  describe('int8 base64 round-trip', () => {
    it('preserves int8 values including negatives and bounds', () => {
      const q = Int8Array.from([0, 1, -1, 127, -127, -128, 42, -42]);
      const decoded = base64ToInt8(int8ToBase64(q));
      expect(Array.from(decoded)).toEqual(Array.from(q));
    });
  });

  describe('buildIndexData', () => {
    it('quantizes with a single global scale and packs the right shape', () => {
      const data = buildIndexData(
        unit([1, 0, 0, 0]),
        ['9999'],
        4,
      );
      expect(data.version).toBe(1);
      expect(data.dim).toBe(4);
      expect(data.count).toBe(1);
      expect(data.scale).toBeCloseTo(DEFAULT_INT8_SCALE, 8);
      expect(data.partNums).toEqual(['9999']);
      // [1,0,0,0] / (1/127) = [127,0,0,0]
      expect(Array.from(base64ToInt8(data.vectors))).toEqual([127, 0, 0, 0]);
    });

    it('throws when vectors length disagrees with count*dim', () => {
      expect(() => buildIndexData(new Float32Array(3), ['a'], 4)).toThrow();
    });
  });

  describe('PartIndex.fromData validation', () => {
    it('rejects an unsupported version', () => {
      const bad = { ...SYNTHETIC_STUB_INDEX, version: 2 } as unknown as PartIndexData;
      expect(() => PartIndex.fromData(bad)).toThrow(/version/);
    });

    it('rejects partNums/count mismatch', () => {
      const bad: PartIndexData = { ...SYNTHETIC_STUB_INDEX, count: 99 };
      expect(() => PartIndex.fromData(bad)).toThrow();
    });

    it('exposes dim and size from the synthetic stub', () => {
      const idx = PartIndex.synthetic();
      expect(idx.dimension()).toBe(4);
      expect(idx.size()).toBe(6);
    });
  });

  describe('search on the synthetic stub', () => {
    const idx = PartIndex.synthetic();

    it('returns the correct part for a query aligned to each axis', () => {
      expect(idx.search(unit([1, 0, 0, 0]), 1)[0].partNum).toBe('3001');
      expect(idx.search(unit([0, 1, 0, 0]), 1)[0].partNum).toBe('3002');
      expect(idx.search(unit([0, 0, 1, 0]), 1)[0].partNum).toBe('3003');
    });

    it('ranks both exemplars of the matching part above other parts', () => {
      // Query near axis 0 → both 3001 exemplars should be the top-2.
      const hits = idx.search(unit([1, 0, 0, 0]), 3);
      expect(hits.map((h) => h.partNum).slice(0, 2)).toEqual(['3001', '3001']);
      expect(hits[2].partNum).not.toBe('3001');
    });

    it('returns scores in descending order', () => {
      const hits = idx.search(unit([0.9, 0.3, 0, 0]), 6);
      for (let i = 1; i < hits.length; i++) {
        expect(hits[i - 1].score).toBeGreaterThanOrEqual(hits[i].score);
      }
    });

    it('top score equals cosine of the nearest exemplar (int8 dequant)', () => {
      // Exact match to the first stored vector [1,0,0,0] → quantizes to
      // [127,0,0,0], dequantizes to [1,0,0,0]; cosine with unit query = 1.
      const top = idx.search(unit([1, 0, 0, 0]), 1)[0];
      expect(top.score).toBeCloseTo(1, 5);
    });

    it('clamps k to the gallery size and never returns more than count', () => {
      const hits = idx.search(unit([1, 0, 0, 0]), 100);
      expect(hits).toHaveLength(6);
    });

    it('returns a single best match when k=1', () => {
      expect(idx.search(unit([0, 0, 0, 1]), 1)).toHaveLength(1);
    });
  });

  describe('search input validation', () => {
    it('throws when the query dimensionality does not match the index', () => {
      const idx = PartIndex.synthetic();
      expect(() => idx.search(new Float32Array(3), 1)).toThrow(/dim/);
    });

    it('returns [] for an empty index', () => {
      const empty = PartIndex.fromData(
        buildIndexData(new Float32Array(0), [], 4),
      );
      expect(empty.search(unit([1, 0, 0, 0]), 5)).toEqual([]);
    });
  });

  describe('asymmetric cosine ranking matches a float reference', () => {
    // Build a small random gallery, then confirm the int8 asymmetric search
    // (float query · dequantized int8 gallery) agrees with an exact float
    // cosine top-1 — the property ondevice_index.py validates (agree@1).
    it('agrees with float cosine top-1 across many random queries', () => {
      const dim = 16;
      const n = 40;
      const rng = mulberry32(1234);
      const floatRows: Float32Array[] = [];
      const flat = new Float32Array(n * dim);
      const partNums: string[] = [];
      for (let i = 0; i < n; i++) {
        const raw = Array.from({ length: dim }, () => rng() * 2 - 1);
        const u = unit(raw);
        floatRows.push(u);
        flat.set(u, i * dim);
        partNums.push(`p${i}`);
      }
      const idx = PartIndex.fromData(buildIndexData(flat, partNums, dim));

      let agree = 0;
      const trials = 50;
      for (let t = 0; t < trials; t++) {
        const qu = unit(Array.from({ length: dim }, () => rng() * 2 - 1));
        // Exact float cosine reference (vectors are unit, so cosine = dot).
        let bestF = -Infinity;
        let bestPart = '';
        for (let i = 0; i < n; i++) {
          let dot = 0;
          for (let d = 0; d < dim; d++) dot += qu[d] * floatRows[i][d];
          if (dot > bestF) {
            bestF = dot;
            bestPart = partNums[i];
          }
        }
        const top = idx.search(qu, 1)[0];
        if (top.partNum === bestPart) agree++;
        // int8 score should track the float cosine closely (global 1/127 scale).
        expect(Math.abs(top.score - bestF)).toBeLessThan(0.02);
      }
      // Quantization to int8 with a global scale should agree with float top-1
      // on essentially every well-separated random query.
      expect(agree / trials).toBeGreaterThanOrEqual(0.9);
    });
  });
});

/** Tiny deterministic PRNG so the random-gallery test is reproducible. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
