/**
 * On-device color classifier — a pure-TS port of the Phase-1 LDA-kNN color
 * matcher (ml/scripts/color_model.py + color_eval.py). Given a brick crop it
 * returns the most likely Rebrickable color id, matching the reference Python
 * inference (84.2% top-1 / 90.8% top-3 on color_v1/val).
 *
 * Pipeline (must mirror color_eval.py / color_model.py exactly):
 *   resize-to-128 crop (caller's responsibility) → shades-of-gray white balance
 *   → border-frame background estimate → keep pixels far from background
 *   → drop specular highlights + deep shadow → sRGB→CIELAB → 12-d rich feature
 *   → z-score → LDA projection ((z - xbar) @ scalings)
 *   → distance-weighted kNN (k=3) over LDA-projected real exemplars.
 *
 * The model (z-score stats, LDA transform, the projected gallery, extraction
 * params, color metadata) is bundled as assets/models/color_model.json, built
 * by ml/scripts/color_model_to_json.py from the npz artifact. We reimplement
 * the feature extraction in TS (no skimage); the CIELAB conversion mirrors
 * skimage.color.rgb2lab (sRGB→XYZ D65→Lab) and the percentile/median calls
 * mirror numpy's linear-interpolation semantics — both validated in
 * src/__tests__/ml/colorClassifier.test.ts.
 *
 * CAVEAT: this is a from-scratch port of skimage/numpy, so it is not guaranteed
 * bit-for-bit. See the test file header for the measured agreement.
 */

// Bundled model asset. Metro inlines JSON via resolveJsonModule (see tsconfig).
import modelAsset from '../../assets/models/color_model.json';

/** Expected asset schema version — keep in lockstep with color_model_to_json.py. */
export const COLOR_ASSET_VERSION = 1;

export interface ColorPrediction {
  /** Rebrickable color id (string, e.g. "4" for Red). */
  colorId: string;
  /** Human-readable color name, e.g. "Red". */
  name: string;
  /** 6-hex color (no leading '#'), e.g. "C91A09". */
  hex: string;
  /** Distance-weighted kNN vote mass for this id (higher = stronger). */
  score: number;
}

export interface ColorClassifyResult {
  /** Best color id, or '' if no brick body could be segmented. */
  colorId: string;
  /** Best color name, or '' if none. */
  name: string;
  /** Best color's vote score, or 0 if none. */
  score: number;
  /** Ranked top-k predictions (best first). Empty if no body segmented. */
  topk: ColorPrediction[];
}

interface ExtractionParams {
  imgSize: number;
  wb: boolean;
  wbP: number;
  borderFrac: number;
  fgThresh: number;
  hiV: number;
  hiS: number;
  loV: number;
}

// Extraction params from the bundled asset (the model bakes these in; see
// color_model.DEFAULTS). Used as the default for the standalone extractFeature.
const DEFAULT_PARAMS: ExtractionParams = {
  imgSize: modelAsset.extraction.imgSize,
  wb: modelAsset.extraction.wb,
  wbP: modelAsset.extraction.wbP,
  borderFrac: modelAsset.extraction.borderFrac,
  fgThresh: modelAsset.extraction.fgThresh,
  hiV: modelAsset.extraction.hiV,
  hiS: modelAsset.extraction.hiS,
  loV: modelAsset.extraction.loV,
};

// ===========================================================================
// sRGB → CIELAB (mirrors skimage.color.rgb2lab; illuminant D65, observer 2°)
// ===========================================================================

// skimage's xyz_from_rgb matrix (sRGB / IEC 61966-2-1). Row-major; xyz = M·rgb.
const XYZ_FROM_RGB = [
  [0.412453, 0.357580, 0.180423],
  [0.212671, 0.715160, 0.072169],
  [0.019334, 0.119193, 0.950227],
] as const;

// D65 white point (skimage illuminants[('D65','2')]).
const D65 = [0.95047, 1.0, 1.08883] as const;

// CIE Lab f() threshold/slope (delta = 6/29).
const LAB_DELTA = 6.0 / 29.0;
const LAB_DELTA3 = LAB_DELTA * LAB_DELTA * LAB_DELTA;
const LAB_SLOPE = 3.0 * LAB_DELTA * LAB_DELTA; // 3·delta²

/** sRGB inverse companding for a single channel in [0,1] → linear-light. */
function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

function labF(t: number): number {
  return t > LAB_DELTA3 ? Math.cbrt(t) : t / LAB_SLOPE + 4.0 / 29.0;
}

/**
 * Convert a single sRGB pixel in [0,1] to CIELAB. Mirrors skimage.color.rgb2lab
 * (which composes rgb2xyz then xyz2lab). Returns [L, a, b].
 */
export function srgbToLab(r: number, g: number, b: number): [number, number, number] {
  const lr = srgbToLinear(r);
  const lg = srgbToLinear(g);
  const lb = srgbToLinear(b);
  const x = (XYZ_FROM_RGB[0][0] * lr + XYZ_FROM_RGB[0][1] * lg + XYZ_FROM_RGB[0][2] * lb) / D65[0];
  const y = (XYZ_FROM_RGB[1][0] * lr + XYZ_FROM_RGB[1][1] * lg + XYZ_FROM_RGB[1][2] * lb) / D65[1];
  const z = (XYZ_FROM_RGB[2][0] * lr + XYZ_FROM_RGB[2][1] * lg + XYZ_FROM_RGB[2][2] * lb) / D65[2];
  const fx = labF(x);
  const fy = labF(y);
  const fz = labF(z);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

// ===========================================================================
// numpy-compatible reductions
// ===========================================================================

/**
 * numpy.percentile with the default 'linear' interpolation, for a list of
 * values and a single percentile q in [0,100]. Sorts a copy ascending, then
 * interpolates at virtual index (n-1)·q/100.
 */
export function percentileLinear(values: number[], q: number): number {
  const n = values.length;
  if (n === 0) return NaN;
  if (n === 1) return values[0];
  const sorted = values.slice().sort((a, b) => a - b);
  const pos = ((n - 1) * q) / 100;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  const frac = pos - lo;
  return sorted[lo] * (1 - frac) + sorted[hi] * frac;
}

/** numpy.median for a 1-D array (linear interpolation == average of middle two). */
export function median(values: number[]): number {
  return percentileLinear(values, 50);
}

// ===========================================================================
// Feature extraction (mirrors brick_body_lab + rich_feature in color_model.py)
// ===========================================================================

/** A flat RGB pixel buffer in [0,1], row-major, with explicit length n*3. */
interface Rgb01 {
  data: Float64Array; // [n*3]
  n: number;
}

/**
 * Shades-of-gray (Minkowski-norm) white balance, in place on an Rgb01 buffer.
 * illum_c = mean(channel^p)^(1/p), normalized so mean(illum)=1; out = img/illum,
 * clipped to [0,1]. Matches color_model.shades_of_gray_wb.
 */
function shadesOfGrayWb(img: Rgb01, p: number): void {
  const { data, n } = img;
  let s0 = 0;
  let s1 = 0;
  let s2 = 0;
  for (let i = 0; i < n; i++) {
    const j = i * 3;
    s0 += Math.pow(data[j], p);
    s1 += Math.pow(data[j + 1], p);
    s2 += Math.pow(data[j + 2], p);
  }
  let i0 = Math.pow(s0 / n, 1 / p);
  let i1 = Math.pow(s1 / n, 1 / p);
  let i2 = Math.pow(s2 / n, 1 / p);
  const meanIllum = (i0 + i1 + i2) / 3 + 1e-8;
  i0 = i0 / meanIllum + 1e-8;
  i1 = i1 / meanIllum + 1e-8;
  i2 = i2 / meanIllum + 1e-8;
  for (let i = 0; i < n; i++) {
    const j = i * 3;
    data[j] = Math.min(1, Math.max(0, data[j] / i0));
    data[j + 1] = Math.min(1, Math.max(0, data[j + 1] / i1));
    data[j + 2] = Math.min(1, Math.max(0, data[j + 2] / i2));
  }
}

/** HSV value (max channel) for one RGB pixel. */
function hsvV(r: number, g: number, b: number): number {
  return Math.max(r, g, b);
}

/** HSV saturation (chroma/max) matching color_model._v_s. */
function hsvS(r: number, g: number, b: number): number {
  const mx = Math.max(r, g, b);
  const mn = Math.min(r, g, b);
  return mx > 1e-6 ? (mx - mn) / (mx + 1e-6) : 0.0;
}

interface BrickBody {
  bodyLab: Float64Array; // [k*3]
  k: number;
  bgLab: [number, number, number];
  fgFrac: number;
}

/**
 * Extract brick-body CIELAB pixels + context from an RGBA crop. Mirrors
 * color_model.brick_body_lab: white-balance → border-frame background median →
 * keep pixels with RGB distance > fgThresh from bg (center-crop fallback) →
 * drop specular highlights (bright + low chroma) and deep shadow (with a
 * "skip the guard if it removes too much" fallback).
 *
 * @param rgba   row-major RGBA Uint8 (or Uint8Clamped) of width*height*4
 * @param width  crop width  (should equal extraction imgSize, typically 128)
 * @param height crop height
 */
function brickBodyLab(
  rgba: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
  p: ExtractionParams,
): BrickBody | null {
  const n = width * height;
  // RGBA u8 → RGB f64 [0,1].
  const img: Rgb01 = { data: new Float64Array(n * 3), n };
  for (let i = 0; i < n; i++) {
    const s = i * 4;
    const d = i * 3;
    img.data[d] = rgba[s] / 255;
    img.data[d + 1] = rgba[s + 1] / 255;
    img.data[d + 2] = rgba[s + 2] / 255;
  }

  if (p.wb) shadesOfGrayWb(img, p.wbP);
  const data = img.data;

  // Border frame for the background estimate. Python concatenates the top b
  // rows, bottom b rows, left b cols, right b cols (corners counted twice), then
  // takes the per-channel median. We collect the same multiset of pixels.
  const b = Math.max(2, Math.floor(Math.min(height, width) * p.borderFrac));
  const br: number[] = [];
  const bg_: number[] = [];
  const bb: number[] = [];
  const pushPx = (idx: number) => {
    const j = idx * 3;
    br.push(data[j]);
    bg_.push(data[j + 1]);
    bb.push(data[j + 2]);
  };
  // Top b rows and bottom b rows (full width).
  for (let y = 0; y < b; y++) {
    for (let x = 0; x < width; x++) pushPx(y * width + x);
  }
  for (let y = height - b; y < height; y++) {
    for (let x = 0; x < width; x++) pushPx(y * width + x);
  }
  // Left b cols and right b cols (full height).
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < b; x++) pushPx(y * width + x);
  }
  for (let y = 0; y < height; y++) {
    for (let x = width - b; x < width; x++) pushPx(y * width + x);
  }
  const bg: [number, number, number] = [median(br), median(bg_), median(bb)];

  // Foreground mask: RGB euclidean distance from bg > fgThresh.
  const fgIdx: number[] = [];
  for (let i = 0; i < n; i++) {
    const j = i * 3;
    const dr = data[j] - bg[0];
    const dg = data[j + 1] - bg[1];
    const db = data[j + 2] - bg[2];
    if (Math.sqrt(dr * dr + dg * dg + db * db) > p.fgThresh) fgIdx.push(i);
  }
  const fgFrac = fgIdx.length / n;

  // Build the foreground pixel list (or center-crop fallback when too little fg).
  let fg: Float64Array;
  let fgN: number;
  if (fgIdx.length < 0.04 * n) {
    const cy0 = Math.trunc(height * 0.3);
    const cy1 = Math.trunc(height * 0.7);
    const cx0 = Math.trunc(width * 0.3);
    const cx1 = Math.trunc(width * 0.7);
    const idxs: number[] = [];
    for (let y = cy0; y < cy1; y++) {
      for (let x = cx0; x < cx1; x++) idxs.push(y * width + x);
    }
    fgN = idxs.length;
    fg = new Float64Array(fgN * 3);
    for (let t = 0; t < fgN; t++) {
      const j = idxs[t] * 3;
      fg[t * 3] = data[j];
      fg[t * 3 + 1] = data[j + 1];
      fg[t * 3 + 2] = data[j + 2];
    }
  } else {
    fgN = fgIdx.length;
    fg = new Float64Array(fgN * 3);
    for (let t = 0; t < fgN; t++) {
      const j = fgIdx[t] * 3;
      fg[t * 3] = data[j];
      fg[t * 3 + 1] = data[j + 1];
      fg[t * 3 + 2] = data[j + 2];
    }
  }

  // Highlight/shadow guard: drop (v>hiV & s<hiS) and (v<loV). Skip the guard
  // entirely if it would keep fewer than max(20, 0.15*fgN) pixels.
  const keep = new Uint8Array(fgN);
  let keepCount = 0;
  for (let t = 0; t < fgN; t++) {
    const r = fg[t * 3];
    const g = fg[t * 3 + 1];
    const bl = fg[t * 3 + 2];
    const v = hsvV(r, g, bl);
    const s = hsvS(r, g, bl);
    const drop = ((v > p.hiV && s < p.hiS) || v < p.loV);
    if (!drop) {
      keep[t] = 1;
      keepCount++;
    }
  }
  const minKeep = Math.max(20, Math.trunc(0.15 * fgN));
  const useAll = keepCount < minKeep;
  const bodyN = useAll ? fgN : keepCount;
  if (bodyN === 0) return null;

  // sRGB→Lab the body pixels.
  const bodyLab = new Float64Array(bodyN * 3);
  let w = 0;
  for (let t = 0; t < fgN; t++) {
    if (!useAll && !keep[t]) continue;
    const lab = srgbToLab(fg[t * 3], fg[t * 3 + 1], fg[t * 3 + 2]);
    bodyLab[w * 3] = lab[0];
    bodyLab[w * 3 + 1] = lab[1];
    bodyLab[w * 3 + 2] = lab[2];
    w++;
  }

  const bgLab = srgbToLab(bg[0], bg[1], bg[2]);
  return { bodyLab, k: bodyN, bgLab, fgFrac };
}

/**
 * 12-d color feature from brick-body LAB pixels. Mirrors color_model.rich_feature:
 * median L,a,b; L 10/90 pct; chroma median/std; a IQR; b IQR; L std; body↔bg
 * LAB distance; foreground fraction.
 */
function richFeature(body: BrickBody): Float64Array {
  const { bodyLab, k, bgLab, fgFrac } = body;
  const L = new Array<number>(k);
  const a = new Array<number>(k);
  const bb = new Array<number>(k);
  const chroma = new Array<number>(k);
  for (let i = 0; i < k; i++) {
    const li = bodyLab[i * 3];
    const ai = bodyLab[i * 3 + 1];
    const bi = bodyLab[i * 3 + 2];
    L[i] = li;
    a[i] = ai;
    bb[i] = bi;
    chroma[i] = Math.sqrt(ai * ai + bi * bi);
  }
  const medL = median(L);
  const medA = median(a);
  const medB = median(bb);
  const Lp10 = percentileLinear(L, 10);
  const Lp90 = percentileLinear(L, 90);
  const medChroma = median(chroma);
  const stdChroma = std(chroma);
  const aIqr = percentileLinear(a, 75) - percentileLinear(a, 25);
  const bIqr = percentileLinear(bb, 75) - percentileLinear(bb, 25);
  const stdL = std(L);
  const dr = medL - bgLab[0];
  const dg = medA - bgLab[1];
  const db = medB - bgLab[2];
  const bgDist = Math.sqrt(dr * dr + dg * dg + db * db);

  return Float64Array.from([
    medL, medA, medB,
    Lp10, Lp90,
    medChroma, stdChroma,
    aIqr, bIqr,
    stdL,
    bgDist,
    fgFrac,
  ]);
}

/**
 * Extract the raw 12-d color feature from an RGBA crop, or null if no brick
 * body could be segmented. Same pipeline `classify` runs internally; exposed
 * so callers (e.g. a fusion model) and parity tests can use the feature vector
 * directly. Feature layout matches color_model.rich_feature:
 *   [medL, medA, medB, Lp10, Lp90, medChroma, stdChroma, aIQR, bIQR, stdL,
 *    body↔bg LAB dist, fgFrac].
 *
 * @param rgba   row-major RGBA bytes, length width*height*4 (alpha ignored)
 * @param p      extraction params; defaults to the bundled model's params
 */
export function extractFeature(
  rgba: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
  p: ExtractionParams = DEFAULT_PARAMS,
): Float64Array | null {
  const expected = width * height * 4;
  if (rgba.length !== expected) {
    throw new Error(
      `extractFeature: expected ${expected} RGBA bytes (${width}x${height}x4), got ${rgba.length}`,
    );
  }
  const body = brickBodyLab(rgba, width, height, p);
  if (body === null || body.k === 0) return null;
  return richFeature(body);
}

/** numpy.std (population, ddof=0). */
function std(values: number[]): number {
  const n = values.length;
  if (n === 0) return 0;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += values[i];
  mean /= n;
  let v = 0;
  for (let i = 0; i < n; i++) {
    const d = values[i] - mean;
    v += d * d;
  }
  return Math.sqrt(v / n);
}

// ===========================================================================
// base64 → typed array (mirrors nativePixelBridge.decodeTensorBase64's atob path)
// ===========================================================================

function b64ToBytes(b64: string): Uint8Array {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const g: any = globalThis;
  if (typeof g.atob === 'function') {
    const bin: string = g.atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return bytes;
  }
  // Manual decode fallback (non-Hermes test/runtime).
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

/** Decode a little-endian int16 base64 blob into an Int16Array. */
function b64ToInt16(b64: string): Int16Array {
  const bytes = b64ToBytes(b64);
  // Copy into an aligned buffer; the source may not be 2-byte aligned.
  const aligned = new Uint8Array(bytes.length);
  aligned.set(bytes);
  return new Int16Array(aligned.buffer, 0, aligned.byteLength >> 1);
}

/** Decode a little-endian uint16 base64 blob into a Uint16Array. */
function b64ToUint16(b64: string): Uint16Array {
  const bytes = b64ToBytes(b64);
  const aligned = new Uint8Array(bytes.length);
  aligned.set(bytes);
  return new Uint16Array(aligned.buffer, 0, aligned.byteLength >> 1);
}

// ===========================================================================
// The classifier
// ===========================================================================

export class ColorClassifier {
  private readonly featMu: Float64Array;
  private readonly featSd: Float64Array;
  private readonly ldaXbar: Float64Array;
  private readonly ldaScalings: Float64Array; // [featDim*ldaDim] row-major
  private readonly featDim: number;
  private readonly ldaDim: number;

  private readonly galleryProj: Float64Array; // [count*ldaDim] dequantized
  private readonly galleryYIdx: Uint16Array; // [count] index into classes
  private readonly galleryCount: number;
  private readonly knnK: number;

  private readonly classes: string[];
  private readonly params: ExtractionParams;

  private readonly nameById: Record<string, string>;
  private readonly hexById: Record<string, string>;

  constructor(asset: typeof modelAsset = modelAsset) {
    if (asset.assetVersion !== COLOR_ASSET_VERSION) {
      throw new Error(
        `colorClassifier: asset version ${asset.assetVersion} != expected ${COLOR_ASSET_VERSION}`,
      );
    }
    this.featMu = Float64Array.from(asset.featMu);
    this.featSd = Float64Array.from(asset.featSd);
    this.ldaXbar = Float64Array.from(asset.ldaXbar);
    this.featDim = asset.featDim;
    this.ldaDim = asset.ldaDim;
    // Flatten scalings [featDim][ldaDim] → row-major Float64Array.
    this.ldaScalings = new Float64Array(this.featDim * this.ldaDim);
    for (let i = 0; i < this.featDim; i++) {
      const row = asset.ldaScalings[i];
      for (let j = 0; j < this.ldaDim; j++) {
        this.ldaScalings[i * this.ldaDim + j] = row[j];
      }
    }

    this.galleryCount = asset.gallery.count;
    const i16 = b64ToInt16(asset.gallery.projI16B64);
    const projScale = asset.gallery.projScale;
    this.galleryProj = new Float64Array(i16.length);
    for (let i = 0; i < i16.length; i++) this.galleryProj[i] = i16[i] / projScale;
    this.galleryYIdx = b64ToUint16(asset.gallery.yIdxU16B64);
    this.knnK = asset.knnK;

    this.classes = asset.classes.slice();
    this.params = {
      imgSize: asset.extraction.imgSize,
      wb: asset.extraction.wb,
      wbP: asset.extraction.wbP,
      borderFrac: asset.extraction.borderFrac,
      fgThresh: asset.extraction.fgThresh,
      hiV: asset.extraction.hiV,
      hiS: asset.extraction.hiS,
      loV: asset.extraction.loV,
    };

    this.nameById = {};
    this.hexById = {};
    for (let i = 0; i < asset.colorIds.length; i++) {
      this.nameById[asset.colorIds[i]] = asset.colorNames[i];
      this.hexById[asset.colorIds[i]] = asset.colorHex[i];
    }
  }

  /** The crop size the model expects (resize the crop to imgSize×imgSize). */
  get inputSize(): number {
    return this.params.imgSize;
  }

  /**
   * Project a 12-d feature into LDA space: z-score then (z - xbar) @ scalings.
   * Mirrors ColorClassifier.project in color_model.py.
   */
  project(feat: ArrayLike<number>): Float64Array {
    const z = new Float64Array(this.featDim);
    for (let i = 0; i < this.featDim; i++) {
      z[i] = (feat[i] - this.featMu[i]) / this.featSd[i] - this.ldaXbar[i];
    }
    const out = new Float64Array(this.ldaDim);
    for (let j = 0; j < this.ldaDim; j++) {
      let acc = 0;
      for (let i = 0; i < this.featDim; i++) {
        acc += z[i] * this.ldaScalings[i * this.ldaDim + j];
      }
      out[j] = acc;
    }
    return out;
  }

  /**
   * Distance-weighted kNN ranking for one projected query. Reproduces
   * color_model.ColorClassifier._rank: weight = 1/distance summed per class,
   * descending; ties broken by class order (stable argsort of -acc).
   * Returns {id, score} pairs for every class with non-zero vote mass, best first.
   */
  private rank(qProj: Float64Array): { id: string; score: number }[] {
    const count = this.galleryCount;
    const dim = this.ldaDim;
    const dist = new Float64Array(count);
    for (let i = 0; i < count; i++) {
      let s = 0;
      const base = i * dim;
      for (let j = 0; j < dim; j++) {
        const d = this.galleryProj[base + j] - qProj[j];
        s += d * d;
      }
      dist[i] = Math.sqrt(s);
    }

    const k = Math.min(this.knnK, count);
    // Indices of the k smallest distances. k is tiny (3), so a partial
    // selection over a single pass is cheaper and clearer than a full sort.
    const nn = kSmallestIndices(dist, k);

    const acc = new Float64Array(this.classes.length);
    for (const idx of nn) {
      const d = dist[idx];
      const w = 1.0 / (d === 0 ? 1e-300 : d);
      acc[this.galleryYIdx[idx]] += w;
    }

    // Stable descending sort by accumulated weight; ties keep class order
    // (matches numpy argsort(-acc), which is stable). Only emit voted classes.
    const order: number[] = [];
    for (let c = 0; c < acc.length; c++) if (acc[c] > 0) order.push(c);
    order.sort((x, y) => (acc[y] - acc[x]) || (x - y));
    return order.map((c) => ({ id: this.classes[c], score: acc[c] }));
  }

  /**
   * Classify an RGBA crop. The crop should already be resized to
   * `inputSize`×`inputSize` (the model's imgSize, 128) in RGBA row-major order;
   * the alpha channel is ignored. Returns the best color id plus a ranked
   * top-k. If no brick body can be segmented, returns empty (`colorId: ''`).
   *
   * @param rgba   row-major RGBA bytes, length width*height*4
   * @param width  crop width
   * @param height crop height
   * @param topk   number of ranked predictions to return (default 3)
   */
  classify(
    rgba: Uint8Array | Uint8ClampedArray,
    width: number,
    height: number,
    topk = 3,
  ): ColorClassifyResult {
    const expected = width * height * 4;
    if (rgba.length !== expected) {
      throw new Error(
        `colorClassifier.classify: expected ${expected} RGBA bytes (${width}x${height}x4), got ${rgba.length}`,
      );
    }
    const body = brickBodyLab(rgba, width, height, this.params);
    if (body === null || body.k === 0) {
      return { colorId: '', name: '', score: 0, topk: [] };
    }
    const feat = richFeature(body);
    const ranked = this.rank(this.project(feat)).slice(0, topk);
    const preds: ColorPrediction[] = ranked.map((r) => ({
      colorId: r.id,
      name: this.nameById[r.id] ?? r.id,
      hex: this.hexById[r.id] ?? '',
      score: r.score,
    }));
    if (preds.length === 0) {
      return { colorId: '', name: '', score: 0, topk: [] };
    }
    return {
      colorId: preds[0].colorId,
      name: preds[0].name,
      score: preds[0].score,
      topk: preds,
    };
  }

  /**
   * Lower-level entry point: classify directly from a precomputed 12-d feature
   * vector. Exposed mainly for tests / parity checks against the Python model.
   */
  classifyFeature(feat: ArrayLike<number>, topk = 3): ColorPrediction[] {
    const ranked = this.rank(this.project(feat)).slice(0, topk);
    return ranked.map((r) => ({
      colorId: r.id,
      name: this.nameById[r.id] ?? r.id,
      hex: this.hexById[r.id] ?? '',
      score: r.score,
    }));
  }
}

/**
 * Indices of the k smallest values in `arr`, unordered among themselves (k is
 * expected to be small, e.g. 3). Linear scan maintaining a small candidate set;
 * matches the *set* np.argpartition returns (order within the set is irrelevant
 * to the kNN vote accumulator, which sums over the set).
 */
function kSmallestIndices(arr: Float64Array, k: number): number[] {
  const n = arr.length;
  if (k >= n) {
    const all = new Array<number>(n);
    for (let i = 0; i < n; i++) all[i] = i;
    return all;
  }
  // Maintain a list of k current-smallest (index) sorted ascending by value.
  const idx: number[] = [];
  let worst = Infinity;
  for (let i = 0; i < n; i++) {
    const v = arr[i];
    if (idx.length < k) {
      // Insert keeping ascending order by value.
      let p = idx.length;
      idx.push(i);
      while (p > 0 && arr[idx[p - 1]] > arr[idx[p]]) {
        const t = idx[p - 1];
        idx[p - 1] = idx[p];
        idx[p] = t;
        p--;
      }
      worst = arr[idx[idx.length - 1]];
    } else if (v < worst) {
      // Replace the current worst, then bubble into position.
      idx[k - 1] = i;
      let p = k - 1;
      while (p > 0 && arr[idx[p - 1]] > arr[idx[p]]) {
        const t = idx[p - 1];
        idx[p - 1] = idx[p];
        idx[p] = t;
        p--;
      }
      worst = arr[idx[k - 1]];
    }
  }
  return idx;
}

/** Default singleton, ready to use with the bundled asset. */
export const colorClassifier = new ColorClassifier();
