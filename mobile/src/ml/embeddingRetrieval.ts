/**
 * On-device embedding model — loads the distilled FastViT student (exported to
 * ONNX) and runs it via onnxruntime-react-native to turn a detected-piece crop
 * into an L2-normalized float embedding. That embedding feeds `partIndex` for
 * cosine k-NN retrieval against the bundled gallery.
 *
 * Phase 2 interface contract (fixed):
 *   RGBA crop  ->  letterbox/resize to 224, NCHW float  ->  student model
 *               ->  L2-normalized Float32Array (dim D)
 *
 * The real student model is produced by Phase 1 and is NOT bundled yet, so this
 * module is feature-flagged on a CONFIGURABLE model path and degrades
 * gracefully: `embed()` returns `null` (and logs once) when no session is
 * loaded. The session pattern mirrors `yoloDetector.ts`; preprocessing reuses
 * the letterbox/NCHW helpers in `preprocess.ts`.
 */

import { hwcRgbToNchwFloat32, letterboxRgba } from './preprocess';

// Lazy reference to onnxruntime-react-native so this module imports cleanly on
// Expo Go / release builds before the native lib is linked. The actual require
// happens inside load(). Same approach as yoloDetector.ts.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type OrtModule = any;

/** Square input size of the distilled student (DINOv2/FastViT family → 224). */
export const EMBED_INPUT_SIZE = 224;

export interface EmbeddingOptions {
  /**
   * Square model input size. The Phase 1 student has a STATIC 224 input (see
   * ONDEVICE_NOTES #3); exposed only so the index/model can be swapped without
   * a code change.
   */
  inputSize: number;
  /**
   * Per-channel mean subtracted from the [0,1] tensor before the model. Phase 1
   * folds ImageNet normalization INTO the exported graph (ONDEVICE_NOTES #4),
   * so the default is "no normalization here" ([0,0,0] / [1,1,1]). Provide
   * ImageNet stats only if a future export expects raw [0,1] input instead.
   */
  mean: readonly [number, number, number];
  /** Per-channel std the tensor is divided by. See `mean`. */
  std: readonly [number, number, number];
  /**
   * If the model does NOT L2-normalize its output internally, set this so the
   * wrapper normalizes before returning. The contract requires unit vectors;
   * normalizing an already-unit vector is a harmless no-op, so this defaults
   * to true to guarantee the contract regardless of the export.
   */
  l2Normalize: boolean;
}

export const DEFAULT_EMBEDDING_OPTS: EmbeddingOptions = {
  inputSize: EMBED_INPUT_SIZE,
  mean: [0, 0, 0],
  std: [1, 1, 1],
  l2Normalize: true,
};

export class EmbeddingModel {
  private session: OrtModule | null = null;
  private inputName = 'image';
  private outputName = 'embedding';
  private readonly opts: EmbeddingOptions;
  /** Output embedding dimension, discovered on first successful run. */
  private dim: number | null = null;
  /** Guards the "model missing" warning so it logs at most once per instance. */
  private warned = false;

  constructor(opts: Partial<EmbeddingOptions> = {}) {
    this.opts = { ...DEFAULT_EMBEDDING_OPTS, ...opts };
  }

  /**
   * Load the ONNX student from a local file URI (resolved Asset, or a file in
   * DocumentDirectory after a first-launch CDN download — same lifecycle as the
   * YOLO model). Returns `false` on failure; callers degrade gracefully rather
   * than throwing, since the student lands in a later phase.
   */
  async load(modelUri: string): Promise<boolean> {
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const ort: OrtModule = require('onnxruntime-react-native');
      this.session = await ort.InferenceSession.create(modelUri, {
        executionProviders: ['cpu'], // 'coreml' once the provider is bundled
        graphOptimizationLevel: 'all',
      });
      const ins = this.session.inputNames ?? [];
      const outs = this.session.outputNames ?? [];
      if (ins.length) this.inputName = ins[0];
      if (outs.length) this.outputName = outs[0];
      return true;
    } catch (err) {
      console.warn('[EmbeddingModel] load failed:', err);
      this.session = null;
      return false;
    }
  }

  isReady(): boolean {
    return this.session !== null;
  }

  /** Embedding dimension once known (after the first successful `embed`). */
  embeddingDim(): number | null {
    return this.dim;
  }

  /**
   * Embed one RGBA crop (HWC, 4 bytes/pixel — the canvas / `jpeg-js` decode
   * format). The crop is letterboxed to a square `inputSize` (aspect preserved,
   * grey-padded), converted to NCHW float in [0,1], optionally mean/std
   * normalized, run through the student, and L2-normalized.
   *
   * Returns the unit-norm embedding, or `null` when no model is loaded (the
   * graceful-fallback path — the caller skips retrieval for this crop). Throws
   * only on genuinely malformed input (wrong buffer length) or an unexpected
   * session output shape, mirroring `YoloDetector.detect`.
   */
  async embed(
    rgba: Uint8Array,
    width: number,
    height: number,
  ): Promise<Float32Array | null> {
    if (!this.session) {
      if (!this.warned) {
        console.warn(
          '[EmbeddingModel] embed() called before a model was loaded; ' +
            'returning null (student model not bundled yet).',
        );
        this.warned = true;
      }
      return null;
    }

    const size = this.opts.inputSize;

    // 1. Letterbox RGBA → RGB HWC, then deinterleave to NCHW float in [0,1].
    const { rgb } = letterboxRgba(rgba, width, height, size);
    const input = hwcRgbToNchwFloat32(rgb, size);

    // 2. Optional in-JS mean/std normalization. Skipped entirely when the
    //    export already folds normalization in (the default opts) so we don't
    //    pay a per-pixel pass for nothing.
    this.applyNormalization(input, size);

    // 3. Run the session.
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const ort: OrtModule = require('onnxruntime-react-native');
    const tensor = new ort.Tensor('float32', input, [1, 3, size, size]);
    const feeds: Record<string, unknown> = { [this.inputName]: tensor };
    const outputs = await this.session.run(feeds);

    const raw = outputs[this.outputName] ?? outputs[Object.keys(outputs)[0]];
    if (!raw || !raw.data) {
      throw new Error('EmbeddingModel: unexpected session output shape');
    }

    // ORT returns Float32Array for a float output; copy into a plain
    // Float32Array so we own the buffer (the session may reuse its own).
    const vec = Float32Array.from(raw.data as ArrayLike<number>);
    this.dim = vec.length;

    if (this.opts.l2Normalize) {
      l2NormalizeInPlace(vec);
    }
    return vec;
  }

  /**
   * Subtract per-channel mean and divide by per-channel std, in place, over an
   * NCHW [1,3,size,size] buffer. No-op when mean=[0,0,0] and std=[1,1,1].
   */
  private applyNormalization(input: Float32Array, size: number): void {
    const { mean, std } = this.opts;
    const isIdentity =
      mean[0] === 0 && mean[1] === 0 && mean[2] === 0 &&
      std[0] === 1 && std[1] === 1 && std[2] === 1;
    if (isIdentity) return;
    const plane = size * size;
    for (let c = 0; c < 3; c++) {
      const m = mean[c];
      const s = std[c] || 1;
      const base = c * plane;
      for (let i = 0; i < plane; i++) {
        input[base + i] = (input[base + i] - m) / s;
      }
    }
  }

  async dispose(): Promise<void> {
    if (this.session?.release) {
      try { await this.session.release(); } catch { /* noop */ }
    }
    this.session = null;
    this.dim = null;
  }
}

/**
 * L2-normalize a vector in place. A zero vector is left untouched (its norm is
 * clamped to 1) so we never produce NaNs from a degenerate embedding.
 */
export function l2NormalizeInPlace(vec: Float32Array): Float32Array {
  let sumSq = 0;
  for (let i = 0; i < vec.length; i++) sumSq += vec[i] * vec[i];
  const norm = Math.sqrt(sumSq);
  if (norm > 1e-8) {
    for (let i = 0; i < vec.length; i++) vec[i] /= norm;
  }
  return vec;
}
