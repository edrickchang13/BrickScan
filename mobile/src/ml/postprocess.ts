/**
 * YOLOv8 detection-head decode + NMS.
 *
 * The Ultralytics YOLOv8 ONNX head output has shape [1, 4+C, N]:
 *   rows 0..3 → cxcywh in letterboxed pixel space (0..inputSize)
 *   rows 4..3+C → per-class scores (already sigmoided in default exports)
 *   N = number of anchor positions (8400 for 640×640 on a YOLOv8-L).
 *
 * We decode candidates above `scoreThreshold`, run greedy NMS per-class-
 * agnostic, map boxes back through the letterbox to the original image,
 * and return normalised [x1,y1,x2,y2] in [0,1].
 */

import type { LetterboxInfo } from './preprocess';
import type { NormalizedBbox, RawDetection } from './types';

/**
 * Decode YOLOv8 head output.
 *
 * @param output   Flat Float32Array of length (4+numClasses) * numAnchors.
 *                 Laid out row-major with rows=(4+numClasses), cols=numAnchors.
 * @param shape    [1, 4+numClasses, numAnchors]
 * @param info     Letterbox info from `computeLetterbox` used at preprocess.
 * @param classNames  Optional label lookup by class index.
 */
export function decodeYoloOutput(
  output: Float32Array,
  shape: readonly number[],
  info: LetterboxInfo,
  opts: { scoreThreshold: number; iouThreshold: number; maxDetections: number },
  classNames?: readonly string[],
): { detections: RawDetection[]; preNmsCount: number } {
  if (shape.length !== 3 || shape[0] !== 1) {
    throw new Error(`decodeYoloOutput: expected [1, 4+C, N], got [${shape.join(',')}]`);
  }
  const rows = shape[1];
  const cols = shape[2];
  const numClasses = rows - 4;
  if (numClasses <= 0) {
    throw new Error(`decodeYoloOutput: invalid class count ${numClasses}`);
  }
  if (output.length !== rows * cols) {
    throw new Error(
      `decodeYoloOutput: output length ${output.length} != ${rows}*${cols}=${rows * cols}`,
    );
  }

  // Stride into the flat array: output[r * cols + c].
  // We walk anchors (c) and gather cx,cy,w,h + max class.
  const candidates: {
    cx: number; cy: number; w: number; h: number;
    score: number; classIdx: number;
  }[] = [];
  const { scoreThreshold } = opts;

  const needsSigmoid = detectNeedsSigmoid(output, rows, cols, numClasses);

  for (let c = 0; c < cols; c++) {
    // Find top-1 class for this anchor.
    let best = -Infinity;
    let bestIdx = -1;
    for (let k = 0; k < numClasses; k++) {
      const raw = output[(4 + k) * cols + c];
      if (raw > best) {
        best = raw;
        bestIdx = k;
      }
    }
    const score = needsSigmoid ? sigmoid(best) : best;
    if (score < scoreThreshold) continue;
    candidates.push({
      cx: output[c],
      cy: output[cols + c],
      w: output[2 * cols + c],
      h: output[3 * cols + c],
      score,
      classIdx: bestIdx,
    });
  }

  const preNmsCount = candidates.length;

  // Sort by score desc for NMS.
  candidates.sort((a, b) => b.score - a.score);

  const kept: typeof candidates = [];
  for (const cand of candidates) {
    if (kept.length >= opts.maxDetections) break;
    const candBox: [number, number, number, number] = [
      cand.cx - cand.w / 2,
      cand.cy - cand.h / 2,
      cand.cx + cand.w / 2,
      cand.cy + cand.h / 2,
    ];
    let suppressed = false;
    for (const k of kept) {
      const kBox: [number, number, number, number] = [
        k.cx - k.w / 2,
        k.cy - k.h / 2,
        k.cx + k.w / 2,
        k.cy + k.h / 2,
      ];
      if (iou(candBox, kBox) >= opts.iouThreshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) kept.push(cand);
  }

  // Map letterboxed xyxy → original image xyxy → normalised [0,1].
  const detections: RawDetection[] = kept.map(k => {
    const x1Pix = k.cx - k.w / 2;
    const y1Pix = k.cy - k.h / 2;
    const x2Pix = k.cx + k.w / 2;
    const y2Pix = k.cy + k.h / 2;
    const bbox = unletterbox([x1Pix, y1Pix, x2Pix, y2Pix], info);
    return {
      bbox,
      classIdx: k.classIdx,
      className: classNames?.[k.classIdx] ?? '',
      score: k.score,
    };
  });

  return { detections, preNmsCount };
}

/**
 * Heuristic: if any raw class "logit" in the first few anchors is outside
 * [0,1], the export didn't bake sigmoid into the head and we need to apply
 * it ourselves. Newer Ultralytics exports (>= 8.2) do apply sigmoid; older
 * ones don't. Cheaper than assuming one way and being wrong silently.
 */
function detectNeedsSigmoid(
  output: Float32Array, rows: number, cols: number, numClasses: number,
): boolean {
  const probe = Math.min(cols, 64);
  for (let c = 0; c < probe; c++) {
    for (let k = 0; k < numClasses; k++) {
      const v = output[(4 + k) * cols + c];
      if (v < 0 || v > 1) return true;
    }
  }
  return false;
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function iou(
  a: readonly [number, number, number, number],
  b: readonly [number, number, number, number],
): number {
  const ix1 = Math.max(a[0], b[0]);
  const iy1 = Math.max(a[1], b[1]);
  const ix2 = Math.min(a[2], b[2]);
  const iy2 = Math.min(a[3], b[3]);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  const aArea = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const bArea = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const union = aArea + bArea - inter;
  return union > 0 ? inter / union : 0;
}

/**
 * Reverse the letterbox transform: a box in the input-canvas coordinate
 * frame is rescaled back to the original image frame, then normalised to
 * [0, 1] against the original image dimensions.
 */
function unletterbox(
  pix: readonly [number, number, number, number],
  info: LetterboxInfo,
): NormalizedBbox {
  const { scale, padX, padY, origW, origH } = info;
  const x1 = Math.max(0, Math.min(origW, (pix[0] - padX) / scale));
  const y1 = Math.max(0, Math.min(origH, (pix[1] - padY) / scale));
  const x2 = Math.max(0, Math.min(origW, (pix[2] - padX) / scale));
  const y2 = Math.max(0, Math.min(origH, (pix[3] - padY) / scale));
  return [x1 / origW, y1 / origH, x2 / origW, y2 / origH];
}
