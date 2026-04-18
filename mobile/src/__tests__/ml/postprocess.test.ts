import { computeLetterbox } from '../../ml/preprocess';
import { decodeYoloOutput } from '../../ml/postprocess';

function buildYoloOutput(
  numClasses: number,
  numAnchors: number,
  anchors: { cx: number; cy: number; w: number; h: number; scores: number[] }[],
): Float32Array {
  const rows = 4 + numClasses;
  const arr = new Float32Array(rows * numAnchors);
  for (let i = 0; i < anchors.length && i < numAnchors; i++) {
    const a = anchors[i];
    arr[0 * numAnchors + i] = a.cx;
    arr[1 * numAnchors + i] = a.cy;
    arr[2 * numAnchors + i] = a.w;
    arr[3 * numAnchors + i] = a.h;
    for (let k = 0; k < numClasses; k++) {
      arr[(4 + k) * numAnchors + i] = a.scores[k] ?? 0;
    }
  }
  return arr;
}

describe('ml/postprocess.decodeYoloOutput', () => {
  const info = computeLetterbox(640, 640, 640); // identity letterbox
  const opts = { scoreThreshold: 0.25, iouThreshold: 0.45, maxDetections: 10 };

  it('filters detections below the score threshold', () => {
    const out = buildYoloOutput(3, 4, [
      { cx: 100, cy: 100, w: 50, h: 50, scores: [0.10, 0.05, 0.01] }, // dropped
      { cx: 300, cy: 300, w: 40, h: 40, scores: [0.60, 0.10, 0.05] }, // kept
    ]);
    const { detections } = decodeYoloOutput(out, [1, 7, 4], info, opts, ['a', 'b', 'c']);
    expect(detections).toHaveLength(1);
    expect(detections[0].classIdx).toBe(0);
    expect(detections[0].score).toBeCloseTo(0.6, 3);
  });

  it('applies NMS to overlapping same-class boxes', () => {
    const out = buildYoloOutput(2, 3, [
      { cx: 100, cy: 100, w: 100, h: 100, scores: [0.9, 0] },
      { cx: 105, cy: 105, w: 100, h: 100, scores: [0.8, 0] }, // heavy overlap — suppress
      { cx: 400, cy: 400, w: 100, h: 100, scores: [0.7, 0] }, // distant — keep
    ]);
    const { detections } = decodeYoloOutput(out, [1, 6, 3], info, opts);
    expect(detections).toHaveLength(2);
    // Highest score first
    expect(detections[0].score).toBeCloseTo(0.9, 3);
    expect(detections[1].score).toBeCloseTo(0.7, 3);
  });

  it('converts cxcywh pixels to normalised xyxy', () => {
    const out = buildYoloOutput(1, 1, [{ cx: 320, cy: 320, w: 160, h: 160, scores: [0.9] }]);
    const { detections } = decodeYoloOutput(out, [1, 5, 1], info, opts);
    expect(detections).toHaveLength(1);
    const [x1, y1, x2, y2] = detections[0].bbox;
    expect(x1).toBeCloseTo(240 / 640, 3);
    expect(y1).toBeCloseTo(240 / 640, 3);
    expect(x2).toBeCloseTo(400 / 640, 3);
    expect(y2).toBeCloseTo(400 / 640, 3);
  });

  it('reverses letterbox padding when mapping back to original', () => {
    const lbInfo = computeLetterbox(320, 640, 640); // 40px pad on each x-side
    const out = buildYoloOutput(1, 1, [{ cx: 320, cy: 320, w: 160, h: 160, scores: [0.9] }]);
    const { detections } = decodeYoloOutput(out, [1, 5, 1], lbInfo, opts);
    // Center of canvas (320,320) → center of original (160, 320)
    // In original pixel coords: x1 = 80, x2 = 240, y1 = 240, y2 = 400
    // Normalised against origW=320, origH=640:
    const [x1, y1, x2, y2] = detections[0].bbox;
    expect(x1).toBeCloseTo(80 / 320, 3);
    expect(x2).toBeCloseTo(240 / 320, 3);
    expect(y1).toBeCloseTo(240 / 640, 3);
    expect(y2).toBeCloseTo(400 / 640, 3);
  });

  it('caps output at maxDetections', () => {
    const anchors = Array.from({ length: 20 }, (_, i) => ({
      cx: 30 + i * 40,
      cy: 30,
      w: 20,
      h: 20,
      scores: [0.9 - i * 0.01],
    }));
    const out = buildYoloOutput(1, 20, anchors);
    const capped = { ...opts, maxDetections: 5 };
    const { detections } = decodeYoloOutput(out, [1, 5, 20], info, capped);
    expect(detections.length).toBe(5);
  });

  it('applies sigmoid when raw class logits are outside [0,1]', () => {
    // Raw logit 2.0 → sigmoid ~0.88, which passes threshold 0.25.
    const out = buildYoloOutput(1, 1, [{ cx: 100, cy: 100, w: 50, h: 50, scores: [2.0] }]);
    const { detections } = decodeYoloOutput(out, [1, 5, 1], info, opts);
    expect(detections).toHaveLength(1);
    expect(detections[0].score).toBeCloseTo(1 / (1 + Math.exp(-2.0)), 3);
  });

  it('assigns className from lookup when provided', () => {
    const out = buildYoloOutput(3, 1, [{ cx: 50, cy: 50, w: 30, h: 30, scores: [0.1, 0.9, 0.1] }]);
    const { detections } = decodeYoloOutput(out, [1, 7, 1], info, opts, ['a', '2x4_blue', 'c']);
    expect(detections[0].className).toBe('2x4_blue');
    expect(detections[0].classIdx).toBe(1);
  });
});
