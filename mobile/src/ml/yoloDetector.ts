/**
 * On-device YOLOv8 detector — loads the int8-quantized ONNX and runs it via
 * onnxruntime-react-native. Produces `RawDetection[]` matching the shape
 * `bboxTracker.updateTracks` expects, so the Phase 2 tracking pipeline works
 * unchanged.
 *
 * Feature-flagged off by default. See SettingsScreen → "On-device detection".
 * If this module fails to load the model, callers fall back to
 * `apiClient.scanMultiPiece` — parity preserved.
 */

import { computeLetterbox, hwcRgbToNchwFloat32, letterboxRgba, YOLO_INPUT_SIZE } from './preprocess';
import { decodeYoloOutput } from './postprocess';
import type { DetectorOptions, DetectResult, RawDetection } from './types';
import { DEFAULT_DETECTOR_OPTS } from './types';

// We reference onnxruntime-react-native lazily so the module can be
// imported on release builds / Expo Go without the native lib crashing on
// load. The actual require happens inside loadSession().
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type OrtModule = any;

export class YoloDetector {
  private session: OrtModule | null = null;
  private inputName = 'images';
  private outputName = 'output0';
  private readonly opts: DetectorOptions;
  private readonly classNames: readonly string[];

  constructor(
    classNames: readonly string[],
    opts: Partial<DetectorOptions> = {},
  ) {
    this.classNames = classNames;
    this.opts = { ...DEFAULT_DETECTOR_OPTS, ...opts };
  }

  /**
   * Load the ONNX session from a local file URI (e.g. resolved Asset or a
   * file in DocumentDirectory after first-launch CDN download). Returns
   * `false` on failure — caller is expected to fall back to the backend.
   */
  async load(modelUri: string): Promise<boolean> {
    try {
      // Lazy require so we don't crash on Expo Go / before pods install.
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
      console.warn('[YoloDetector] load failed:', err);
      this.session = null;
      return false;
    }
  }

  isReady(): boolean {
    return this.session !== null;
  }

  /**
   * Run detection on one RGBA frame. Caller provides the raw pixel buffer
   * plus original dimensions — typically from an expo-camera frame after
   * an ImageManipulator resize. Returns normalised bboxes in original-image
   * coordinates.
   */
  async detect(
    rgba: Uint8Array,
    origW: number,
    origH: number,
  ): Promise<DetectResult> {
    if (!this.session) {
      throw new Error('YoloDetector: not loaded. Call load() first.');
    }
    const tStart = Date.now();

    // 1. Letterbox RGBA → RGB HWC
    const tPre0 = Date.now();
    const { rgb, info } = letterboxRgba(rgba, origW, origH, YOLO_INPUT_SIZE);
    const input = hwcRgbToNchwFloat32(rgb, YOLO_INPUT_SIZE);
    const preprocessMs = Date.now() - tPre0;

    // 2. Run session
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const ort: OrtModule = require('onnxruntime-react-native');
    const tensor = new ort.Tensor('float32', input, [1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE]);
    const feeds: Record<string, unknown> = { [this.inputName]: tensor };
    const tInf0 = Date.now();
    const outputs = await this.session.run(feeds);
    const inferenceMs = Date.now() - tInf0;

    const raw = outputs[this.outputName] ?? outputs[Object.keys(outputs)[0]];
    if (!raw || !raw.data || !raw.dims) {
      throw new Error('YoloDetector: unexpected session output shape');
    }

    // 3. Decode + NMS
    const tPost0 = Date.now();
    const { detections, preNmsCount } = decodeYoloOutput(
      raw.data as Float32Array,
      raw.dims as number[],
      {
        scale: info.scale,
        padX: info.padX,
        padY: info.padY,
        resizedW: info.resizedW,
        resizedH: info.resizedH,
        inputSize: info.inputSize,
        origW,
        origH,
      },
      this.opts,
      this.classNames,
    );
    const postprocessMs = Date.now() - tPost0;

    return {
      detections,
      metrics: {
        preprocessMs,
        inferenceMs,
        postprocessMs,
        totalMs: Date.now() - tStart,
        preNmsCount,
        postNmsCount: detections.length,
      },
    };
  }

  async dispose(): Promise<void> {
    if (this.session?.release) {
      try { await this.session.release(); } catch { /* noop */ }
    }
    this.session = null;
  }
}

/**
 * Build a `DetectedPiece`-shaped stub from a `RawDetection`. The tracker
 * and overlay expect this shape; when running on-device we fill the
 * classifier-owned fields with the 28-class YOLO label as a starting point,
 * then a downstream refinement (backend classifier or cached lookup) can
 * replace it in a later frame's update.
 */
export function rawDetectionToStubPiece(
  det: RawDetection,
  pieceIndex: number,
): {
  pieceIndex: number;
  bbox: number[];
  predictions: {
    partNum: string; partName: string; colorId: string;
    colorName: string; colorHex: string; confidence: number; source?: string;
  }[];
  primaryPrediction: {
    partNum: string; partName: string; colorId: string;
    colorName: string; colorHex: string; confidence: number; source?: string;
  };
} {
  const { size, color } = parseYoloClass(det.className);
  const primary = {
    partNum: sizeToPartNum(size),
    partName: `Brick ${size}`,
    colorId: '',
    colorName: color,
    colorHex: colorNameToHex(color),
    confidence: det.score,
    source: 'on-device-yolo',
  };
  return {
    pieceIndex,
    bbox: det.bbox as number[],
    predictions: [primary],
    primaryPrediction: primary,
  };
}

function parseYoloClass(className: string): { size: string; color: string } {
  const m = /^([0-9]+x[0-9]+)_([a-z]+)$/.exec(className);
  if (!m) return { size: className, color: '' };
  return { size: m[1], color: m[2] };
}

// Standard LEGO part numbers for plain bricks — these match the Rebrickable
// canonical IDs used throughout the app.
const BRICK_PART_NUMS: Record<string, string> = {
  '1x1': '3005',
  '1x2': '3004',
  '2x1': '3004',
  '2x2': '3003',
  '2x3': '3002',
  '2x4': '3001',
};

function sizeToPartNum(size: string): string {
  return BRICK_PART_NUMS[size] ?? size;
}

// Only the colours present in the 28-class Roboflow Hex:Lego label set.
const COLOR_HEX: Record<string, string> = {
  black: '#05131D',
  blue: '#0055BF',
  brown: '#582A12',
  green: '#237841',
  pink: '#FC97AC',
  red: '#C91A09',
  yellow: '#F2CD37',
};

function colorNameToHex(name: string): string {
  return COLOR_HEX[name] ?? '#CCCCCC';
}
