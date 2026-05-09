/**
 * Thin TS wrapper around the iOS native `PixelBridgeModule`.
 *
 * Behaviour:
 *  - On iOS with the native module linked: returns the prepared NCHW Float32
 *    tensor (as a base64 string) and letterbox geometry, all computed in
 *    Swift via vImage / Accelerate. ~8-15ms vs ~50-80ms in pure JS.
 *  - On Android, simulator builds without the module, or if pod install
 *    hasn't happened: `isAvailable()` returns false and callers fall back
 *    to the JS path in `preprocess.ts`.
 *
 * This file is import-safe: requiring it on a build without the native module
 * returns a stub. Never throws at import time.
 */
import { NativeModules, Platform } from 'react-native';

interface NativePixelBridge {
  prepareTensorFromJpegUri(
    uri: string,
    targetSize: number,
    fillR: number,
    fillG: number,
    fillB: number,
  ): Promise<{
    base64: string;
    width: number;
    height: number;
    letterboxScale: number;
    padX: number;
    padY: number;
    sourceWidth: number;
    sourceHeight: number;
  }>;
}

const native: NativePixelBridge | undefined =
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (NativeModules as any).PixelBridgeModule;

export interface PrepareTensorResult {
  /** Float32 NCHW tensor encoded as base64. JS decodes once and feeds to ORT. */
  tensorBase64: string;
  /** Width of the prepared (square) tensor — matches targetSize. */
  width: number;
  /** Height — same as width since output is square. */
  height: number;
  /** Letterbox scale factor used (min(target/srcW, target/srcH)). */
  letterboxScale: number;
  /** Letterbox padding offsets in target-image space. */
  padX: number;
  padY: number;
  /** Source image dims for un-letterbox in postprocess. */
  sourceWidth: number;
  sourceHeight: number;
}

export function isAvailable(): boolean {
  return Platform.OS === 'ios' && !!native?.prepareTensorFromJpegUri;
}

/**
 * Prepare a YOLO-ready tensor from a JPEG file URI.
 *
 * @param jpegUri        file:// URI or absolute path
 * @param targetSize     square model input dim, e.g. 640 for YOLOv8
 * @param fill           letterbox fill colour (default YOLO grey 114)
 */
export async function prepareTensor(
  jpegUri: string,
  targetSize: number = 640,
  fill: { r: number; g: number; b: number } = { r: 114, g: 114, b: 114 },
): Promise<PrepareTensorResult> {
  if (!native?.prepareTensorFromJpegUri) {
    throw new Error('Native PixelBridgeModule not available on this build');
  }
  const r = await native.prepareTensorFromJpegUri(
    jpegUri, targetSize, fill.r, fill.g, fill.b,
  );
  return {
    tensorBase64: r.base64,
    width: r.width,
    height: r.height,
    letterboxScale: r.letterboxScale,
    padX: r.padX,
    padY: r.padY,
    sourceWidth: r.sourceWidth,
    sourceHeight: r.sourceHeight,
  };
}

/**
 * Decode the base64 tensor blob from native into a Float32Array. Single
 * allocation; no per-pixel JS work.
 */
export function decodeTensorBase64(b64: string): Float32Array {
  // RN provides global atob via Hermes. Fall back to a manual decode if not.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const g: any = globalThis;
  let bytes: Uint8Array;
  if (typeof g.atob === 'function') {
    const bin = g.atob(b64);
    bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  } else {
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
    bytes = new Uint8Array(out);
  }
  // Reinterpret the byte buffer as a Float32 array. The buffer is
  // already in NCHW order from the native side.
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}
