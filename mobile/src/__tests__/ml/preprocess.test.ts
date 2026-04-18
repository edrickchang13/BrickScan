import {
  computeLetterbox,
  hwcRgbToNchwFloat32,
  letterboxRgba,
  PAD_COLOR,
  YOLO_INPUT_SIZE,
} from '../../ml/preprocess';

describe('ml/preprocess', () => {
  describe('computeLetterbox', () => {
    it('pads horizontally when image is portrait', () => {
      const info = computeLetterbox(480, 640, 640);
      expect(info.scale).toBe(1); // 640/640 = 1 (h limits)
      expect(info.resizedH).toBe(640);
      expect(info.resizedW).toBe(480);
      expect(info.padX).toBe(80);
      expect(info.padY).toBe(0);
    });

    it('pads vertically when image is landscape', () => {
      const info = computeLetterbox(640, 360, 640);
      expect(info.scale).toBe(1);
      expect(info.resizedW).toBe(640);
      expect(info.resizedH).toBe(360);
      expect(info.padX).toBe(0);
      expect(info.padY).toBe(140);
    });

    it('scales down images larger than the input size', () => {
      const info = computeLetterbox(1920, 1080, 640);
      expect(info.scale).toBeCloseTo(640 / 1920, 5);
      expect(info.resizedW).toBe(640);
      expect(info.resizedH).toBe(360);
    });

    it('defaults to 640 when size omitted', () => {
      const info = computeLetterbox(800, 600);
      expect(info.inputSize).toBe(YOLO_INPUT_SIZE);
    });
  });

  describe('hwcRgbToNchwFloat32', () => {
    it('deinterleaves channels and normalises to [0,1]', () => {
      const size = 2;
      // HWC: pixel(0,0) = [255, 128, 0]; pixel(0,1) = [0, 255, 64];
      //      pixel(1,0) = [32, 64, 128]; pixel(1,1) = [10, 20, 30];
      const hwc = new Uint8Array([
        255, 128, 0,
        0, 255, 64,
        32, 64, 128,
        10, 20, 30,
      ]);
      const out = hwcRgbToNchwFloat32(hwc, size);
      expect(out.length).toBe(3 * 4);
      // R plane
      expect(out[0]).toBeCloseTo(255 / 255);
      expect(out[1]).toBeCloseTo(0 / 255);
      expect(out[2]).toBeCloseTo(32 / 255);
      expect(out[3]).toBeCloseTo(10 / 255);
      // G plane
      expect(out[4]).toBeCloseTo(128 / 255);
      expect(out[5]).toBeCloseTo(255 / 255);
      expect(out[6]).toBeCloseTo(64 / 255);
      expect(out[7]).toBeCloseTo(20 / 255);
      // B plane
      expect(out[8]).toBeCloseTo(0 / 255);
      expect(out[9]).toBeCloseTo(64 / 255);
      expect(out[10]).toBeCloseTo(128 / 255);
      expect(out[11]).toBeCloseTo(30 / 255);
    });

    it('throws when buffer length mismatches size', () => {
      expect(() => hwcRgbToNchwFloat32(new Uint8Array(11), 2)).toThrow();
    });
  });

  describe('letterboxRgba', () => {
    it('fills pad region with PAD_COLOR', () => {
      // 4x2 RGBA image (landscape). Letterboxed to 4x4 → 1 row of pad top+bottom.
      const rgba = new Uint8Array(4 * 2 * 4);
      for (let i = 0; i < 4 * 2; i++) {
        rgba[i * 4] = 200; rgba[i * 4 + 1] = 100; rgba[i * 4 + 2] = 50; rgba[i * 4 + 3] = 255;
      }
      const { rgb, info } = letterboxRgba(rgba, 4, 2, 4);
      expect(info.scale).toBe(1);
      expect(info.resizedW).toBe(4);
      expect(info.resizedH).toBe(2);
      expect(info.padY).toBe(1);
      // First row = pad
      expect(rgb[0]).toBe(PAD_COLOR);
      expect(rgb[1]).toBe(PAD_COLOR);
      expect(rgb[2]).toBe(PAD_COLOR);
      // Second row (y=1 after padY=1) = content
      const contentRowStart = 4 * 3;
      expect(rgb[contentRowStart]).toBe(200);
      expect(rgb[contentRowStart + 1]).toBe(100);
      expect(rgb[contentRowStart + 2]).toBe(50);
      // Last row = pad again
      expect(rgb[rgb.length - 3]).toBe(PAD_COLOR);
    });

    it('returns square output at requested size', () => {
      const rgba = new Uint8Array(100 * 50 * 4);
      const { rgb } = letterboxRgba(rgba, 100, 50, 64);
      expect(rgb.length).toBe(64 * 64 * 3);
    });

    it('rejects buffers whose length does not match origW*origH*4', () => {
      expect(() => letterboxRgba(new Uint8Array(99), 10, 10, 16)).toThrow();
    });
  });
});
