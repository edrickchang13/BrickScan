/**
 * Unit tests for image utilities used in BrickScan
 */
import { getPartImageUrl, getSetImageUrl, colorHexToRGBA } from '../../utils/imageUtils';

describe('imageUtils', () => {
  describe('getPartImageUrl', () => {
    it('returns valid Rebrickable URL for standard part number', () => {
      const url = getPartImageUrl('3001');
      expect(url).toContain('rebrickable.com');
      expect(url).toContain('3001');
      expect(url).toMatch(/\.(jpg|png)$/i);
    });

    it('pads part numbers to 5 digits', () => {
      const url = getPartImageUrl('1');
      expect(url).toContain('00001');
    });

    it('handles part numbers with special characters', () => {
      const url = getPartImageUrl('3069b');
      expect(url).toBeDefined();
      expect(url).toContain('3069b');
    });

    it('handles long part numbers without modification', () => {
      const url = getPartImageUrl('32524b01');
      expect(url).toBeDefined();
      expect(url).toContain('32524b01');
    });

    it('generates valid URLs multiple times', () => {
      const url1 = getPartImageUrl('3001');
      const url2 = getPartImageUrl('3001');
      expect(url1).toBe(url2);
    });
  });

  describe('getSetImageUrl', () => {
    it('returns valid Rebrickable set URL', () => {
      const url = getSetImageUrl('75257-1');
      expect(url).toContain('rebrickable.com');
      expect(url).toContain('sets');
      expect(url).toContain('75257-1');
    });

    it('handles different set number formats', () => {
      const urls = [
        getSetImageUrl('10265-1'),
        getSetImageUrl('60198-1'),
        getSetImageUrl('42110-1'),
      ];

      urls.forEach(url => {
        expect(url).toContain('rebrickable.com');
        expect(url).toMatch(/sets\/\d+-\d/);
      });
    });

    it('handles sets without version number', () => {
      const url = getSetImageUrl('21318');
      expect(url).toBeDefined();
      expect(url).toContain('rebrickable.com');
    });
  });

  describe('colorHexToRGBA', () => {
    it('converts hex to rgba string with default alpha', () => {
      const result = colorHexToRGBA('FF0000');
      expect(result).toBe('rgba(255, 0, 0, 1)');
    });

    it('converts hex with custom alpha', () => {
      const result = colorHexToRGBA('FF0000', 0.5);
      expect(result).toBe('rgba(255, 0, 0, 0.5)');
    });

    it('handles alpha of 0', () => {
      const result = colorHexToRGBA('0000FF', 0);
      expect(result).toBe('rgba(0, 0, 255, 0)');
    });

    it('handles alpha of 1 explicitly', () => {
      const result = colorHexToRGBA('00FF00', 1);
      expect(result).toBe('rgba(0, 255, 0, 1)');
    });

    it('handles hash prefix in hex code', () => {
      const result = colorHexToRGBA('#0000FF');
      expect(result).toBe('rgba(0, 0, 255, 1)');
    });

    it('converts all white', () => {
      const result = colorHexToRGBA('FFFFFF');
      expect(result).toBe('rgba(255, 255, 255, 1)');
    });

    it('converts all black', () => {
      const result = colorHexToRGBA('000000');
      expect(result).toBe('rgba(0, 0, 0, 1)');
    });

    it('handles partial alpha values', () => {
      const result = colorHexToRGBA('FF00FF', 0.75);
      expect(result).toBe('rgba(255, 0, 255, 0.75)');
    });

    it('handles lowercase hex values', () => {
      const result = colorHexToRGBA('ff00ff');
      expect(result).toBe('rgba(255, 0, 255, 1)');
    });

    it('handles mixed case hex values', () => {
      const result = colorHexToRGBA('FfEe00');
      expect(result).toBe('rgba(255, 238, 0, 1)');
    });

    it('converts LEGO color examples', () => {
      const testCases = [
        { hex: 'FFFFFF', expected: 'rgba(255, 255, 255, 1)' }, // White
        { hex: 'C91A09', expected: 'rgba(201, 26, 9, 1)' }, // Red
        { hex: '1E5AA8', expected: 'rgba(30, 90, 168, 1)' }, // Blue
        { hex: '00852B', expected: 'rgba(0, 133, 43, 1)' }, // Green
        { hex: '212121', expected: 'rgba(33, 33, 33, 1)' }, // Black
      ];

      testCases.forEach(({ hex, expected }) => {
        const result = colorHexToRGBA(hex);
        expect(result).toBe(expected);
      });
    });
  });

  describe('edge cases', () => {
    it('getPartImageUrl handles empty string gracefully', () => {
      const url = getPartImageUrl('');
      expect(url).toBeDefined();
      expect(typeof url).toBe('string');
    });

    it('colorHexToRGBA handles edge alpha values', () => {
      const result1 = colorHexToRGBA('FF0000', 0.001);
      expect(result1).toContain('0.001');

      const result2 = colorHexToRGBA('FF0000', 0.999);
      expect(result2).toContain('0.999');
    });

    it('colorHexToRGBA handles very small hex values', () => {
      const result = colorHexToRGBA('000001');
      expect(result).toBe('rgba(0, 0, 1, 1)');
    });
  });

  describe('integration', () => {
    it('URLs can be used in image requests', () => {
      const partUrl = getPartImageUrl('3001');
      const setUrl = getSetImageUrl('75257-1');

      expect(partUrl).toMatch(/^https?:\/\//);
      expect(setUrl).toMatch(/^https?:\/\//);
    });

    it('color conversions work for styling', () => {
      const colors = ['FFFFFF', 'FF0000', '0000FF', '000000'];

      colors.forEach(hex => {
        const rgba = colorHexToRGBA(hex);
        expect(rgba).toMatch(/^rgba\(\d+,\s*\d+,\s*\d+,\s*[\d.]+\)$/);
      });
    });

    it('combined usage for part with color', () => {
      const partUrl = getPartImageUrl('3001');
      const partColor = colorHexToRGBA('C91A09', 0.8);

      expect(partUrl).toBeDefined();
      expect(partColor).toBeDefined();
      expect(partUrl).toContain('3001');
      expect(partColor).toBe('rgba(201, 26, 9, 0.8)');
    });
  });
});
