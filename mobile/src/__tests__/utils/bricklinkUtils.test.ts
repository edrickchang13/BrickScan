/**
 * Unit tests for BrickLink utilities used in BrickScan
 */
import {
  generateBricklinkPartUrl,
  formatMissingPartsText,
  generateWantedListXml,
  MissingPart,
} from '../../utils/bricklinkUtils';

describe('bricklinkUtils', () => {
  describe('generateBricklinkPartUrl', () => {
    it('generates valid BrickLink part URL without color', () => {
      const url = generateBricklinkPartUrl('3001');
      expect(url).toContain('bricklink.com');
      expect(url).toContain('3001');
      expect(url).toMatch(/^https?:\/\//);
    });

    it('includes color when provided', () => {
      const url = generateBricklinkPartUrl('3001', 5);
      expect(url).toContain('3001');
      expect(url).toContain('5');
    });

    it('handles part numbers with letters', () => {
      const url = generateBricklinkPartUrl('3069b', 1);
      expect(url).toContain('3069b');
      expect(url).toBeDefined();
    });

    it('generates different URLs for different colors', () => {
      const url1 = generateBricklinkPartUrl('3001', 1);
      const url2 = generateBricklinkPartUrl('3001', 5);
      expect(url1).not.toBe(url2);
    });

    it('handles undefined color gracefully', () => {
      const url = generateBricklinkPartUrl('3001', undefined);
      expect(url).toBeDefined();
      expect(url).toContain('3001');
    });

    it('generates valid BrickLink catalog URLs', () => {
      const url = generateBricklinkPartUrl('3001');
      expect(url).toContain('catalogitem');
      expect(url).toMatch(/[P=]/);
    });
  });

  describe('formatMissingPartsText', () => {
    it('returns no missing parts message for empty array', () => {
      const text = formatMissingPartsText([]);
      expect(text).toContain('missing');
      expect(text.toLowerCase()).toContain('no missing');
    });

    it('formats single missing part correctly', () => {
      const parts: MissingPart[] = [
        {
          partNum: '3001',
          colorName: 'Red',
          colorHex: '#C91A09',
          quantity: 4,
        },
      ];
      const text = formatMissingPartsText(parts);
      expect(text).toContain('3001');
      expect(text).toContain('Red');
      expect(text).toContain('4');
    });

    it('formats multiple missing parts', () => {
      const parts: MissingPart[] = [
        {
          partNum: '3001',
          colorName: 'Red',
          colorHex: '#C91A09',
          quantity: 4,
        },
        {
          partNum: '3002',
          colorName: 'Blue',
          colorHex: '#1E5AA8',
          quantity: 8,
        },
        {
          partNum: '3003',
          colorName: 'Black',
          colorHex: '#212121',
          quantity: 12,
        },
      ];
      const text = formatMissingPartsText(parts);
      expect(text).toContain('3001');
      expect(text).toContain('3002');
      expect(text).toContain('3003');
      expect(text).toContain('Red');
      expect(text).toContain('Blue');
      expect(text).toContain('Black');
    });

    it('includes quantity in formatted text', () => {
      const parts: MissingPart[] = [
        {
          partNum: '3001',
          colorName: 'Red',
          colorHex: '#C91A09',
          quantity: 50,
        },
      ];
      const text = formatMissingPartsText(parts);
      expect(text).toContain('50');
    });

    it('handles large quantities', () => {
      const parts: MissingPart[] = [
        {
          partNum: '3001',
          colorName: 'Red',
          colorHex: '#C91A09',
          quantity: 1000,
        },
      ];
      const text = formatMissingPartsText(parts);
      expect(text).toContain('1000');
    });
  });

  describe('generateWantedListXml', () => {
    it('generates valid XML for single part', () => {
      const parts: MissingPart[] = [
        {
          partNum: '3001',
          colorName: 'Red',
          colorHex: '#C91A09',
          quantity: 4,
        },
      ];
      const xml = generateWantedListXml(parts, 'Test Build');
      expect(xml).toContain('<?xml');
      expect(xml).toContain('WantedList');
      expect(xml).toContain('Test Build');
      expect(xml).toContain('3001');
      expect(xml).toContain('<Item>');
      expect(xml).toContain('</Item>');
    });

    it('generates valid XML for multiple parts', () => {
      const parts: MissingPart[] = [
        {
          partNum: '3001',
          colorName: 'Red',
          colorHex: '#C91A09',
          quantity: 4,
        },
        {
          partNum: '3002',
          colorName: 'Blue',
          colorHex: '#1E5AA8',
          quantity: 8,
        },
      ];
      const xml = generateWantedListXml(parts, 'Millennium Falcon');
      expect(xml).toContain('Millennium Falcon');
      expect(xml).toContain('3001');
      expect(xml).toContain('3002');
      expect(xml.match(/<Item>/g)?.length).toBe(2);
    });

    it('includes color mapping in XML', () => {
      const parts: MissingPart[] = [
        {
          partNum: '3001',
          colorName: 'White',
          colorHex: '#FFFFFF',
          quantity: 10,
        },
      ];
      const xml = generateWantedListXml(parts, 'Test');
      expect(xml).toContain('ColorID');
    });

    it('includes quantity in XML', () => {
      const parts: MissingPart[] = [
        {
          partNum: '3001',
          colorName: 'Red',
          colorHex: '#C91A09',
          quantity: 25,
        },
      ];
      const xml = generateWantedListXml(parts, 'Test');
      expect(xml).toContain('MinQty');
      expect(xml).toContain('25');
    });

    it('escapes special characters in list name', () => {
      const parts: MissingPart[] = [
        {
          partNum: '3001',
          colorName: 'Red',
          colorHex: '#C91A09',
          quantity: 4,
        },
      ];
      const xml = generateWantedListXml(parts, 'Test & Build <2024>');
      expect(xml).toContain('Test &amp; Build &lt;2024&gt;');
    });

    it('generates XML with proper structure', () => {
      const parts: MissingPart[] = [
        {
          partNum: '3001',
          colorName: 'Red',
          colorHex: '#C91A09',
          quantity: 4,
        },
      ];
      const xml = generateWantedListXml(parts, 'Test');

      // Verify XML structure
      expect(xml).toMatch(/^<\?xml/);
      expect(xml).toContain('<WantedList>');
      expect(xml).toContain('</WantedList>');
      expect(xml).toContain('<WantedListName>');
      expect(xml).toContain('<Items>');
      expect(xml).toContain('<Item>');
      expect(xml).toContain('</Item>');
      expect(xml).toContain('</Items>');
    });

    it('includes BrickLink required fields', () => {
      const parts: MissingPart[] = [
        {
          partNum: '3001',
          colorName: 'Red',
          colorHex: '#C91A09',
          quantity: 4,
        },
      ];
      const xml = generateWantedListXml(parts, 'Test');
      expect(xml).toContain('ItemID');
      expect(xml).toContain('ItemTypeID');
      expect(xml).toContain('ColorID');
      expect(xml).toContain('MinQty');
    });

    it('handles empty parts array', () => {
      const xml = generateWantedListXml([], 'Empty Build');
      expect(xml).toContain('Empty Build');
      expect(xml).toContain('<Items>');
      expect(xml).toContain('</Items>');
    });

    it('generates correct ItemTypeID for parts', () => {
      const parts: MissingPart[] = [
        {
          partNum: '3001',
          colorName: 'Red',
          colorHex: '#C91A09',
          quantity: 4,
        },
      ];
      const xml = generateWantedListXml(parts, 'Test');
      expect(xml).toContain('<ItemTypeID>P</ItemTypeID>');
    });
  });

  describe('edge cases', () => {
    it('handles part numbers with special characters', () => {
      const url = generateBricklinkPartUrl('3069b', 1);
      expect(url).toBeDefined();
    });

    it('handles zero quantity in parts list', () => {
      const parts: MissingPart[] = [
        {
          partNum: '3001',
          colorName: 'Red',
          colorHex: '#C91A09',
          quantity: 0,
        },
      ];
      const text = formatMissingPartsText(parts);
      expect(text).toBeDefined();
      expect(text).toContain('3001');
    });

    it('handles very long part numbers', () => {
      const url = generateBricklinkPartUrl('32524b01pb02', 1);
      expect(url).toBeDefined();
      expect(url).toContain('32524b01pb02');
    });

    it('handles unicode characters in list name', () => {
      const parts: MissingPart[] = [];
      const xml = generateWantedListXml(parts, 'Millennium Falcon 飛鷹');
      expect(xml).toContain('Millennium Falcon');
    });
  });

  describe('integration', () => {
    it('combines URL generation and formatting', () => {
      const parts: MissingPart[] = [
        {
          partNum: '3001',
          colorName: 'Red',
          colorHex: '#C91A09',
          quantity: 4,
        },
      ];
      const url = generateBricklinkPartUrl(parts[0].partNum, 5);
      const text = formatMissingPartsText(parts);

      expect(url).toBeDefined();
      expect(text).toBeDefined();
      expect(url).toContain('3001');
      expect(text).toContain('3001');
    });

    it('generates complete wanted list XML with formatted text', () => {
      const parts: MissingPart[] = [
        {
          partNum: '3001',
          colorName: 'Red',
          colorHex: '#C91A09',
          quantity: 4,
        },
        {
          partNum: '3002',
          colorName: 'Blue',
          colorHex: '#1E5AA8',
          quantity: 8,
        },
      ];
      const xml = generateWantedListXml(parts, 'Test Build');
      const text = formatMissingPartsText(parts);

      expect(xml).toContain('Test Build');
      expect(text).toContain('3001');
      expect(text).toContain('3002');
    });

    it('handles realistic build scenarios', () => {
      const millenniumFalconParts: MissingPart[] = [
        {
          partNum: '3001',
          colorName: 'Black',
          colorHex: '#212121',
          quantity: 150,
        },
        {
          partNum: '3002',
          colorName: 'Dark Bluish Gray',
          colorHex: '#595D60',
          quantity: 120,
        },
        {
          partNum: '3003',
          colorName: 'White',
          colorHex: '#FFFFFF',
          quantity: 100,
        },
      ];

      const xml = generateWantedListXml(millenniumFalconParts, 'Millennium Falcon - Missing Parts');
      const text = formatMissingPartsText(millenniumFalconParts);

      expect(xml).toContain('Millennium Falcon');
      expect(text).toContain('150');
      expect(text).toContain('120');
      expect(text).toContain('100');
    });
  });
});
