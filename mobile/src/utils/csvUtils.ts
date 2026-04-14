import * as FileSystem from 'expo-file-system/legacy';
import { Share } from 'react-native';

export interface InventoryItem {
  id: string;
  partNum: string;
  partName: string;
  colorHex: string;
  colorName: string;
  quantity: number;
}

export function inventoryToCsv(items: InventoryItem[]): string {
  const headers = ['part_num', 'part_name', 'color_name', 'color_hex', 'quantity'];
  const headerLine = headers.join(',');

  const lines = items.map((item) => {
    const partNum = escapeCsvValue(item.partNum);
    const partName = escapeCsvValue(item.partName);
    const colorName = escapeCsvValue(item.colorName);
    const colorHex = escapeCsvValue(item.colorHex);
    const quantity = item.quantity.toString();

    return [partNum, partName, colorName, colorHex, quantity].join(',');
  });

  return [headerLine, ...lines].join('\n');
}

export function parseCsvToInventory(
  csvContent: string
): Array<{ part_num: string; color_name: string; quantity: number }> {
  const lines = csvContent.trim().split('\n');

  if (lines.length < 2) {
    throw new Error('CSV must contain header and at least one data row');
  }

  const headers = lines[0].toLowerCase().split(',').map((h) => h.trim());
  const partNumIndex = headers.indexOf('part_num');
  const colorNameIndex = headers.indexOf('color_name');
  const quantityIndex = headers.indexOf('quantity');

  if (
    partNumIndex === -1 ||
    colorNameIndex === -1 ||
    quantityIndex === -1
  ) {
    throw new Error(
      'CSV must contain part_num, color_name, and quantity columns'
    );
  }

  const items = [];
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;

    const values = parseCsvLine(line);

    const partNum = values[partNumIndex]?.trim();
    const colorName = values[colorNameIndex]?.trim();
    const quantityStr = values[quantityIndex]?.trim();

    if (!partNum || !colorName || !quantityStr) {
      continue;
    }

    const quantity = parseInt(quantityStr, 10);
    if (isNaN(quantity) || quantity < 0) {
      continue;
    }

    items.push({
      part_num: partNum,
      color_name: colorName,
      quantity,
    });
  }

  return items;
}

export async function downloadAndShareCsv(
  csvContent: string,
  filename: string
): Promise<void> {
  try {
    const timestamp = new Date().toISOString().split('T')[0];
    const fullFilename = `${filename}-${timestamp}.csv`;
    const documentDirectory = FileSystem.documentDirectory;
    if (!documentDirectory) {
      throw new Error('Document directory is unavailable on this device');
    }
    const filePath = `${documentDirectory}${fullFilename}`;

    await FileSystem.writeAsStringAsync(filePath, csvContent, {
      encoding: FileSystem.EncodingType.UTF8,
    });

    await Share.share({
      url: filePath,
      title: 'Export Inventory',
      message: `BrickScan Inventory Export\n${filename}`,
    });
  } catch (error) {
    console.error('Failed to download and share CSV:', error);
    throw error;
  }
}

function escapeCsvValue(value: string): string {
  if (value.includes(',') || value.includes('"') || value.includes('\n')) {
    return `"${value.replace(/"/g, '""')}"`;
  }
  return value;
}

function parseCsvLine(line: string): string[] {
  const result = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const char = line[i];

    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === ',' && !inQuotes) {
      result.push(current);
      current = '';
    } else {
      current += char;
    }
  }

  result.push(current);
  return result;
}
