import { Share, Linking } from 'react-native';
import { Config } from '../constants/config';

export interface MissingPart {
  partNum: string;
  colorHex: string;
  colorName: string;
  quantity: number;
}

export function generateBricklinkWantedListUrl(xmlContent: string): string {
  const encoded = encodeURIComponent(xmlContent);
  return `${Config.BRICKLINK_BASE_URL}/v3/api/upload/wanted-list?data=${encoded}`;
}

export function generateBricklinkPartUrl(
  partNum: string,
  colorId?: number
): string {
  const baseUrl = `${Config.BRICKLINK_BASE_URL}/v2/catalog/catalogitem.page`;
  const params = new URLSearchParams({
    P: partNum,
    ...(colorId && { colorID: colorId.toString() }),
  });
  return `${baseUrl}?${params}`;
}

export async function shareBricklinkXml(
  xml: string,
  setName: string
): Promise<void> {
  try {
    const message = `BrickLink Wanted List for ${setName}\n\nInstructions:\n1. Copy the XML below\n2. Go to BrickLink.com\n3. My BrickLink > Wanted Lists > Upload\n4. Paste the XML`;

    await Share.share({
      message: `${message}\n\nXML:\n${xml}`,
      title: `BrickLink Wanted List - ${setName}`,
    });
  } catch (error) {
    console.error('Failed to share BrickLink XML:', error);
    throw error;
  }
}

export function formatMissingPartsText(missingParts: MissingPart[]): string {
  if (missingParts.length === 0) {
    return 'No missing parts!';
  }

  const lines = missingParts.map(
    (part) => `Part #${part.partNum} (${part.colorName}) - Qty: ${part.quantity}`
  );

  return `Missing Parts:\n${lines.join('\n')}`;
}

export function generateWantedListXml(
  missingParts: MissingPart[],
  listName: string
): string {
  const items = missingParts
    .map(
      (part) => `
    <Item>
      <ItemID>${part.partNum}</ItemID>
      <ItemTypeID>P</ItemTypeID>
      <ColorID>${getColorIdFromHex(part.colorHex)}</ColorID>
      <MaxPrice>0.0000</MaxPrice>
      <MinQty>${part.quantity}</MinQty>
      <Condition>Any</Condition>
      <Remarks>Wanted for build</Remarks>
    </Item>`
    )
    .join('');

  return `<?xml version="1.0" encoding="UTF-8"?>
<WantedList>
  <WantedListName>${listName}</WantedListName>
  <Items>${items}
  </Items>
</WantedList>`;
}

function getColorIdFromHex(hex: string): number {
  const colorMap: { [key: string]: number } = {
    '#FFFFFF': 1,
    '#FFFF00': 3,
    '#FF0000': 5,
    '#0000FF': 9,
    '#000000': 11,
    '#008000': 78,
    '#FF8000': 84,
    '#A0A5A9': 194,
    '#FFC40C': 226,
  };
  return colorMap[hex.toUpperCase()] || 0;
}

export function openBricklinkPart(partNum: string, colorId?: number): void {
  const url = generateBricklinkPartUrl(partNum, colorId);
  Linking.openURL(url);
}
