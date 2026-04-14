import * as ImageManipulator from 'expo-image-manipulator';
import * as FileSystem from 'expo-file-system';
import { Config } from '../constants/config';

export async function compressImage(
  uri: string,
  quality: number = 0.75
): Promise<string> {
  try {
    const manipulated = await ImageManipulator.manipulateAsync(uri, [], {
      compress: quality,
      format: ImageManipulator.SaveFormat.JPEG,
    });

    const base64 = await FileSystem.readAsStringAsync(manipulated.uri, {
      encoding: FileSystem.EncodingType.Base64,
    });

    const sizeInBytes = Math.round((base64.length * 3) / 4);
    if (sizeInBytes > 500000) {
      return compressImage(uri, quality * 0.9);
    }

    return `data:image/jpeg;base64,${base64}`;
  } catch (error) {
    console.error('Failed to compress image:', error);
    throw error;
  }
}

export async function uriToBase64(uri: string): Promise<string> {
  try {
    const base64 = await FileSystem.readAsStringAsync(uri, {
      encoding: FileSystem.EncodingType.Base64,
    });
    return `data:image/jpeg;base64,${base64}`;
  } catch (error) {
    console.error('Failed to convert URI to base64:', error);
    throw error;
  }
}

export function getPartImageUrl(partNum: string): string {
  const paddedPartNum = partNum.padStart(5, '0');
  return `${Config.REBRICKABLE_IMAGE_CDN}/${paddedPartNum}.jpg`;
}

export function getSetImageUrl(setNum: string): string {
  return `https://cdn.rebrickable.com/media/sets/${setNum}.jpg`;
}

export function colorHexToRGBA(hex: string, alpha: number = 1): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);

  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export async function generateThumbnail(
  uri: string,
  size: number = 150
): Promise<string> {
  try {
    const manipulated = await ImageManipulator.manipulateAsync(
      uri,
      [{ resize: { width: size, height: size } }],
      {
        compress: 0.7,
        format: ImageManipulator.SaveFormat.JPEG,
      }
    );

    const base64 = await FileSystem.readAsStringAsync(manipulated.uri, {
      encoding: FileSystem.EncodingType.Base64,
    });

    return `data:image/jpeg;base64,${base64}`;
  } catch (error) {
    console.error('Failed to generate thumbnail:', error);
    throw error;
  }
}
