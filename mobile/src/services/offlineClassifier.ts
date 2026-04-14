/**
 * Offline brick classification module for React Native.
 *
 * Provides native CoreML inference on iOS via a Native Module bridge.
 * Falls back to HTTP API if the native module is unavailable.
 *
 * Usage:
 *   import { classifyOffline, isOfflineAvailable } from './services/offlineClassifier';
 *
 *   if (isOfflineAvailable()) {
 *     const results = await classifyOffline(imageUri);
 *   } else {
 *     // Fall back to API
 *   }
 */

import { NativeModules, Platform } from 'react-native';

// Type definitions

export interface ClassifyResult {
  partNum: string;
  colorId: number;
  colorName: string;
  confidence: number;
}

export interface OfflineClassifierInterface {
  classify(imageUri: string, topK?: number): Promise<ClassifyResult[]>;
  isAvailable(): Promise<boolean>;
  getModelInfo(): Promise<{ numParts: number; numColors: number } | null>;
}

// Native module bridge
const BrickClassifierModule = Platform.OS === 'ios'
  ? NativeModules.BrickClassifier
  : null;

// Cache for availability check
let cachedOfflineAvailable: boolean | null = null;

/**
 * Check if offline classification is available.
 * Runs once and caches the result.
 */
export async function isOfflineAvailable(): Promise<boolean> {
  if (cachedOfflineAvailable !== null) {
    return cachedOfflineAvailable;
  }

  if (!BrickClassifierModule) {
    cachedOfflineAvailable = false;
    return false;
  }

  try {
    const available = await BrickClassifierModule.isAvailable();
    cachedOfflineAvailable = Boolean(available);
    return cachedOfflineAvailable;
  } catch (error) {
    console.warn('Error checking offline availability:', error);
    cachedOfflineAvailable = false;
    return false;
  }
}

/**
 * Classify a brick image using offline CoreML model.
 *
 * @param imageUri - File URI to the image (file:// or asset path)
 * @param topK - Number of top predictions to return (default: 5)
 * @returns Array of ClassifyResult sorted by confidence descending
 * @throws Error if offline classification is unavailable or inference fails
 */
export async function classifyOffline(
  imageUri: string,
  topK: number = 5
): Promise<ClassifyResult[]> {
  const available = await isOfflineAvailable();

  if (!available) {
    throw new Error(
      'Offline classification is not available. Ensure BrickClassifier native module is linked and the CoreML model is embedded in the app bundle.'
    );
  }

  if (!BrickClassifierModule) {
    throw new Error('BrickClassifier native module not found');
  }

  try {
    const results = await BrickClassifierModule.classify(imageUri, topK);

    // Validate and normalize results
    if (!Array.isArray(results)) {
      throw new Error('Native module returned non-array result');
    }

    return results.map((result: any) => ({
      partNum: String(result.partNum),
      colorId: Number(result.colorId),
      colorName: String(result.colorName),
      confidence: Number(result.confidence),
    }));
  } catch (error) {
    throw new Error(
      `Offline classification failed: ${error instanceof Error ? error.message : String(error)}`
    );
  }
}

/**
 * Classify an image using the best available method.
 * Tries offline first, falls back to HTTP API if unavailable.
 *
 * @param imageUri - File URI to the image
 * @param apiEndpoint - Fallback HTTP endpoint (default: http://localhost:8000/classify)
 * @returns Array of ClassifyResult
 */
export async function classifyHybrid(
  imageUri: string,
  apiEndpoint: string = 'http://localhost:8000/classify'
): Promise<ClassifyResult[]> {
  const offline = await isOfflineAvailable();

  if (offline) {
    try {
      return await classifyOffline(imageUri);
    } catch (error) {
      console.warn('Offline classification failed, falling back to API:', error);
    }
  }

  // Fall back to HTTP API
  return await classifyViaAPI(imageUri, apiEndpoint);
}

/**
 * Classify image via HTTP API (fallback method).
 *
 * @param imageUri - File URI to the image
 * @param apiEndpoint - HTTP endpoint
 * @returns Array of ClassifyResult
 */
async function classifyViaAPI(
  imageUri: string,
  apiEndpoint: string
): Promise<ClassifyResult[]> {
  // Convert file URI to blob if needed
  const response = await fetch(imageUri);
  const blob = await response.blob();

  const formData = new FormData();
  formData.append('file', blob);

  const result = await fetch(apiEndpoint, {
    method: 'POST',
    body: formData,
  });

  if (!result.ok) {
    throw new Error(`API request failed: ${result.statusText}`);
  }

  const json = await result.json();

  if (!json.success || !Array.isArray(json.predictions)) {
    throw new Error('Invalid API response');
  }

  return json.predictions.map((pred: any) => ({
    partNum: String(pred.part_num),
    colorId: Number(pred.color_id),
    colorName: getColorName(Number(pred.color_id)),
    confidence: Number(pred.confidence),
  }));
}

/**
 * Get human-readable color name from color ID.
 */
function getColorName(colorId: number): string {
  const colorMap: { [key: number]: string } = {
    0: 'Black',
    1: 'White',
    2: 'Red',
    3: 'Green',
    4: 'Blue',
    5: 'Yellow',
    6: 'Brown',
    7: 'Gray',
    8: 'Orange',
    9: 'Pink',
  };
  return colorMap[colorId] ?? 'Unknown';
}

/**
 * Get model information (number of parts and colors).
 * Returns null if offline classification is unavailable.
 */
export async function getOfflineModelInfo(): Promise<{
  numParts: number;
  numColors: number;
} | null> {
  const available = await isOfflineAvailable();

  if (!available || !BrickClassifierModule) {
    return null;
  }

  try {
    return await BrickClassifierModule.getModelInfo();
  } catch (error) {
    console.warn('Error fetching model info:', error);
    return null;
  }
}

/**
 * Pre-warm the model by doing a dummy classification.
 * Helps reduce latency on first real inference.
 */
export async function prewarmOfflineModel(): Promise<boolean> {
  const available = await isOfflineAvailable();

  if (!available) {
    return false;
  }

  try {
    // Create a blank image as warmup
    // In production, you'd use a real test image
    console.log('Prewarming offline classifier...');

    if (BrickClassifierModule?.prewarm) {
      await BrickClassifierModule.prewarm();
    }

    return true;
  } catch (error) {
    console.warn('Prewarm failed:', error);
    return false;
  }
}

// (ClassifyResult and OfflineClassifierInterface are exported inline above)

// Default exports for common patterns
export default {
  classifyOffline,
  classifyHybrid,
  isOfflineAvailable,
  getOfflineModelInfo,
  prewarmOfflineModel,
};
