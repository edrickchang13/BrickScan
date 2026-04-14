import AsyncStorage from '@react-native-async-storage/async-storage';

export interface SetSummary {
  setNum: string;
  setName: string;
  year: number;
  theme: string;
  partCount: number;
  imageUrl?: string;
}

export interface BuildCheckResult {
  setNum: string;
  setName: string;
  totalParts: number;
  totalQuantity: number;
  haveParts: number;
  haveQuantity: number;
  percentComplete: number;
  timestamp: number;
}

export interface ScanPrediction {
  id: string;
  partNum: string;
  partName: string;
  colorId?: string;
  colorName?: string;
  colorHex?: string;
  confidence: number;
  imageUrl?: string;
  timestamp: number;
  // Scan metadata preserved for history replay
  scanMode?: 'photo' | 'video' | 'multi';
  source?: string;
  framesAnalyzed?: number;
  agreementScore?: number;
  allPredictions?: Array<{
    partNum: string;
    partName: string;
    colorId?: string;
    colorName?: string;
    colorHex?: string;
    confidence: number;
    imageUrl?: string;
    source?: string;
  }>;
}

const WISHLIST_KEY = 'brickscan_wishlist';
const BUILD_CHECK_KEY = 'brickscan_build_check_';
const RECENT_SCANS_KEY = 'brickscan_recent_scans';
const MAX_RECENT_SCANS = 50;

export async function saveWishlist(sets: SetSummary[]): Promise<void> {
  try {
    await AsyncStorage.setItem(WISHLIST_KEY, JSON.stringify(sets));
  } catch (error) {
    console.error('Failed to save wishlist:', error);
    throw error;
  }
}

export async function loadWishlist(): Promise<SetSummary[]> {
  try {
    const data = await AsyncStorage.getItem(WISHLIST_KEY);
    return data ? JSON.parse(data) : [];
  } catch (error) {
    console.error('Failed to load wishlist:', error);
    return [];
  }
}

export async function addToWishlist(set: SetSummary): Promise<void> {
  try {
    const wishlist = await loadWishlist();
    const exists = wishlist.some((s) => s.setNum === set.setNum);

    if (!exists) {
      wishlist.unshift(set);
      await saveWishlist(wishlist);
    }
  } catch (error) {
    console.error('Failed to add to wishlist:', error);
    throw error;
  }
}

export async function removeFromWishlist(setNum: string): Promise<void> {
  try {
    const wishlist = await loadWishlist();
    const filtered = wishlist.filter((s) => s.setNum !== setNum);
    await saveWishlist(filtered);
  } catch (error) {
    console.error('Failed to remove from wishlist:', error);
    throw error;
  }
}

export async function isInWishlist(setNum: string): Promise<boolean> {
  try {
    const wishlist = await loadWishlist();
    return wishlist.some((s) => s.setNum === setNum);
  } catch (error) {
    console.error('Failed to check wishlist:', error);
    return false;
  }
}

export async function saveBuildCheckResult(
  setNum: string,
  result: BuildCheckResult
): Promise<void> {
  try {
    await AsyncStorage.setItem(
      `${BUILD_CHECK_KEY}${setNum}`,
      JSON.stringify(result)
    );
  } catch (error) {
    console.error('Failed to save build check result:', error);
    throw error;
  }
}

export async function loadBuildCheckResult(
  setNum: string
): Promise<BuildCheckResult | null> {
  try {
    const data = await AsyncStorage.getItem(`${BUILD_CHECK_KEY}${setNum}`);
    return data ? JSON.parse(data) : null;
  } catch (error) {
    console.error('Failed to load build check result:', error);
    return null;
  }
}

export async function saveRecentScan(
  prediction: ScanPrediction
): Promise<void> {
  try {
    const scans = await loadRecentScans();
    const updated = [{ ...prediction, timestamp: Date.now() }, ...scans].slice(
      0,
      MAX_RECENT_SCANS
    );
    await AsyncStorage.setItem(RECENT_SCANS_KEY, JSON.stringify(updated));
  } catch (error) {
    console.error('Failed to save recent scan:', error);
    throw error;
  }
}

export async function loadRecentScans(): Promise<ScanPrediction[]> {
  try {
    const data = await AsyncStorage.getItem(RECENT_SCANS_KEY);
    const scans: ScanPrediction[] = data ? JSON.parse(data) : [];
    return scans.sort((a, b) => b.timestamp - a.timestamp);
  } catch (error) {
    console.error('Failed to load recent scans:', error);
    return [];
  }
}

export async function clearRecentScans(): Promise<void> {
  try {
    await AsyncStorage.removeItem(RECENT_SCANS_KEY);
  } catch (error) {
    console.error('Failed to clear recent scans:', error);
    throw error;
  }
}

export async function clearAllCache(): Promise<void> {
  try {
    const keys = await AsyncStorage.getAllKeys();
    const brickscanKeys = keys.filter((key) => key.startsWith('brickscan_'));
    await AsyncStorage.multiRemove(brickscanKeys);
  } catch (error) {
    console.error('Failed to clear cache:', error);
    throw error;
  }
}
