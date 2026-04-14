import { NativeModules } from 'react-native';

const DEFAULT_API_BASE_URL = 'http://localhost:8000';

type RuntimeEnv = {
  process?: {
    env?: Record<string, string | undefined>;
  };
};

export function getApiBaseUrl(): string {
  // PRIORITY 1 (dev): use Metro's current scriptURL host so the API auto-follows
  // whatever IP Metro is serving on (USB link-local, WiFi, etc.).
  if (__DEV__) {
    const scriptURL = NativeModules.SourceCode?.scriptURL as string | undefined;
    if (scriptURL) {
      try {
        const host = new URL(scriptURL).hostname;
        const url = `http://${host}:8000`;
        console.log('[Config] API_BASE_URL (from Metro scriptURL):', url);
        return url;
      } catch (e) {
        console.warn('[Config] Metro scriptURL parse failed:', scriptURL, e);
      }
    } else {
      console.warn('[Config] NativeModules.SourceCode.scriptURL is empty — falling back to EXPO_PUBLIC_API_URL');
    }
  }

  // PRIORITY 2: explicit env override (set in .env.local).
  const envUrl = (globalThis as RuntimeEnv).process?.env?.EXPO_PUBLIC_API_URL;
  if (envUrl) {
    if (__DEV__) console.log('[Config] API_BASE_URL (from EXPO_PUBLIC_API_URL):', envUrl);
    return envUrl;
  }

  // PRIORITY 3: fallback — only reachable on simulator.
  if (__DEV__) console.warn('[Config] Falling back to localhost — scan will fail on physical device');
  return DEFAULT_API_BASE_URL;
}

export const Config = {
  // API Configuration
  API_BASE_URL: getApiBaseUrl(),
  REBRICKABLE_IMAGE_CDN: 'https://cdn.rebrickable.com/media/parts/photos/0',
  BRICKLINK_BASE_URL: 'https://www.bricklink.com',

  // Confidence Thresholds
  HIGH_CONFIDENCE: 0.85,
  MEDIUM_CONFIDENCE: 0.60,
  LOW_CONFIDENCE: 0.40,

  // Pagination
  PAGE_SIZE: 20,

  // Cache TTL (milliseconds)
  SET_CACHE_TTL: 24 * 60 * 60 * 1000,
  PART_CACHE_TTL: 60 * 60 * 1000,
  SEARCH_CACHE_TTL: 5 * 60 * 1000,
  BUILD_CHECK_CACHE_TTL: 10 * 60 * 1000,

  // Image Processing
  MAX_IMAGE_SIZE: 500000,
  IMAGE_COMPRESSION_QUALITY: 0.75,
  THUMBNAIL_SIZE: 150,

  // Export
  MAX_EXPORT_ITEMS: 10000,

  // Timeouts
  REQUEST_TIMEOUT: 30000,
  SYNC_INTERVAL: 5 * 60 * 1000,

  // Feature Flags
  ENABLE_OFFLINE_MODE: true,
  ENABLE_WISHLIST: true,
  ENABLE_CLOUD_SYNC: false,

  // Debounce delays (ms)
  SEARCH_DEBOUNCE_DELAY: 300,
  SYNC_DEBOUNCE_DELAY: 1000,
};
