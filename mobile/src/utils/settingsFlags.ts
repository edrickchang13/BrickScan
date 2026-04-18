/**
 * Shared AsyncStorage keys for user-facing feature flags.
 *
 * Kept separate from the screen that writes them so other surfaces (e.g.
 * ContinuousScanScreen) can read without duplicating string literals and
 * without creating a circular import back into SettingsScreen.
 */
import AsyncStorage from '@react-native-async-storage/async-storage';

export const SETTINGS_KEYS = {
  scanMode: 'brickscan_default_scan_mode',
  localOnly: 'brickscan_local_only',
  onDeviceDetect: 'brickscan_on_device_detect',
  highPerfMode: 'brickscan_high_perf_mode',
} as const;

export async function readBool(key: string, fallback = false): Promise<boolean> {
  const v = await AsyncStorage.getItem(key);
  if (v == null) return fallback;
  return v === 'true';
}

export async function writeBool(key: string, value: boolean): Promise<void> {
  await AsyncStorage.setItem(key, value ? 'true' : 'false');
}
