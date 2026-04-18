/**
 * Persists continuous-scan session state to AsyncStorage so a returning
 * user picks up where they left off (e.g. after backgrounding the app to
 * check a reference and coming back).
 *
 * Stored shape is intentionally lossy — we drop the Kalman state and the
 * per-track history buffer. Only the IDs + part_num + lock status +
 * confidence survive across reloads. That's what the user needs to
 * re-confirm; anything derived can be re-computed on resume.
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import type { BrickTrack } from './bboxTracker';

const STORAGE_KEY = 'brickscan.continuous_scan.session.v1';
const MAX_AGE_MS = 2 * 60 * 60 * 1000;  // 2 hours — older sessions are stale

interface SerializedTrack {
  id: string;
  partNum: string;
  partName: string;
  colorName?: string;
  colorHex?: string;
  fusedConfidence: number;
  consecutiveAgreements: number;
  firstSeenAt: number;
  lastSeenAt: number;
  lockedAt: number | null;
  bbox: [number, number, number, number];
}

interface SerializedSession {
  version: 1;
  savedAt: number;
  tracks: SerializedTrack[];
}

export async function saveSession(tracks: BrickTrack[]): Promise<void> {
  if (tracks.length === 0) {
    await AsyncStorage.removeItem(STORAGE_KEY);
    return;
  }
  const payload: SerializedSession = {
    version: 1,
    savedAt: Date.now(),
    tracks: tracks.map(t => ({
      id: t.id,
      partNum: t.partNum,
      partName: t.partName,
      colorName: t.colorName,
      colorHex: t.colorHex,
      fusedConfidence: t.fusedConfidence,
      consecutiveAgreements: t.consecutiveAgreements,
      firstSeenAt: t.firstSeenAt,
      lastSeenAt: t.lastSeenAt,
      lockedAt: t.lockedAt,
      bbox: t.bbox,
    })),
  };
  try {
    await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch {
    /* storage errors are non-fatal */
  }
}

export async function loadSession(): Promise<BrickTrack[] | null> {
  try {
    const raw = await AsyncStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const payload: SerializedSession = JSON.parse(raw);
    if (payload.version !== 1) return null;
    if (Date.now() - payload.savedAt > MAX_AGE_MS) {
      await AsyncStorage.removeItem(STORAGE_KEY);
      return null;
    }
    // Rebuild tracks — re-init Kalman from last-known bbox; drop history.
    const { initBboxKalman } = await import('./kalmanBbox');
    return payload.tracks.map(t => ({
      id: t.id,
      partNum: t.partNum,
      partName: t.partName,
      colorName: t.colorName,
      colorHex: t.colorHex,
      bbox: t.bbox,
      kalman: initBboxKalman(t.bbox, t.lastSeenAt),
      history: [],
      fusedConfidence: t.fusedConfidence,
      consecutiveAgreements: t.consecutiveAgreements,
      firstSeenAt: t.firstSeenAt,
      lastSeenAt: t.lastSeenAt,
      lockedAt: t.lockedAt,
      version: 1,
    }));
  } catch {
    return null;
  }
}

export async function clearSession(): Promise<void> {
  try {
    await AsyncStorage.removeItem(STORAGE_KEY);
  } catch {
    /* ignore */
  }
}
