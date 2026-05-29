/**
 * Sinks for the Phase 4 live-scan telemetry recorder (liveScanTelemetry.ts).
 *
 * The recorder is framework-free; persistence lives here so the native/network
 * deps (expo-file-system, the API base URL) stay out of the pure core and the
 * recorder remains unit-testable. ContinuousScanScreen wires these up only when
 * the debug flag is on.
 *
 * Two sinks, both best-effort (a failure is logged, never thrown — the recorder
 * already wraps each sink in try/catch):
 *   - fileSink:    write the doc to <documentDirectory>livescan_telemetry/<session>.json
 *                  so a sweep is retrievable off the device (Xcode container,
 *                  Files app via expo-sharing, etc.) even with no network.
 *   - backendSink: POST the doc to /api/local-inventory/telemetry/livescan so
 *                  device sweeps are collected centrally and diffable against the
 *                  inner-loop harness baseline.
 */
import * as FileSystem from 'expo-file-system/legacy';
import { apiClient } from '@/services/api';
import type { LiveScanTelemetryDoc, TelemetrySink } from './liveScanTelemetry';

/** Subdir (under the app document dir) where session JSON files are written. */
export const TELEMETRY_DIR_NAME = 'livescan_telemetry';

/** Absolute file URI a given session's telemetry is/was written to. */
export function telemetryFileUri(sessionId: string): string {
  const safe = sessionId.replace(/[^A-Za-z0-9._-]+/g, '_');
  return `${FileSystem.documentDirectory}${TELEMETRY_DIR_NAME}/${safe}.json`;
}

/**
 * File sink — writes the doc as pretty JSON under the document dir. Returns the
 * written URI on success (also logged) so the caller can surface it / share it.
 */
export const fileSink: TelemetrySink = async (doc: LiveScanTelemetryDoc): Promise<void> => {
  const dir = `${FileSystem.documentDirectory}${TELEMETRY_DIR_NAME}`;
  const info = await FileSystem.getInfoAsync(dir);
  if (!info.exists) {
    await FileSystem.makeDirectoryAsync(dir, { intermediates: true });
  }
  const uri = telemetryFileUri(doc.meta.session_id);
  await FileSystem.writeAsStringAsync(uri, JSON.stringify(doc, null, 2));
  console.log('[telemetry] wrote session to', uri);
};

/**
 * Backend sink — POSTs the doc to the telemetry ingest route. Uses a bare fetch
 * against the same base URL the API client resolves (Metro host on a tethered
 * device), with a short timeout so a slow/absent server never stalls "Done".
 */
export const backendSink: TelemetrySink = async (doc: LiveScanTelemetryDoc): Promise<void> => {
  const url = `${apiClient.getBaseUrl()}/api/local-inventory/telemetry/livescan`;
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), 8000);
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(doc),
      signal: controller.signal,
    });
    if (!res.ok) {
      console.warn('[telemetry] backend POST failed:', res.status, await res.text().catch(() => ''));
    } else {
      console.log('[telemetry] posted session to backend:', doc.meta.session_id);
    }
  } finally {
    clearTimeout(t);
  }
};

/**
 * The default sink set: always persist locally; also POST when `toBackend`.
 * ContinuousScanScreen chooses `toBackend` (e.g. only when not local-only).
 */
export function defaultSinks(toBackend: boolean): TelemetrySink[] {
  return toBackend ? [fileSink, backendSink] : [fileSink];
}
