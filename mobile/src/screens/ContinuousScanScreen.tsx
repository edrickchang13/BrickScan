/**
 * ContinuousScanScreen — Phase 1 of the live-feed scan experience.
 *
 * Flow:
 *   1. Camera opens full-screen.
 *   2. Every ~1200ms, grab a frame (lightweight JPEG) and POST to /api/scan/pile.
 *   3. Results accumulate into `tracks`. Each tracked part_num keeps a history
 *      of sightings and a fused confidence (EMA).
 *   4. A brick "locks" when it appears as a top match in `LOCK_AGREEMENT_COUNT`
 *      consecutive scans AND fused confidence ≥ LOCK_CONFIDENCE.
 *   5. Locked + pending bricks appear in the top-right DetectedBricksDrawer.
 *   6. User taps "Done" to navigate to MultiResultScreen with the locked
 *      inventory pre-populated, or "Pause" to freeze the stream.
 *
 * This is the MVP — no on-device YOLO yet, no per-bbox tracking. Each frame
 * is a fresh pile scan; we fuse by part_num across frames. Works great for
 * the common "bricks on a table, sweep camera around" use case.
 */
import React, {
  useCallback, useEffect, useMemo, useRef, useState,
} from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet, StatusBar, Alert,
  ActivityIndicator, Animated, Platform,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { useIsFocused } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import * as ImageManipulator from 'expo-image-manipulator';
import * as FileSystem from 'expo-file-system/legacy';

// Optional: expo-haptics is nice-to-have. Falls back to a no-op if not installed.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let Haptics: any = null;
try {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  Haptics = require('expo-haptics');
} catch {
  /* haptics unavailable — silent fallback */
}
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { ScanStackParamList } from '@/types';
import { apiClient, PileResult } from '@/services/api';
import { C, R, S, shadow } from '@/constants/theme';
import {
  DetectedBricksDrawer,
  ContinuousBrickTrack,
} from '@/components/DetectedBricksDrawer';

type Props = NativeStackScreenProps<ScanStackParamList, 'ContinuousScanScreen'>;

// Tuning constants — exposed at the top so we can move them to a config file
// later if we want per-device or A/B tuning.
const FRAME_INTERVAL_MS = 1200;            // how often we try to grab a frame
const LOCK_AGREEMENT_COUNT = 3;            // consecutive frames with same top-1
const LOCK_CONFIDENCE = 0.85;              // fused (EMA) confidence threshold
const EMA_ALPHA = 0.4;                      // smoothing — closer to 1 = less smoothing
const TRACK_TIMEOUT_MS = 15_000;           // forget a track after N ms of no sightings
const MAX_IMAGE_DIM = 720;                 // resize frames for speed
const JPEG_QUALITY = 0.55;                 // 0-1
const AUTO_PAUSE_AFTER_NO_NEW_LOCKS_MS = 45_000;  // stop if nothing new for 45s

export const ContinuousScanScreen: React.FC<Props> = ({ navigation }) => {
  const [permission, requestPermission] = useCameraPermissions();
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [isStreaming, setIsStreaming] = useState(true);
  const [isInflight, setIsInflight] = useState(false);
  const [tracks, setTracks] = useState<ContinuousBrickTrack[]>([]);
  const [drawerExpanded, setDrawerExpanded] = useState(false);
  const [lastScanAt, setLastScanAt] = useState<number | null>(null);
  const [errorBanner, setErrorBanner] = useState<string | null>(null);

  const cameraRef = useRef<CameraView>(null);
  const isFocused = useIsFocused();
  const scanLoopRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastLockAtRef = useRef<number>(Date.now());
  const isInflightRef = useRef(false);
  const streamingRef = useRef(isStreaming);
  streamingRef.current = isStreaming;

  // Permission gate
  useEffect(() => {
    if (!permission) requestPermission();
  }, [permission]);

  // Loop: grab-frame → scan → fuse → repeat, while this screen is focused
  useEffect(() => {
    if (!isFocused || !isCameraReady || !permission?.granted) return;
    scanLoopRef.current = setInterval(() => {
      if (!streamingRef.current) return;
      if (isInflightRef.current) return;   // skip if the previous scan hasn't finished
      void processOneFrame();
    }, FRAME_INTERVAL_MS);
    return () => {
      if (scanLoopRef.current) clearInterval(scanLoopRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isFocused, isCameraReady, permission?.granted]);

  // Auto-pause when nothing new has locked for a while
  useEffect(() => {
    if (!isStreaming) return;
    const interval = setInterval(() => {
      if (Date.now() - lastLockAtRef.current > AUTO_PAUSE_AFTER_NO_NEW_LOCKS_MS) {
        setIsStreaming(false);
        setErrorBanner('Paused — no new bricks detected for a while. Tap resume.');
      }
    }, 5000);
    return () => clearInterval(interval);
  }, [isStreaming]);

  // Forget stale tracks that haven't been seen recently (unless already locked)
  useEffect(() => {
    const interval = setInterval(() => {
      const now = Date.now();
      setTracks(prev =>
        prev.filter(t => t.lockedAt !== null || now - t.firstSeenAt < TRACK_TIMEOUT_MS)
      );
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const processOneFrame = useCallback(async () => {
    if (!cameraRef.current) return;
    isInflightRef.current = true;
    setIsInflight(true);
    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: JPEG_QUALITY,
        skipProcessing: true,
        shutterSound: false,
      });
      if (!photo?.uri) return;

      const manipulated = await ImageManipulator.manipulateAsync(
        photo.uri,
        [{ resize: { width: MAX_IMAGE_DIM } }],
        { compress: JPEG_QUALITY, format: ImageManipulator.SaveFormat.JPEG },
      );
      const base64 = await FileSystem.readAsStringAsync(manipulated.uri, {
        encoding: FileSystem.EncodingType.Base64,
      });

      const results = await apiClient.scanPile(base64);
      const now = Date.now();
      setLastScanAt(now);
      setErrorBanner(null);

      if (!results || results.length === 0) {
        // No hits this frame — decay confidence on pending tracks slightly
        setTracks(prev => prev.map(t =>
          t.lockedAt !== null
            ? t
            : { ...t, consecutiveAgreements: 0, fusedConfidence: t.fusedConfidence * 0.9 }
        ));
        return;
      }

      // Fuse: for each scan result, update matching track OR create a new one
      setTracks(prev => {
        const byKey = new Map(prev.map(t => [trackKey(t.partNum, t.colorName), t]));
        const seenKeys = new Set<string>();

        for (const r of results) {
          const key = trackKey(r.partNum, r.colorName);
          seenKeys.add(key);
          const existing = byKey.get(key);
          if (existing) {
            byKey.set(key, fuseTrack(existing, r, now));
          } else {
            byKey.set(key, newTrack(r, now));
          }
        }

        // Decay pending tracks that WEREN'T seen this frame
        for (const [key, t] of byKey) {
          if (!seenKeys.has(key) && t.lockedAt === null) {
            byKey.set(key, {
              ...t,
              consecutiveAgreements: 0,
              fusedConfidence: t.fusedConfidence * 0.9,
            });
          }
        }

        const nextList = Array.from(byKey.values());
        // Trigger haptic + update "last lock" timestamp if anything new locked this cycle
        const newlyLocked = nextList.filter(t =>
          t.lockedAt !== null && !prev.find(p => p.id === t.id && p.lockedAt !== null)
        );
        if (newlyLocked.length > 0) {
          lastLockAtRef.current = Date.now();
          if (Haptics?.notificationAsync) {
            void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
          }
        }
        return nextList;
      });
    } catch (err: any) {
      // Don't spam the banner on transient network errors — just log.
      if (err?.message && !/timeout|network|aborted/i.test(err.message)) {
        setErrorBanner(`Scan error: ${err.message}`);
      }
    } finally {
      isInflightRef.current = false;
      setIsInflight(false);
    }
  }, []);

  const handleRemove = useCallback((id: string) => {
    setTracks(prev => prev.filter(t => t.id !== id));
  }, []);

  const handleClear = useCallback(() => {
    setTracks([]);
  }, []);

  const lockedTracks = useMemo(
    () => tracks.filter(t => t.lockedAt !== null),
    [tracks],
  );

  const handleDone = useCallback(() => {
    if (lockedTracks.length === 0) {
      Alert.alert(
        'No bricks locked yet',
        'Keep scanning until bricks are confirmed (green check), or cancel to go back.',
      );
      return;
    }
    setIsStreaming(false);
    navigation.navigate('MultiResultScreen', {
      pieces: lockedTracks.map((t, i) => {
        const primary = {
          partNum: t.partNum,
          partName: t.partName,
          colorId: '',
          colorName: t.colorName ?? '',
          colorHex: t.colorHex ?? '',
          confidence: t.fusedConfidence,
          source: 'continuous_scan',
        };
        return {
          pieceIndex: i,
          predictions: [primary],
          primaryPrediction: primary,
        };
      }),
    });
  }, [lockedTracks, navigation]);

  if (!permission) {
    return (
      <View style={styles.center}>
        <ActivityIndicator color={C.red} />
      </View>
    );
  }
  if (!permission.granted) {
    return (
      <View style={styles.center}>
        <Ionicons name="camera-outline" size={48} color={C.textMuted} />
        <Text style={styles.permText}>Camera access is required for continuous scan.</Text>
        <TouchableOpacity style={styles.permBtn} onPress={requestPermission}>
          <Text style={styles.permBtnText}>Grant access</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.root}>
      <StatusBar barStyle="light-content" />
      <CameraView
        ref={cameraRef}
        style={StyleSheet.absoluteFillObject}
        facing="back"
        onCameraReady={() => setIsCameraReady(true)}
        animateShutter={false}
      />

      {/* Top bar: back, live indicator, detection drawer */}
      <View style={styles.topBar}>
        <TouchableOpacity
          onPress={() => navigation.goBack()}
          style={styles.iconBtn}
          hitSlop={10}
        >
          <Ionicons name="chevron-back" size={24} color={C.white} />
        </TouchableOpacity>

        <View style={styles.liveBadge}>
          <View style={[
            styles.liveDot,
            { backgroundColor: isStreaming ? C.red : C.textMuted },
          ]} />
          <Text style={styles.liveText}>
            {isStreaming
              ? (isInflight ? 'SCANNING…' : 'LIVE')
              : 'PAUSED'}
          </Text>
        </View>

        <DetectedBricksDrawer
          tracks={tracks}
          expanded={drawerExpanded}
          onToggle={() => setDrawerExpanded(x => !x)}
          onRemove={handleRemove}
          onClear={handleClear}
        />
      </View>

      {/* Error banner */}
      {errorBanner && (
        <View style={styles.errorBanner}>
          <Ionicons name="information-circle" size={14} color={C.white} />
          <Text style={styles.errorText}>{errorBanner}</Text>
        </View>
      )}

      {/* Bottom control bar */}
      <View style={styles.bottomBar}>
        <TouchableOpacity
          style={styles.controlBtn}
          onPress={() => setIsStreaming(x => !x)}
          hitSlop={10}
        >
          <Ionicons
            name={isStreaming ? 'pause' : 'play'}
            size={22}
            color={C.white}
          />
          <Text style={styles.controlText}>{isStreaming ? 'Pause' : 'Resume'}</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.doneBtn, lockedTracks.length === 0 && styles.doneBtnDisabled]}
          onPress={handleDone}
          activeOpacity={0.85}
        >
          <Ionicons name="checkmark-circle" size={20} color={C.white} />
          <Text style={styles.doneBtnText}>
            Done · {lockedTracks.length} brick{lockedTracks.length === 1 ? '' : 's'}
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

function trackKey(partNum: string, colorName?: string): string {
  return `${partNum}|${colorName ?? ''}`;
}

function newTrack(r: PileResult, now: number): ContinuousBrickTrack {
  return {
    id: `${r.partNum}_${now}`,
    partNum: r.partNum,
    partName: r.partName,
    colorName: r.colorName,
    thumbnailUrl: r.cropImageBase64
      ? `data:image/jpeg;base64,${r.cropImageBase64}`
      : undefined,
    sightings: 1,
    consecutiveAgreements: 1,
    fusedConfidence: r.confidence,
    firstSeenAt: now,
    lockedAt: null,
  };
}

function fuseTrack(
  t: ContinuousBrickTrack,
  r: PileResult,
  now: number,
): ContinuousBrickTrack {
  const sightings = t.sightings + 1;
  const consecutiveAgreements = t.consecutiveAgreements + 1;
  const fusedConfidence = EMA_ALPHA * r.confidence + (1 - EMA_ALPHA) * t.fusedConfidence;
  const isLocked = t.lockedAt !== null ||
    (consecutiveAgreements >= LOCK_AGREEMENT_COUNT && fusedConfidence >= LOCK_CONFIDENCE);
  return {
    ...t,
    sightings,
    consecutiveAgreements,
    fusedConfidence,
    thumbnailUrl: t.thumbnailUrl ?? (r.cropImageBase64
      ? `data:image/jpeg;base64,${r.cropImageBase64}`
      : undefined),
    lockedAt: isLocked && t.lockedAt === null ? now : t.lockedAt,
  };
}

// ─── Styles ──────────────────────────────────────────────────────────────────

const TOP_INSET = Platform.OS === 'ios' ? 52 : 32;

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: C.black },
  center: {
    flex: 1, alignItems: 'center', justifyContent: 'center',
    backgroundColor: C.bg, padding: S.xl,
  },
  permText: {
    color: C.textSub,
    textAlign: 'center',
    marginVertical: S.md,
  },
  permBtn: {
    backgroundColor: C.red,
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: R.sm,
  },
  permBtnText: { color: C.white, fontWeight: '700' },

  topBar: {
    position: 'absolute',
    top: TOP_INSET,
    left: S.md,
    right: S.md,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  iconBtn: {
    width: 40, height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(0,0,0,0.5)',
    alignItems: 'center', justifyContent: 'center',
  },
  liveBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.5)',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: R.full,
    alignSelf: 'center',
  },
  liveDot: {
    width: 8, height: 8,
    borderRadius: 4,
    marginRight: 6,
  },
  liveText: {
    color: C.white,
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 1,
  },

  errorBanner: {
    position: 'absolute',
    top: TOP_INSET + 56,
    left: S.md,
    right: S.md,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(227, 0, 11, 0.9)',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: R.sm,
  },
  errorText: { color: C.white, fontSize: 12, marginLeft: 6, flex: 1 },

  bottomBar: {
    position: 'absolute',
    bottom: Platform.OS === 'ios' ? 40 : 24,
    left: S.md,
    right: S.md,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  controlBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.55)',
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: R.full,
  },
  controlText: {
    color: C.white,
    fontWeight: '600',
    fontSize: 13,
    marginLeft: 6,
  },
  doneBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: C.red,
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: R.full,
    ...shadow(2),
  },
  doneBtnDisabled: { backgroundColor: 'rgba(227, 0, 11, 0.4)' },
  doneBtnText: {
    color: C.white,
    fontWeight: '700',
    fontSize: 14,
    marginLeft: 6,
  },
});
