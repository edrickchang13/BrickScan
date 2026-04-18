/**
 * ContinuousScanScreen — live-feed brick scanner.
 *
 * Phase 1 shipped: throttled per-frame scan + EMA fusion by part_num.
 * Phase 2 shipped: per-bbox IoU tracking + live bbox overlay.
 * Phase 3 (this rev): Kalman smoothing (tracker-internal), session persistence
 *         via AsyncStorage, dev-mode performance telemetry, keep-existing
 *         ascending-track-count throttling stays.
 * Phase 4 (this rev): multi-brick confirmation modal w/ editable qtys, drawer
 *         sort modes, polished AR-style bbox labels (BboxOverlay).
 *
 * True on-device YOLO inference is deferred to Phase 5 — the existing 167MB
 * ONNX needs quantization and a native onnxruntime bridge before it's
 * mobile-deployable. See docs/CONTINUOUS_SCAN_PHASE5.md for plan.
 */
import React, {
  useCallback, useEffect, useMemo, useRef, useState,
} from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet, StatusBar, Alert,
  ActivityIndicator, Platform, AppState,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { useIsFocused } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import * as ImageManipulator from 'expo-image-manipulator';
import * as FileSystem from 'expo-file-system/legacy';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { ScanStackParamList } from '@/types';
import { apiClient } from '@/services/api';
import { C, R, S, shadow } from '@/constants/theme';
import {
  DetectedBricksDrawer, ContinuousBrickTrack, DrawerSortMode,
} from '@/components/DetectedBricksDrawer';
import { BboxOverlay, BboxTrackVisual } from '@/components/BboxOverlay';
import { ConfirmBricksModal, ConfirmedBrickEntry } from '@/components/ConfirmBricksModal';
import {
  updateTracks, BrickTrack, DEFAULT_TRACKER_OPTS,
} from '@/utils/bboxTracker';
import { saveSession, loadSession, clearSession } from '@/utils/scanSession';
import { useInventoryStore } from '@/store/inventoryStore';

// Optional expo-haptics — no-ops gracefully if module isn't installed.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let Haptics: any = null;
try {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  Haptics = require('expo-haptics');
} catch {}

type Props = NativeStackScreenProps<ScanStackParamList, 'ContinuousScanScreen'>;

const FRAME_INTERVAL_MS = 1200;
const MAX_IMAGE_DIM = 720;
const JPEG_QUALITY = 0.55;
const AUTO_PAUSE_AFTER_NO_NEW_LOCKS_MS = 45_000;
const SAVE_DEBOUNCE_MS = 2_000;

export const ContinuousScanScreen: React.FC<Props> = ({ navigation }) => {
  const [permission, requestPermission] = useCameraPermissions();
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [isStreaming, setIsStreaming] = useState(true);
  const [isInflight, setIsInflight] = useState(false);
  const [tracks, setTracks] = useState<BrickTrack[]>([]);
  const [drawerExpanded, setDrawerExpanded] = useState(false);
  const [sortMode, setSortMode] = useState<DrawerSortMode>('recent');
  const [errorBanner, setErrorBanner] = useState<string | null>(null);
  const [sourceAR, setSourceAR] = useState<number | undefined>(undefined);
  const [showConfirm, setShowConfirm] = useState(false);
  const [lastLatencyMs, setLastLatencyMs] = useState<number | null>(null);
  const [sessionRestored, setSessionRestored] = useState(false);

  const cameraRef = useRef<CameraView>(null);
  const isFocused = useIsFocused();
  const scanLoopRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const saveDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastLockAtRef = useRef<number>(Date.now());
  const isInflightRef = useRef(false);
  const streamingRef = useRef(isStreaming);
  streamingRef.current = isStreaming;

  const addItem = useInventoryStore((s) => s.addItem);

  // ── Permission ────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!permission) requestPermission();
  }, [permission]);

  // ── Session restore on mount ──────────────────────────────────────────────
  useEffect(() => {
    (async () => {
      const restored = await loadSession();
      if (restored && restored.length > 0) {
        setTracks(restored);
        setSessionRestored(true);
        setErrorBanner(
          `Resumed previous session — ${restored.length} brick${restored.length === 1 ? '' : 's'} restored.`,
        );
        setTimeout(() => setErrorBanner(null), 4000);
      }
    })();
  }, []);

  // ── Persist tracks on change (debounced) ──────────────────────────────────
  useEffect(() => {
    if (saveDebounceRef.current) clearTimeout(saveDebounceRef.current);
    saveDebounceRef.current = setTimeout(() => {
      void saveSession(tracks);
    }, SAVE_DEBOUNCE_MS);
    return () => {
      if (saveDebounceRef.current) clearTimeout(saveDebounceRef.current);
    };
  }, [tracks]);

  // Flush a save when the app backgrounds
  useEffect(() => {
    const sub = AppState.addEventListener('change', (state) => {
      if (state === 'background' || state === 'inactive') {
        void saveSession(tracks);
      }
    });
    return () => sub.remove();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tracks]);

  // ── Main scan loop ────────────────────────────────────────────────────────
  useEffect(() => {
    if (!isFocused || !isCameraReady || !permission?.granted) return;
    scanLoopRef.current = setInterval(() => {
      if (!streamingRef.current) return;
      if (isInflightRef.current) return;
      void processOneFrame();
    }, FRAME_INTERVAL_MS);
    return () => {
      if (scanLoopRef.current) clearInterval(scanLoopRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isFocused, isCameraReady, permission?.granted]);

  // ── Auto-pause after N seconds of no new locks ────────────────────────────
  useEffect(() => {
    if (!isStreaming) return;
    const id = setInterval(() => {
      if (Date.now() - lastLockAtRef.current > AUTO_PAUSE_AFTER_NO_NEW_LOCKS_MS) {
        setIsStreaming(false);
        setErrorBanner('Paused — no new bricks detected for a while. Tap resume.');
      }
    }, 5_000);
    return () => clearInterval(id);
  }, [isStreaming]);

  const processOneFrame = useCallback(async () => {
    if (!cameraRef.current) return;
    isInflightRef.current = true;
    setIsInflight(true);
    const tStart = Date.now();
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
      if (manipulated.width && manipulated.height) {
        setSourceAR(manipulated.width / manipulated.height);
      }
      const base64 = await FileSystem.readAsStringAsync(manipulated.uri, {
        encoding: FileSystem.EncodingType.Base64,
      });

      const res = await apiClient.scanMultiPiece(base64);
      setErrorBanner(prev =>
        prev && prev.startsWith('Resumed previous session') ? prev : null,
      );

      const now = Date.now();
      setLastLatencyMs(now - tStart);
      setTracks(prev => {
        const { tracks: next, newlyLocked } = updateTracks(
          prev, res.pieces, now, DEFAULT_TRACKER_OPTS,
        );
        if (newlyLocked.length > 0) {
          lastLockAtRef.current = now;
          if (Haptics?.notificationAsync) {
            void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
          }
        }
        return next;
      });
    } catch (err: any) {
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
    void clearSession();
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
    setShowConfirm(true);
  }, [lockedTracks]);

  const handleConfirm = useCallback(async (entries: ConfirmedBrickEntry[]) => {
    // Commit to inventory via the Zustand store — same path single-scan uses.
    // addItem signature: (partNum, colorId, quantity, colorName?, colorHex?).
    for (const entry of entries) {
      try {
        await addItem(entry.partNum, '', entry.quantity, entry.colorName ?? '', '');
      } catch {
        // Individual add failures are non-fatal; continue with the rest.
      }
    }
    await clearSession();
    setShowConfirm(false);
    setTracks([]);
    navigation.goBack();
  }, [addItem, navigation]);

  // ── Derived props ─────────────────────────────────────────────────────────
  const bboxVisuals = useMemo<BboxTrackVisual[]>(() => tracks.map(t => ({
    id: t.id,
    bbox: t.bbox,
    partNum: t.partNum,
    confidence: t.fusedConfidence,
    state:
      t.lockedAt !== null ? 'locked'
      : (Date.now() - t.lastSeenAt > FRAME_INTERVAL_MS * 2) ? 'decaying'
      : 'pending',
  })), [tracks]);

  const drawerTracks = useMemo<ContinuousBrickTrack[]>(() => tracks.map(t => ({
    id: t.id,
    partNum: t.partNum,
    partName: t.partName,
    colorName: t.colorName,
    colorHex: t.colorHex,
    consecutiveAgreements: t.consecutiveAgreements,
    fusedConfidence: t.fusedConfidence,
    firstSeenAt: t.firstSeenAt,
    lockedAt: t.lockedAt,
  })), [tracks]);

  // ── Render ────────────────────────────────────────────────────────────────
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

      <BboxOverlay tracks={bboxVisuals} sourceAspectRatio={sourceAR} />

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
          {__DEV__ && lastLatencyMs !== null && (
            <Text style={styles.perfText}>{lastLatencyMs}ms</Text>
          )}
        </View>

        <DetectedBricksDrawer
          tracks={drawerTracks}
          expanded={drawerExpanded}
          sortMode={sortMode}
          onSortModeChange={setSortMode}
          onToggle={() => setDrawerExpanded(x => !x)}
          onRemove={handleRemove}
          onClear={handleClear}
        />
      </View>

      {errorBanner && (
        <View style={[
          styles.errorBanner,
          sessionRestored && { backgroundColor: 'rgba(22, 163, 74, 0.9)' },
        ]}>
          <Ionicons name="information-circle" size={14} color={C.white} />
          <Text style={styles.errorText}>{errorBanner}</Text>
        </View>
      )}

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
          style={[
            styles.doneBtn,
            lockedTracks.length === 0 && styles.doneBtnDisabled,
          ]}
          onPress={handleDone}
          activeOpacity={0.85}
        >
          <Ionicons name="checkmark-circle" size={20} color={C.white} />
          <Text style={styles.doneBtnText}>
            Done · {lockedTracks.length} brick{lockedTracks.length === 1 ? '' : 's'}
          </Text>
        </TouchableOpacity>
      </View>

      <ConfirmBricksModal
        visible={showConfirm}
        tracks={drawerTracks.filter(t => t.lockedAt !== null)}
        onCancel={() => {
          setShowConfirm(false);
          setIsStreaming(true);
        }}
        onConfirm={handleConfirm}
      />
    </View>
  );
};

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
  perfText: {
    color: 'rgba(255,255,255,0.6)',
    fontSize: 10,
    marginLeft: 6,
    fontFamily: Platform.OS === 'ios' ? 'Courier New' : 'monospace',
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
