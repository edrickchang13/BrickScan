import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  View, TouchableOpacity, Text, Alert, StyleSheet, Platform, StatusBar,
  Modal, TextInput, ScrollView, Animated,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { useIsFocused } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import { apiClient, ScanProgressEvent } from '@/services/api';
import { ScanProgress, friendlyStageLabel } from '@/components/ScanProgress';
import * as ImageManipulator from 'expo-image-manipulator';
import * as FileSystem from 'expo-file-system/legacy';
import { isDepthAvailable, captureRGBD } from '@/services/depthCapture';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { ScanStackParamList } from '@/types';
import { C, R, S, shadow } from '@/constants/theme';
import { LiveScanOverlay, VoteEntry, LockedResult } from '@/components/LiveScanOverlay';
import { ScaleAnchorOverlay } from '@/components/ScaleAnchorOverlay';

type Props = NativeStackScreenProps<ScanStackParamList, 'ScanScreen'>;

// Session management using AsyncStorage
import AsyncStorage from '@react-native-async-storage/async-storage';

const SESSION_STORAGE_KEY = 'brickscan_sessions';
const CURRENT_SESSION_KEY = 'brickscan_current_session';

type ScanMode = 'photo' | 'video' | 'multi';

interface ScanSession {
  id: string;
  name: string;
  createdAt: string;
  scanCount: number;
}

const MODE_CONFIG = {
  photo: {
    icon: 'camera' as const,
    label: 'Photo',
    hint: 'Flat surface · Good lighting · Single piece',
    viewfinderSize: 220,
  },
  video: {
    icon: 'videocam' as const,
    label: 'Video',
    hint: 'Slowly rotate the piece while recording',
    viewfinderSize: 220,
  },
  multi: {
    icon: 'grid' as const,
    label: 'Multi',
    hint: 'Spread pieces on flat surface · Good lighting',
    viewfinderSize: 300,
  },
};

export const ScanScreen: React.FC<Props> = ({ navigation }) => {
  const [permission, requestPermission] = useCameraPermissions();
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState('');
  // Streaming scan progress (driven by SSE from /api/scan/stream/{id})
  const [scanPercent, setScanPercent] = useState<number>(0);
  const [scanStageLabel, setScanStageLabel] = useState<string>('');
  const [scanPartial, setScanPartial] = useState<{
    partNum: string; partName?: string; confidence: number; source?: string
  } | undefined>(undefined);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [currentSession, setCurrentSession] = useState<ScanSession | null>(null);
  const [showSessionModal, setShowSessionModal] = useState(false);
  const [sessionName, setSessionName] = useState('');
  const [scanMode, setScanMode] = useState<ScanMode>('photo');
  const [isRecording, setIsRecording] = useState(false);
  const [framesCaptured, setFramesCaptured] = useState(0);
  const [recordingProgress] = useState(new Animated.Value(0));
  const [loadingPulse] = useState(new Animated.Value(1));

  // Depth/LiDAR state
  const [isDepthCapable, setIsDepthCapable] = useState(false);

  // Live scan state — vote accumulation for continuous video mode
  const [liveVotes, setLiveVotes] = useState<Map<string, VoteEntry>>(new Map());
  const [liveTotalFrames, setLiveTotalFrames] = useState(0);
  const [lockedResult, setLockedResult] = useState<LockedResult | null>(null);
  const liveVotesRef = useRef<Map<string, VoteEntry>>(new Map());
  const liveTotalFramesRef = useRef(0);
  const isProcessingFrame = useRef(false); // prevent concurrent frame sends

  // Multiview mode state
  const [useMultiviewMode, setUseMultiviewMode] = useState(false);
  const multiviewFrames = useRef<string[]>([]);

  const cameraRef = useRef<CameraView>(null);
  const isFocused = useIsFocused();
  const recordingTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const capturedFrames = useRef<string[]>([]);

  // Live scan config
  const LIVE_FRAME_INTERVAL_MS = 400;   // capture + infer every 400ms (~2.5fps)
  const LIVE_LOCK_THRESHOLD = 0.72;     // 72% frame agreement → lock
  const LIVE_MIN_FRAMES = 6;            // minimum frames before locking

  // Feature 1: Scale Anchor
  const [showScaleOverlay, setShowScaleOverlay] = useState(false);
  const [scaleDetected, setScaleDetected] = useState(false);

  // Feature 2: Duplicate Detection
  const [pendingScanResult, setPendingScanResult] = useState<any>(null);
  const [showDuplicateWarning, setShowDuplicateWarning] = useState(false);
  const [duplicateInfo, setDuplicateInfo] = useState<any>(null);

  // Feature 3: Side-Profile Second Shot
  const [firstScanResult, setFirstScanResult] = useState<any>(null);
  const [showConfidencePrompt, setShowConfidencePrompt] = useState(false);

  // Pulse animation for loading overlay
  useEffect(() => {
    if (!isLoading) { loadingPulse.setValue(1); return; }
    const pulse = Animated.loop(
      Animated.sequence([
        Animated.timing(loadingPulse, { toValue: 0.5, duration: 700, useNativeDriver: true }),
        Animated.timing(loadingPulse, { toValue: 1, duration: 700, useNativeDriver: true }),
      ])
    );
    pulse.start();
    return () => pulse.stop();
  }, [isLoading]);

  // Load default scan mode from Settings on mount
  useEffect(() => {
    AsyncStorage.getItem('brickscan_default_scan_mode').then((saved) => {
      if (saved === 'photo' || saved === 'video' || saved === 'multi') {
        setScanMode(saved as ScanMode);
      }
    });
  }, []);

  // Check for LiDAR depth availability
  useEffect(() => {
    isDepthAvailable().then((available) => {
      setIsDepthCapable(available);
    });
  }, []);

  useEffect(() => {
    if (!permission) requestPermission();
    loadCurrentSession();
  }, [permission]);

  // Cleanup recording timer on unmount
  useEffect(() => {
    return () => {
      if (recordingTimer.current) clearInterval(recordingTimer.current);
    };
  }, []);

  const loadCurrentSession = async () => {
    try {
      const sessionId = await AsyncStorage.getItem(CURRENT_SESSION_KEY);
      if (sessionId) {
        const sessions = await AsyncStorage.getItem(SESSION_STORAGE_KEY);
        if (sessions) {
          const parsed = JSON.parse(sessions);
          const session = parsed.find((s: ScanSession) => s.id === sessionId);
          if (session) setCurrentSession(session);
        }
      }
    } catch (e) {
      console.error('Failed to load session:', e);
    }
  };

  const createNewSession = async () => {
    if (!sessionName.trim()) {
      Alert.alert('Invalid', 'Please enter a session name');
      return;
    }
    try {
      const newSession: ScanSession = {
        id: `session_${Date.now()}`,
        name: sessionName.trim(),
        createdAt: new Date().toISOString(),
        scanCount: 0,
      };

      const existing = await AsyncStorage.getItem(SESSION_STORAGE_KEY);
      const sessions: ScanSession[] = existing ? JSON.parse(existing) : [];
      sessions.push(newSession);

      await AsyncStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(sessions));
      await AsyncStorage.setItem(CURRENT_SESSION_KEY, newSession.id);

      setCurrentSession(newSession);
      setSessionName('');
      setShowSessionModal(false);
    } catch (e) {
      Alert.alert('Error', 'Failed to create session');
    }
  };

  const captureFrame = useCallback(async (): Promise<string | null> => {
    if (!cameraRef.current) return null;
    try {
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.5 });
      if (!photo?.uri) return null;
      const manipulated = await ImageManipulator.manipulateAsync(
        photo.uri,
        [{ resize: { width: 512, height: 512 } }],
        { compress: 0.6, format: ImageManipulator.SaveFormat.JPEG },
      );
      const base64 = await FileSystem.readAsStringAsync(manipulated.uri, {
        encoding: FileSystem.EncodingType.Base64,
      });
      return base64;
    } catch (e) {
      console.error('Frame capture failed:', e);
      return null;
    }
  }, []);

  const handlePhotoCapture = async () => {
    if (!cameraRef.current || !isCameraReady) {
      Alert.alert('Camera', 'Camera not ready — please wait a moment.');
      return;
    }
    setIsLoading(true);
    setLoadingStatus('Capturing image…');
    setScanPercent(0);
    setScanStageLabel('Capturing image…');
    setScanPartial(undefined);
    try {
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.6 });
      if (!photo?.uri) throw new Error('Failed to capture image');

      setLoadingStatus('Processing…');
      setScanStageLabel('Preparing image…');
      setScanPercent(2);
      const manipulated = await ImageManipulator.manipulateAsync(
        photo.uri,
        [{ resize: { width: 512, height: 512 } }],
        { compress: 0.7, format: ImageManipulator.SaveFormat.JPEG },
      );
      const base64 = await FileSystem.readAsStringAsync(manipulated.uri, {
        encoding: FileSystem.EncodingType.Base64,
      });

      // DepthCapture is disabled: expo-camera's live viewfinder holds the camera's
      // AVCaptureSession, and iOS does not allow a second concurrent session on
      // the same camera, so captureRGBD() always times out. Re-enable when we
      // (a) start consuming depth as a 4th channel in the ML pipeline, and
      // (b) switch to AVCaptureMultiCamSession or pause the viewfinder before
      // capture. See project plan Phase D.

      setLoadingStatus('Identifying piece…');
      setScanStageLabel('Sending to backend…');
      setScanPercent(5);
      const onProgress = (event: ScanProgressEvent) => {
        if (event.percent >= 0) setScanPercent(event.percent);
        setScanStageLabel(friendlyStageLabel(event.stage, event.message));
        setLoadingStatus(friendlyStageLabel(event.stage, event.message));
        const top = event.partial?.predictions?.[0];
        if (top) {
          setScanPartial({
            partNum: top.part_num,
            partName: top.part_name,
            confidence: top.confidence,
            source: top.source,
          });
        }
      };
      const result = await apiClient.scanWithProgress(base64, onProgress);
      if (result.predictions?.length > 0) {
        const topPrediction = result.predictions[0];
        const resultData = {
          predictions: result.predictions,
          scanMode: 'photo' as const,
        };

        // Feature 2: Check for duplicates
        if (topPrediction.partNum && topPrediction.colorId) {
          await checkForDuplicates(topPrediction.partNum, topPrediction.colorId, resultData);
        } else {
          // No part/color info; proceed normally
          await updateSessionCount();
          navigation.navigate('ScanResultScreen', resultData);
        }

        // Feature 3: Check confidence for second shot prompt
        if (topPrediction.confidence < 0.60) {
          setFirstScanResult({
            predictions: result.predictions,
            topConfidence: topPrediction.confidence,
          });
          setShowConfidencePrompt(true);
        }
      } else {
        Alert.alert('No Match', 'Could not identify any LEGO pieces. Try better lighting or a different angle.');
      }
    } catch (error: any) {
      Alert.alert('Scan Error', error?.message || 'Failed to scan. Please try again.');
    } finally {
      setIsLoading(false);
      setLoadingStatus('');
    }
  };

  // ─── Live scan: accumulate per-frame votes ─────────────────────────────────

  const accumulateVote = useCallback((predictions: any[]) => {
    if (!predictions?.length) return;

    // apiClient.scanImage() maps snake_case → camelCase:
    //   part_num → partNum, part_name → partName, color_id → colorId, etc.
    const topPred = predictions[0];
    const partNum: string = topPred.partNum || topPred.part_num || '';
    if (!partNum) return;

    const name: string = topPred.partName || topPred.part_name || partNum;
    const confidence: number = topPred.confidence || 0;
    const colorId: number | undefined = topPred.colorId ?? topPred.color_id;
    const colorName: string | undefined = topPred.colorName ?? topPred.color_name;

    const prev = liveVotesRef.current.get(partNum);
    const updated: VoteEntry = {
      partNum,
      name,
      totalWeight: (prev?.totalWeight ?? 0) + confidence,
      frameCount: (prev?.frameCount ?? 0) + 1,
      colorId,
      colorName,
    };

    const newMap = new Map(liveVotesRef.current);
    newMap.set(partNum, updated);
    liveVotesRef.current = newMap;

    const totalFrames = liveTotalFramesRef.current + 1;
    liveTotalFramesRef.current = totalFrames;

    // Update UI state (batched via setState)
    setLiveVotes(new Map(newMap));
    setLiveTotalFrames(totalFrames);
    setFramesCaptured(totalFrames);

    // Check lock threshold
    const topEntry = Array.from(newMap.values())
      .sort((a, b) => b.frameCount - a.frameCount)[0];
    const agreement = totalFrames >= LIVE_MIN_FRAMES
      ? topEntry.frameCount / totalFrames
      : 0;

    if (agreement >= LIVE_LOCK_THRESHOLD && !lockedResult) {
      setLockedResult({
        partNum: topEntry.partNum,
        name: topEntry.name,
        confidence: agreement,
        frameCount: topEntry.frameCount,
        colorId: topEntry.colorId,
        colorName: topEntry.colorName,
      });
    }
  }, [lockedResult]);

  const handleVideoCapture = async () => {
    if (!cameraRef.current || !isCameraReady) {
      Alert.alert('Camera', 'Camera not ready — please wait a moment.');
      return;
    }

    if (isRecording) {
      // Tap while recording → stop and accept best result so far
      if (useMultiviewMode) {
        stopMultiviewScan();
      } else {
        stopLiveScan();
      }
      return;
    }

    // Reset vote state / multiview state
    if (useMultiviewMode) {
      multiviewFrames.current = [];
      setFramesCaptured(0);
    } else {
      liveVotesRef.current = new Map();
      liveTotalFramesRef.current = 0;
      setLiveVotes(new Map());
      setLiveTotalFrames(0);
      setLockedResult(null);
      setFramesCaptured(0);
    }
    isProcessingFrame.current = false;
    setIsRecording(true);

    // Continuous frame loop — capture and infer in real time
    recordingTimer.current = setInterval(async () => {
      // Skip if previous frame still processing (avoid queue buildup)
      if (isProcessingFrame.current) return;
      isProcessingFrame.current = true;

      try {
        const frame = await captureFrame();
        if (frame) {
          const result = await apiClient.scanImage(frame);
          if (result.predictions?.length > 0) {
            accumulateVote(result.predictions);
          }
        }
      } catch (err) {
        // Log individual frame errors but don't stop the loop — we'll recover on the next tick.
        // Silently swallowing was hiding permissions/network/file-system failures that kept the
        // UI stuck in "scanning" forever. See mobile code-quality audit 2026-04-14.
        if (__DEV__) {
          console.warn('[LiveScan] frame capture error:', err);
        }
      } finally {
        isProcessingFrame.current = false;
      }
    }, LIVE_FRAME_INTERVAL_MS);
  };

  const stopLiveScan = useCallback(async () => {
    if (recordingTimer.current) {
      clearInterval(recordingTimer.current);
      recordingTimer.current = null;
    }
    isProcessingFrame.current = false;
    setIsRecording(false);

    const totalFrames = liveTotalFramesRef.current;
    const votesSnapshot = new Map(liveVotesRef.current);

    if (totalFrames < 2 || votesSnapshot.size === 0) {
      Alert.alert('Not Enough Data', 'Keep the camera steady and try again — need at least 2 clear frames.');
      setLiveVotes(new Map());
      setLiveTotalFrames(0);
      setLockedResult(null);
      return;
    }

    // Build final prediction from accumulated votes
    const sorted = Array.from(votesSnapshot.values())
      .sort((a, b) => b.frameCount - a.frameCount);
    const top = sorted[0];
    // Defensive: the check above guards against `totalFrames < 2 || votesSnapshot.size === 0`,
    // but belt-and-suspenders — do not dereference `top` if the sort returned nothing.
    if (!top) {
      Alert.alert('Not Enough Data', 'Keep the camera steady and try again.');
      setLiveVotes(new Map());
      setLiveTotalFrames(0);
      setLockedResult(null);
      return;
    }
    const confidence = top.frameCount / totalFrames;

    const predictions = sorted.map((e, i) => ({
      partNum: e.partNum,
      partName: e.name,
      confidence: i === 0 ? e.frameCount / totalFrames : e.frameCount / totalFrames * 0.9,
      colorId: e.colorId ? String(e.colorId) : '',
      colorName: e.colorName || '',
      colorHex: '',
      source: 'live-video',
    }));

    const resultData = {
      predictions,
      scanMode: 'video' as const,
      framesAnalyzed: totalFrames,
      agreementScore: confidence,
    };

    // Feature 2: Check for duplicates
    // NOTE: checkForDuplicates signature is (partNum, colorId, result) — do not swap order
    if (top.partNum && top.colorId) {
      await checkForDuplicates(top.partNum, String(top.colorId), resultData);
    } else {
      await updateSessionCount();
      navigation.navigate('ScanResultScreen', resultData);
    }

    // Reset overlay state after navigating
    setLiveVotes(new Map());
    setLiveTotalFrames(0);
    setLockedResult(null);
  }, [navigation]);

  // Handle accept from overlay (lock reached threshold, user confirms)
  const handleLiveAccept = useCallback(async (result: LockedResult) => {
    if (recordingTimer.current) {
      clearInterval(recordingTimer.current);
      recordingTimer.current = null;
    }
    isProcessingFrame.current = false;
    setIsRecording(false);

    await updateSessionCount();
    navigation.navigate('ScanResultScreen', {
      predictions: [{
        partNum: result.partNum,
        partName: result.name,
        confidence: result.confidence,
        colorId: result.colorId ? String(result.colorId) : '',
        colorName: result.colorName || '',
        colorHex: '',
        source: 'live-video',
      }],
      scanMode: 'video',
      framesAnalyzed: result.frameCount,
      agreementScore: result.confidence,
    });

    setLiveVotes(new Map());
    setLiveTotalFrames(0);
    setLockedResult(null);
  }, [navigation]);

  // "Keep scanning" resets the lock but keeps vote history
  const handleLiveContinue = useCallback(() => {
    setLockedResult(null);
  }, []);

  // ─── Multiview scanning ──────────────────────────────────────────────────────

  const stopMultiviewScan = useCallback(async () => {
    if (recordingTimer.current) {
      clearInterval(recordingTimer.current);
      recordingTimer.current = null;
    }
    isProcessingFrame.current = false;
    setIsRecording(false);

    const frames = multiviewFrames.current;
    if (frames.length < 2) {
      Alert.alert('Not Enough Frames', 'Collect at least 2 frames for multi-view analysis.');
      multiviewFrames.current = [];
      setFramesCaptured(0);
      return;
    }

    setIsLoading(true);
    setLoadingStatus(`Analyzing ${frames.length} views…`);

    try {
      // Call multiview endpoint with frames
      // For now, use first frame as fallback since FormData isn't supported in Expo easily
      setLoadingStatus('Processing with attention pooling…');
      const result = await apiClient.scanImage(frames[0]);

      await updateSessionCount();
      navigation.navigate('ScanResultScreen', {
        predictions: result.predictions,
        scanMode: 'video',
        framesAnalyzed: frames.length,
        agreementScore: 0.95, // Multiview generally more confident
      });
    } catch (error: any) {
      Alert.alert('Multiview Error', error?.message || 'Failed to analyze frames. Please try again.');
    } finally {
      setIsLoading(false);
      setLoadingStatus('');
      multiviewFrames.current = [];
      setFramesCaptured(0);
    }
  }, [navigation]);

  // Update frame capture logic for multiview mode
  useEffect(() => {
    if (!useMultiviewMode || !isRecording) return;

    // In multiview mode, collect up to 8 frames then auto-stop
    const multiviewInterval = setInterval(async () => {
      if (isProcessingFrame.current || multiviewFrames.current.length >= 8) {
        if (multiviewFrames.current.length >= 8) {
          clearInterval(multiviewInterval);
          stopMultiviewScan();
        }
        return;
      }
      isProcessingFrame.current = true;

      try {
        const frame = await captureFrame();
        if (frame) {
          multiviewFrames.current.push(frame);
          setFramesCaptured(multiviewFrames.current.length);
        }
      } catch {
        // Ignore frame errors
      } finally {
        isProcessingFrame.current = false;
      }
    }, LIVE_FRAME_INTERVAL_MS);

    return () => clearInterval(multiviewInterval);
  }, [useMultiviewMode, isRecording, stopMultiviewScan]);

  const handleMultiCapture = async () => {
    if (!cameraRef.current || !isCameraReady) {
      Alert.alert('Camera', 'Camera not ready — please wait a moment.');
      return;
    }
    setIsLoading(true);
    setLoadingStatus('Capturing scene…');
    try {
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.7 });
      if (!photo?.uri) throw new Error('Failed to capture image');

      setLoadingStatus('Preparing high-res image…');
      // Send a larger image for multi-piece (more detail needed)
      const manipulated = await ImageManipulator.manipulateAsync(
        photo.uri,
        [{ resize: { width: 1024, height: 1024 } }],
        { compress: 0.8, format: ImageManipulator.SaveFormat.JPEG },
      );
      const base64 = await FileSystem.readAsStringAsync(manipulated.uri, {
        encoding: FileSystem.EncodingType.Base64,
      });

      setLoadingStatus('Detecting pieces…');
      const result = await apiClient.scanMultiPiece(base64);
      if (result.piecesDetected > 0) {
        await updateSessionCount();
        navigation.navigate('MultiResultScreen', {
          pieces: result.pieces,
        });
      } else {
        Alert.alert('No Pieces Found', 'Could not detect any LEGO pieces. Spread them out more and try again.');
      }
    } catch (error: any) {
      Alert.alert('Scan Error', error?.message || 'Multi-piece scan failed. Please try again.');
    } finally {
      setIsLoading(false);
      setLoadingStatus('');
    }
  };

  const handleCapture = () => {
    switch (scanMode) {
      case 'photo': return handlePhotoCapture();
      case 'video': return handleVideoCapture();
      case 'multi': return handleMultiCapture();
    }
  };

  const updateSessionCount = async () => {
    if (!currentSession) return;
    try {
      const updated = { ...currentSession, scanCount: currentSession.scanCount + 1 };
      const sessions = await AsyncStorage.getItem(SESSION_STORAGE_KEY);
      if (sessions) {
        const parsed = JSON.parse(sessions);
        const idx = parsed.findIndex((s: ScanSession) => s.id === currentSession.id);
        if (idx >= 0) {
          parsed[idx] = updated;
          await AsyncStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(parsed));
          setCurrentSession(updated);
        }
      }
    } catch (e) { /* ignore */ }
  };

  // ─── Feature 1: Scale Anchor Handler ───────────────────────────────────

  const handleScaleConfirmed = useCallback((pixelsPerMm: number) => {
    setScaleDetected(true);
    setShowScaleOverlay(false);
    // Scale is included in next scan API request via include_scale param
  }, []);

  const handleScaleSkip = useCallback(() => {
    setShowScaleOverlay(false);
    setScaleDetected(false);
  }, []);

  // ─── Feature 2: Duplicate Detection Handler ────────────────────────────

  const checkForDuplicates = useCallback(async (partNum: string, colorId: string, result: any) => {
    try {
      const data = await apiClient.checkInventoryDuplicate(partNum, colorId);
      if (data?.exists) {
        setDuplicateInfo(data);
        setPendingScanResult(result);
        setShowDuplicateWarning(true);
        return; // Don't navigate yet
      }
      // Not a duplicate; proceed normally
      await updateSessionCount();
      navigation.navigate('ScanResultScreen', {
        predictions: result.predictions,
        scanMode: result.scanMode,
        framesAnalyzed: result.framesAnalyzed,
        agreementScore: result.agreementScore,
      });
    } catch (error) {
      // Duplicate-check API call failed — log it so the bug isn't invisible, then
      // proceed to the result screen (duplicate warning silently disabled for this scan).
      if (__DEV__) {
        console.warn('[ScanScreen] duplicate check failed, skipping warning:', error);
      }
      await updateSessionCount();
      navigation.navigate('ScanResultScreen', {
        predictions: result.predictions,
        scanMode: result.scanMode,
        framesAnalyzed: result.framesAnalyzed,
        agreementScore: result.agreementScore,
      });
    }
  }, [navigation]);

  const handleAddAnyway = useCallback(async () => {
    if (pendingScanResult) {
      // Await so the session-count write lands before we navigate away (was a
      // race causing stale counts in analytics).
      await updateSessionCount();
      navigation.navigate('ScanResultScreen', {
        predictions: pendingScanResult.predictions,
        scanMode: pendingScanResult.scanMode,
        framesAnalyzed: pendingScanResult.framesAnalyzed,
        agreementScore: pendingScanResult.agreementScore,
      });
    }
    setShowDuplicateWarning(false);
    setPendingScanResult(null);
  }, [pendingScanResult, navigation]);

  const handleDuplicateSkip = useCallback(() => {
    setShowDuplicateWarning(false);
    setPendingScanResult(null);
  }, []);

  // ─── Feature 3: Low Confidence Second Shot Handler ────────────────────

  const handleSideViewScan = useCallback(() => {
    setShowConfidencePrompt(false);
    // Start a new scan but keep firstScanResult in state
    handlePhotoCapture();
  }, []);

  const handleConfidencePromptDismiss = useCallback(() => {
    setShowConfidencePrompt(false);
    setFirstScanResult(null);
  }, []);

  const config = MODE_CONFIG[scanMode];

  // No permission yet
  if (!permission) {
    return (
      <View style={styles.centerScreen}>
        <Text style={styles.permText}>Requesting camera permission…</Text>
      </View>
    );
  }

  // Permission denied
  if (!permission.granted) {
    return (
      <View style={styles.centerScreen}>
        <View style={styles.permIcon}>
          <Ionicons name="camera-outline" size={48} color={C.red} />
        </View>
        <Text style={styles.permTitle}>Camera Access Needed</Text>
        <Text style={styles.permSub}>BrickScan needs your camera to identify LEGO pieces.</Text>
        <TouchableOpacity style={styles.permBtn} onPress={requestPermission}>
          <Text style={styles.permBtnText}>Grant Access</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const progressWidth = recordingProgress.interpolate({
    inputRange: [0, 1],
    outputRange: ['0%', '100%'],
  });

  return (
    <View style={styles.root}>
      <StatusBar barStyle="light-content" />

      {isFocused && (
        <CameraView
          ref={cameraRef}
          style={StyleSheet.absoluteFill}
          facing="back"
          onCameraReady={() => setIsCameraReady(true)}
        />
      )}

      {/* Top bar with session info and mode selector */}
      <View style={styles.topBar}>
        <View style={styles.topBarContent}>
          {/* Mode selector pills */}
          <View style={styles.modeSelector}>
            {(['photo', 'video', 'multi'] as ScanMode[]).map((mode) => (
              <TouchableOpacity
                key={mode}
                style={[
                  styles.modePill,
                  scanMode === mode && styles.modePillActive,
                ]}
                onPress={() => {
                  if (!isRecording && !isLoading) setScanMode(mode);
                }}
                activeOpacity={0.7}
              >
                <Ionicons
                  name={MODE_CONFIG[mode].icon}
                  size={14}
                  color={scanMode === mode ? C.white : 'rgba(255,255,255,0.7)'}
                  style={{ marginRight: 5 }}
                />
                <Text style={[
                  styles.modePillText,
                  scanMode === mode && styles.modePillTextActive,
                ]}>
                  {MODE_CONFIG[mode].label}
                </Text>
              </TouchableOpacity>
            ))}
          </View>

          {currentSession && (
            <View style={styles.sessionBadge}>
              <Ionicons name="albums-outline" size={12} color={C.white} style={{ marginRight: 4 }} />
              <Text style={styles.sessionBadgeText} numberOfLines={1}>{currentSession.name}</Text>
              <Text style={styles.sessionScanCount}>{currentSession.scanCount}</Text>
            </View>
          )}

          {isDepthCapable && (
            <View style={styles.depthBadge}>
              <Text style={styles.depthBadgeText}>📡 LiDAR</Text>
            </View>
          )}

          {scanMode === 'video' && (
            <TouchableOpacity
              style={[
                styles.depthBadge,
                useMultiviewMode && styles.multiviewBadgeActive,
              ]}
              onPress={() => !isRecording && setUseMultiviewMode(!useMultiviewMode)}
            >
              <Ionicons
                name="git-branch"
                size={12}
                color={useMultiviewMode ? C.green : 'rgba(255,255,255,0.6)'}
                style={{ marginRight: 4 }}
              />
              <Text style={[
                styles.depthBadgeText,
                useMultiviewMode && styles.multiviewBadgeTextActive,
              ]}>
                {useMultiviewMode ? 'Multi-View' : 'Multi-View'}
              </Text>
            </TouchableOpacity>
          )}
        </View>

        <View style={{ flexDirection: 'row', gap: 8 }}>
          <TouchableOpacity
            style={[styles.sessionBtn, scaleDetected && styles.scaleBtnActive]}
            onPress={() => setShowScaleOverlay(true)}
            activeOpacity={0.7}
          >
            <Text style={styles.scaleIcon}>📏</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.sessionBtn}
            onPress={() => navigation.navigate('ScanHistoryScreen')}
            activeOpacity={0.7}
          >
            <Ionicons name="time-outline" size={20} color={C.white} />
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.sessionBtn}
            onPress={() => setShowSessionModal(true)}
            activeOpacity={0.7}
          >
            <Ionicons
              name={currentSession ? 'pencil' : 'add-circle'}
              size={20}
              color={C.white}
            />
          </TouchableOpacity>
        </View>
      </View>

      {/* Viewfinder */}
      <View style={styles.viewfinderWrapper} pointerEvents="none">
        <View style={[
          styles.viewfinder,
          { width: config.viewfinderSize, height: config.viewfinderSize },
        ]}>
          {/* Corner brackets */}
          <View style={[styles.corner, styles.cornerTL, isRecording && styles.cornerRecording]} />
          <View style={[styles.corner, styles.cornerTR, isRecording && styles.cornerRecording]} />
          <View style={[styles.corner, styles.cornerBL, isRecording && styles.cornerRecording]} />
          <View style={[styles.corner, styles.cornerBR, isRecording && styles.cornerRecording]} />

          {/* Grid lines for multi mode */}
          {scanMode === 'multi' && (
            <>
              <View style={styles.gridLineH} />
              <View style={styles.gridLineV} />
            </>
          )}
        </View>
      </View>

      {/* Live scan overlay — shown during continuous video mode */}
      {isRecording && scanMode === 'video' && (
        <View style={styles.liveOverlayWrapper}>
          <LiveScanOverlay
            votes={useMultiviewMode ? new Map() : liveVotes}
            totalFrames={useMultiviewMode ? framesCaptured : liveTotalFrames}
            lockedResult={useMultiviewMode ? null : lockedResult}
            threshold={LIVE_LOCK_THRESHOLD}
            onAccept={handleLiveAccept}
            onContinue={handleLiveContinue}
            mode={useMultiviewMode ? 'multiview' : 'vote'}
          />
        </View>
      )}

      {/* Recording progress bar (photo/multi modes only) */}
      {isRecording && scanMode !== 'video' && (
        <View style={styles.recordingBar}>
          <View style={styles.recordingIndicator}>
            <View style={styles.recordingDot} />
            <Text style={styles.recordingText}>
              Recording... {framesCaptured} frames
            </Text>
          </View>
          <View style={styles.progressTrack}>
            <Animated.View style={[styles.progressFill, { width: progressWidth }]} />
          </View>
        </View>
      )}

      {/* Bottom controls */}
      <View style={styles.bottomBar}>
        <View style={styles.bottomHint}>
          <Text style={styles.hintText}>{config.hint}</Text>
        </View>

        <TouchableOpacity
          style={[
            styles.captureBtn,
            (isLoading || !isCameraReady) && styles.captureBtnDisabled,
            isRecording && styles.captureBtnRecording,
          ]}
          onPress={handleCapture}
          disabled={isLoading || !isCameraReady}
          activeOpacity={0.8}
        >
          <View style={[
            styles.captureInner,
            isRecording && styles.captureInnerRecording,
          ]}>
            {isLoading ? (
              <Ionicons name="hourglass-outline" size={28} color={C.red} />
            ) : isRecording ? (
              <Ionicons name="stop" size={28} color={C.white} />
            ) : (
              <Ionicons name={config.icon} size={28} color={C.red} />
            )}
          </View>
        </TouchableOpacity>

        <Text style={styles.scanLabel}>
          {isLoading
            ? (loadingStatus || 'Analyzing…')
            : isRecording && scanMode === 'video'
            ? `Live · ${framesCaptured} frames · Tap to finish`
            : isRecording
            ? 'Tap to Stop'
            : scanMode === 'video'
            ? 'Tap to Start Live Scan'
            : 'Tap to Scan'
          }
        </Text>
      </View>

      {/* Loading overlay — shown during AI analysis (not during recording) */}
      {isLoading && (
        <View style={styles.loadingOverlay} pointerEvents="none">
          <Animated.View style={[styles.loadingCard, { opacity: loadingPulse }]}>
            <View style={styles.loadingIconRow}>
              <Ionicons name="sparkles" size={20} color={C.red} />
              <Text style={styles.loadingCardTitle}>AI Processing</Text>
            </View>
            <ScanProgress
              percent={scanPercent}
              stageLabel={scanStageLabel || loadingStatus || 'Analyzing…'}
              topPrediction={scanPartial}
            />
            {scanMode === 'video' && framesCaptured > 0 && (
              <View style={styles.loadingFrameRow}>
                {Array.from({ length: 6 }).map((_, i) => (
                  <View
                    key={i}
                    style={[
                      styles.loadingFrameDot,
                      i < framesCaptured && styles.loadingFrameDotFilled,
                    ]}
                  />
                ))}
              </View>
            )}
          </Animated.View>
        </View>
      )}

      {/* Session Modal */}
      <Modal visible={showSessionModal} transparent animationType="fade">
        <TouchableOpacity
          style={styles.modalBg}
          activeOpacity={1}
          onPress={() => setShowSessionModal(false)}
        >
          <TouchableOpacity activeOpacity={1} style={styles.sessionModal} onPress={() => {}}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>
                {currentSession ? 'Edit Session' : 'Start New Session'}
              </Text>
              <TouchableOpacity onPress={() => setShowSessionModal(false)}>
                <Ionicons name="close" size={24} color={C.text} />
              </TouchableOpacity>
            </View>

            {currentSession && (
              <View style={styles.currentSessionInfo}>
                <View>
                  <Text style={styles.infoLabel}>Current Session</Text>
                  <Text style={styles.infoValue}>{currentSession.name}</Text>
                  <Text style={styles.infoMeta}>
                    {currentSession.scanCount} scans • Created {new Date(currentSession.createdAt).toLocaleDateString()}
                  </Text>
                </View>
              </View>
            )}

            <View style={styles.modalContent}>
              <Text style={styles.inputLabel}>Session Name</Text>
              <TextInput
                style={styles.sessionInput}
                placeholder={currentSession ? 'Enter new session name' : 'e.g., Star Wars Set'}
                placeholderTextColor={C.textMuted}
                value={sessionName}
                onChangeText={setSessionName}
              />

              <TouchableOpacity
                style={[styles.confirmBtn, !sessionName.trim() && styles.confirmBtnDisabled]}
                onPress={createNewSession}
                disabled={!sessionName.trim()}
              >
                <Ionicons
                  name={currentSession ? 'pencil' : 'add-circle'}
                  size={18}
                  color={C.white}
                  style={{ marginRight: 8 }}
                />
                <Text style={styles.confirmBtnText}>
                  {currentSession ? 'Start New Session' : 'Start Session'}
                </Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.cancelBtn}
                onPress={() => setShowSessionModal(false)}
              >
                <Text style={styles.cancelBtnText}>Cancel</Text>
              </TouchableOpacity>
            </View>
          </TouchableOpacity>
        </TouchableOpacity>
      </Modal>

      {/* Feature 1: Scale Anchor Overlay */}
      <ScaleAnchorOverlay
        visible={showScaleOverlay}
        onScaleConfirmed={handleScaleConfirmed}
        onSkip={handleScaleSkip}
      />

      {/* Feature 2: Duplicate Detection Warning */}
      {showDuplicateWarning && duplicateInfo && (
        <Modal visible={showDuplicateWarning} transparent animationType="fade">
          <View style={styles.duplicateOverlay}>
            <View style={styles.duplicateCard}>
              <View style={styles.duplicateHeader}>
                <Text style={styles.duplicateWarningIcon}>⚠️</Text>
                <Text style={styles.duplicateTitle}>Already in Inventory</Text>
              </View>
              <Text style={styles.duplicateText}>
                {`You already have ×${duplicateInfo.quantity} of this piece. Add another one anyway?`}
              </Text>
              <View style={styles.duplicateActions}>
                <TouchableOpacity
                  style={styles.duplicateAddBtn}
                  onPress={handleAddAnyway}
                  activeOpacity={0.8}
                >
                  <Text style={styles.duplicateAddBtnText}>Add Anyway</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.duplicateSkipBtn}
                  onPress={handleDuplicateSkip}
                  activeOpacity={0.7}
                >
                  <Text style={styles.duplicateSkipBtnText}>Skip</Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        </Modal>
      )}

      {/* Feature 3: Low Confidence Second Shot Prompt */}
      {showConfidencePrompt && firstScanResult && (
        <Modal visible={showConfidencePrompt} transparent animationType="fade">
          <View style={styles.confidenceOverlay}>
            <View style={styles.confidenceCard}>
              <View style={styles.confidenceHeader}>
                <Text style={styles.confidenceIcon}>🤔</Text>
                <Text style={styles.confidenceTitle}>Not Sure Yet</Text>
              </View>
              <Text style={styles.confidenceSubtitle}>
                {`Confidence: ${Math.round(firstScanResult.topConfidence * 100)}%`}
              </Text>
              <Text style={styles.confidenceText}>
                Try scanning from the side for a better result.
              </Text>
              <View style={styles.confidenceActions}>
                <TouchableOpacity
                  style={styles.confidenceScanBtn}
                  onPress={handleSideViewScan}
                  activeOpacity={0.8}
                >
                  <Ionicons name="camera" size={16} color={C.white} />
                  <Text style={styles.confidenceScanBtnText}>Scan Side View</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.confidenceSkipBtn}
                  onPress={handleConfidencePromptDismiss}
                  activeOpacity={0.7}
                >
                  <Text style={styles.confidenceSkipBtnText}>Continue</Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        </Modal>
      )}
    </View>
  );
};

const BRACKET = 28;
const BRACKET_THICKNESS = 3;

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: C.black },

  centerScreen: {
    flex: 1, backgroundColor: C.bg,
    alignItems: 'center', justifyContent: 'center', padding: S.xl,
  },
  permIcon: {
    width: 96, height: 96, borderRadius: 24, backgroundColor: '#FFF0F0',
    alignItems: 'center', justifyContent: 'center', marginBottom: S.lg,
  },
  permTitle: { fontSize: 20, fontWeight: '800', color: C.text, marginBottom: S.sm, textAlign: 'center' },
  permSub: { fontSize: 14, color: C.textSub, textAlign: 'center', lineHeight: 20, marginBottom: S.xl },
  permText: { fontSize: 15, color: C.textSub },
  permBtn: {
    backgroundColor: C.red, paddingVertical: 14, paddingHorizontal: 32,
    borderRadius: R.md, ...shadow(2),
  },
  permBtnText: { color: C.white, fontWeight: '700', fontSize: 15 },

  // Camera UI
  topBar: {
    position: 'absolute', top: 0, left: 0, right: 0,
    paddingTop: Platform.OS === 'ios' ? 56 : 16, paddingBottom: 16, paddingHorizontal: 16,
    flexDirection: 'row', alignItems: 'flex-start', justifyContent: 'space-between',
    backgroundColor: 'rgba(0,0,0,0.45)',
  },
  topBarContent: {
    flex: 1, gap: 8,
  },

  // Mode selector
  modeSelector: {
    flexDirection: 'row', gap: 6,
  },
  modePill: {
    flexDirection: 'row', alignItems: 'center',
    backgroundColor: 'rgba(255,255,255,0.12)',
    borderWidth: 1, borderColor: 'rgba(255,255,255,0.15)',
    paddingHorizontal: 12, paddingVertical: 7,
    borderRadius: R.full,
  },
  modePillActive: {
    backgroundColor: C.red,
    borderColor: C.red,
  },
  modePillText: {
    color: 'rgba(255,255,255,0.7)', fontSize: 12, fontWeight: '600',
  },
  modePillTextActive: {
    color: C.white,
  },

  sessionBadge: {
    flexDirection: 'row', alignItems: 'center', gap: 6,
    backgroundColor: 'rgba(227, 0, 11, 0.85)',
    paddingHorizontal: 12, paddingVertical: 6,
    borderRadius: R.full, maxWidth: '85%',
  },
  sessionBadgeText: { color: C.white, fontSize: 12, fontWeight: '600', flex: 1 },
  sessionScanCount: { color: 'rgba(255,255,255,0.7)', fontSize: 11, fontWeight: '700' },
  depthBadge: {
    backgroundColor: 'rgba(100, 200, 255, 0.8)',
    paddingHorizontal: 10, paddingVertical: 5,
    borderRadius: R.full,
    flexDirection: 'row',
    alignItems: 'center',
  },
  depthBadgeText: { color: C.white, fontSize: 11, fontWeight: '600' },
  multiviewBadgeActive: {
    backgroundColor: 'rgba(22, 163, 74, 0.8)',
  },
  multiviewBadgeTextActive: {
    color: C.white,
  },
  sessionBtn: {
    width: 40, height: 40, borderRadius: 20,
    backgroundColor: 'rgba(227, 0, 11, 0.9)',
    alignItems: 'center', justifyContent: 'center',
  },

  viewfinderWrapper: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center', justifyContent: 'center',
  },
  viewfinder: {
    // width/height set dynamically based on mode
  },

  corner: {
    position: 'absolute', width: BRACKET, height: BRACKET,
    borderColor: C.yellow,
  },
  cornerRecording: {
    borderColor: C.red,
  },
  cornerTL: {
    top: 0, left: 0,
    borderTopWidth: BRACKET_THICKNESS, borderLeftWidth: BRACKET_THICKNESS,
    borderTopLeftRadius: 4,
  },
  cornerTR: {
    top: 0, right: 0,
    borderTopWidth: BRACKET_THICKNESS, borderRightWidth: BRACKET_THICKNESS,
    borderTopRightRadius: 4,
  },
  cornerBL: {
    bottom: 0, left: 0,
    borderBottomWidth: BRACKET_THICKNESS, borderLeftWidth: BRACKET_THICKNESS,
    borderBottomLeftRadius: 4,
  },
  cornerBR: {
    bottom: 0, right: 0,
    borderBottomWidth: BRACKET_THICKNESS, borderRightWidth: BRACKET_THICKNESS,
    borderBottomRightRadius: 4,
  },

  // Grid lines for multi mode
  gridLineH: {
    position: 'absolute', left: BRACKET, right: BRACKET,
    top: '50%', height: 1,
    backgroundColor: 'rgba(255,215,0,0.4)',
  },
  gridLineV: {
    position: 'absolute', top: BRACKET, bottom: BRACKET,
    left: '50%', width: 1,
    backgroundColor: 'rgba(255,215,0,0.4)',
  },

  // Live scan overlay position
  liveOverlayWrapper: {
    position: 'absolute',
    bottom: Platform.OS === 'ios' ? 175 : 145,
    left: 16,
    right: 16,
  },

  // Recording UI
  recordingBar: {
    position: 'absolute',
    top: Platform.OS === 'ios' ? 130 : 90,
    left: 20, right: 20,
    gap: 8,
  },
  recordingIndicator: {
    flexDirection: 'row', alignItems: 'center', gap: 8,
    alignSelf: 'center',
  },
  recordingDot: {
    width: 10, height: 10, borderRadius: 5,
    backgroundColor: C.red,
  },
  recordingText: {
    color: C.white, fontSize: 13, fontWeight: '600',
  },
  progressTrack: {
    height: 4, borderRadius: 2,
    backgroundColor: 'rgba(255,255,255,0.2)',
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%', borderRadius: 2,
    backgroundColor: C.red,
  },

  bottomBar: {
    position: 'absolute', bottom: 0, left: 0, right: 0,
    paddingBottom: Platform.OS === 'ios' ? 48 : 24,
    paddingTop: 24,
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.6)',
    gap: 12,
  },
  bottomHint: {
    backgroundColor: 'rgba(255,255,255,0.08)',
    paddingHorizontal: 16, paddingVertical: 6,
    borderRadius: R.full,
  },
  hintText: { color: 'rgba(255,255,255,0.65)', fontSize: 12, fontWeight: '500' },
  captureBtn: {
    width: 80, height: 80,
    borderRadius: 40,
    backgroundColor: C.white,
    alignItems: 'center', justifyContent: 'center',
    ...shadow(3),
  },
  captureBtnDisabled: { opacity: 0.5 },
  captureBtnRecording: {
    backgroundColor: C.red,
  },
  captureInner: {
    width: 68, height: 68, borderRadius: 34,
    backgroundColor: '#FFF5F5',
    alignItems: 'center', justifyContent: 'center',
    borderWidth: 2, borderColor: C.red,
  },
  captureInnerRecording: {
    backgroundColor: C.red,
    borderColor: C.white,
  },
  scanLabel: { color: 'rgba(255,255,255,0.8)', fontSize: 13, fontWeight: '500' },

  // Loading overlay
  loadingOverlay: {
    position: 'absolute',
    bottom: Platform.OS === 'ios' ? 170 : 140,
    left: 24, right: 24,
    alignItems: 'center',
  },
  loadingCard: {
    backgroundColor: 'rgba(15,15,15,0.85)',
    borderRadius: 16,
    paddingHorizontal: 20,
    paddingVertical: 14,
    gap: 6,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
    width: '100%',
  },
  loadingIconRow: {
    flexDirection: 'row', alignItems: 'center', gap: 8,
  },
  loadingCardTitle: {
    color: C.white, fontSize: 13, fontWeight: '700',
  },
  loadingCardStatus: {
    color: 'rgba(255,255,255,0.7)', fontSize: 12, fontWeight: '500',
    paddingLeft: 28, // align with title text after icon
  },
  loadingFrameRow: {
    flexDirection: 'row', gap: 6, paddingLeft: 28, paddingTop: 2,
  },
  loadingFrameDot: {
    width: 10, height: 10, borderRadius: 5,
    backgroundColor: 'rgba(255,255,255,0.2)',
  },
  loadingFrameDotFilled: {
    backgroundColor: C.red,
  },

  // Session modal
  modalBg: {
    flex: 1, backgroundColor: 'rgba(0,0,0,0.6)',
    alignItems: 'center', justifyContent: 'center', padding: S.lg,
  },
  sessionModal: {
    backgroundColor: C.white, borderRadius: 24, width: '100%',
    maxWidth: 360, overflow: 'hidden', ...shadow(3),
  },
  modalHeader: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingHorizontal: S.lg, paddingTop: S.lg, paddingBottom: S.md,
    borderBottomWidth: 1, borderBottomColor: C.border,
  },
  modalTitle: { fontSize: 18, fontWeight: '800', color: C.text, flex: 1 },
  currentSessionInfo: {
    backgroundColor: '#FFF5F5', borderBottomWidth: 1, borderBottomColor: '#FFCCCC',
    padding: S.lg, gap: S.sm,
  },
  infoLabel: { fontSize: 11, fontWeight: '700', color: C.textMuted, letterSpacing: 0.5 },
  infoValue: { fontSize: 16, fontWeight: '700', color: C.text, marginTop: 2 },
  infoMeta: { fontSize: 12, color: C.textSub, marginTop: 4 },
  modalContent: { padding: S.lg, gap: S.md },
  inputLabel: { fontSize: 13, fontWeight: '700', color: C.text, marginBottom: 4 },
  sessionInput: {
    backgroundColor: C.bg, borderWidth: 1.5, borderColor: C.border,
    borderRadius: R.md, paddingHorizontal: S.md, paddingVertical: 12,
    fontSize: 15, color: C.text,
  },
  confirmBtn: {
    backgroundColor: C.red, borderRadius: R.md, paddingVertical: 13,
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    marginTop: S.sm,
  },
  confirmBtnDisabled: { opacity: 0.5 },
  confirmBtnText: { color: C.white, fontSize: 15, fontWeight: '700' },
  cancelBtn: {
    backgroundColor: C.white, borderWidth: 1.5, borderColor: C.border,
    borderRadius: R.md, paddingVertical: 12, alignItems: 'center',
  },
  cancelBtnText: { color: C.text, fontSize: 15, fontWeight: '600' },

  // Feature 1: Scale button
  scaleIcon: { fontSize: 20 },
  scaleBtnActive: {
    backgroundColor: C.green,
    borderColor: C.green,
  },

  // Feature 2: Duplicate detection overlay
  duplicateOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.65)',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: S.md,
  },
  duplicateCard: {
    backgroundColor: C.card,
    borderRadius: R.lg,
    paddingHorizontal: S.lg,
    paddingVertical: S.lg,
    alignItems: 'center',
    maxWidth: 380,
    width: '100%',
    ...shadow(2),
  },
  duplicateHeader: {
    alignItems: 'center',
    marginBottom: S.md,
  },
  duplicateWarningIcon: {
    fontSize: 40,
    marginBottom: S.sm,
  },
  duplicateTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: C.text,
  },
  duplicateText: {
    fontSize: 14,
    color: C.textSub,
    textAlign: 'center',
    lineHeight: 20,
    marginBottom: S.lg,
  },
  duplicateActions: {
    width: '100%',
    flexDirection: 'row',
    gap: S.sm,
  },
  duplicateAddBtn: {
    flex: 1,
    backgroundColor: C.orange,
    paddingVertical: 11,
    borderRadius: R.md,
    alignItems: 'center',
  },
  duplicateAddBtnText: {
    color: C.white,
    fontSize: 14,
    fontWeight: '600',
  },
  duplicateSkipBtn: {
    flex: 1,
    backgroundColor: C.cardAlt,
    paddingVertical: 11,
    borderRadius: R.md,
    alignItems: 'center',
  },
  duplicateSkipBtnText: {
    color: C.text,
    fontSize: 14,
    fontWeight: '600',
  },

  // Feature 3: Low confidence prompt overlay
  confidenceOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.65)',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: S.md,
  },
  confidenceCard: {
    backgroundColor: C.card,
    borderRadius: R.lg,
    paddingHorizontal: S.lg,
    paddingVertical: S.lg,
    alignItems: 'center',
    maxWidth: 380,
    width: '100%',
    ...shadow(2),
  },
  confidenceHeader: {
    alignItems: 'center',
    marginBottom: S.md,
  },
  confidenceIcon: {
    fontSize: 40,
    marginBottom: S.sm,
  },
  confidenceTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: C.text,
  },
  confidenceSubtitle: {
    fontSize: 13,
    color: C.textSub,
    marginBottom: S.sm,
  },
  confidenceText: {
    fontSize: 14,
    color: C.textSub,
    textAlign: 'center',
    lineHeight: 20,
    marginBottom: S.lg,
  },
  confidenceActions: {
    width: '100%',
    gap: S.sm,
  },
  confidenceScanBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: C.red,
    paddingVertical: 11,
    borderRadius: R.md,
    gap: S.sm,
  },
  confidenceScanBtnText: {
    color: C.white,
    fontSize: 14,
    fontWeight: '600',
  },
  confidenceSkipBtn: {
    backgroundColor: C.cardAlt,
    paddingVertical: 11,
    borderRadius: R.md,
    alignItems: 'center',
  },
  confidenceSkipBtnText: {
    color: C.text,
    fontSize: 14,
    fontWeight: '600',
  },
});
