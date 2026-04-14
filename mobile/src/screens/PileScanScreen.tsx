import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  View, TouchableOpacity, Text, Alert, StyleSheet, StatusBar, ScrollView,
  Animated, Image, ActivityIndicator,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { useIsFocused } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import { apiClient, PileResult } from '@/services/api';
import * as ImageManipulator from 'expo-image-manipulator';
import * as FileSystem from 'expo-file-system';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { ScanStackParamList } from '@/types';
import { C, R, S, T, shadow, gs } from '@/constants/theme';

type Props = NativeStackScreenProps<ScanStackParamList, 'PileScanScreen'>;

const PileScanScreen: React.FC<Props> = ({ navigation }) => {
  const [permission, requestPermission] = useCameraPermissions();
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState('');
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [pileResults, setPileResults] = useState<PileResult[]>([]);
  const [totalBricksDetected, setTotalBricksDetected] = useState(0);
  const [hasError, setHasError] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const [loadingPulse] = useState(new Animated.Value(1));

  const cameraRef = useRef<CameraView>(null);
  const isFocused = useIsFocused();

  // Pulse animation for loading overlay
  useEffect(() => {
    if (!isLoading) {
      loadingPulse.setValue(1);
      return;
    }
    const pulse = Animated.loop(
      Animated.sequence([
        Animated.timing(loadingPulse, { toValue: 0.5, duration: 700, useNativeDriver: true }),
        Animated.timing(loadingPulse, { toValue: 1, duration: 700, useNativeDriver: true }),
      ])
    );
    pulse.start();
    return () => pulse.stop();
  }, [isLoading]);

  useEffect(() => {
    if (!permission) requestPermission();
  }, [permission]);

  const handlePileCapture = async () => {
    if (!cameraRef.current || !isCameraReady) {
      Alert.alert('Camera', 'Camera not ready — please wait a moment.');
      return;
    }
    setIsLoading(true);
    setLoadingStatus('Capturing image…');
    setHasError(false);
    setErrorMsg('');

    try {
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.6 });
      if (!photo?.uri) throw new Error('Failed to capture image');

      setLoadingStatus('Processing…');
      const manipulated = await ImageManipulator.manipulateAsync(
        photo.uri,
        [{ resize: { width: 1024, height: 1024 } }],
        { compress: 0.7, format: ImageManipulator.SaveFormat.JPEG },
      );
      const base64 = await FileSystem.readAsStringAsync(manipulated.uri, {
        encoding: FileSystem.EncodingType.Base64,
      });

      setLoadingStatus('Analyzing pile…');
      const results = await apiClient.scanPile(base64);

      if (results && results.length > 0) {
        const totalCount = results.reduce((sum, r) => sum + r.count, 0);
        setTotalBricksDetected(totalCount);
        setPileResults(results);
      } else {
        setHasError(true);
        setErrorMsg('No bricks detected in the pile. Try better lighting or angle.');
        setPileResults([]);
        setTotalBricksDetected(0);
      }
    } catch (error: any) {
      setHasError(true);
      setErrorMsg(error?.message || 'Failed to scan pile. Please try again.');
      setPileResults([]);
      setTotalBricksDetected(0);
    } finally {
      setIsLoading(false);
      setLoadingStatus('');
    }
  };

  const handleAddRowToInventory = async (result: PileResult) => {
    try {
      await apiClient.addToInventory(
        result.partNum,
        result.colorId ? String(result.colorId) : '',
        result.count,
        result.colorName,
      );
      Alert.alert('Added', `${result.count}x ${result.partName} added to inventory`);
    } catch (error: any) {
      Alert.alert('Error', error?.message || 'Failed to add to inventory');
    }
  };

  const handleAddAllToInventory = async () => {
    try {
      let added = 0;
      for (const result of pileResults) {
        await apiClient.addToInventory(
          result.partNum,
          result.colorId ? String(result.colorId) : '',
          result.count,
          result.colorName,
        );
        added++;
      }
      Alert.alert('Success', `${added} part types added to inventory`);
      resetScan();
    } catch (error: any) {
      Alert.alert('Error', error?.message || 'Failed to add items to inventory');
    }
  };

  const resetScan = () => {
    setPileResults([]);
    setTotalBricksDetected(0);
    setHasError(false);
    setErrorMsg('');
  };

  // ─── Camera view (shown when no results yet) ───────────────────────────────
  if (pileResults.length === 0 && !hasError) {
    return (
      <View style={styles.container}>
        <StatusBar barStyle="dark-content" backgroundColor={C.white} />
        <CameraView
          ref={cameraRef}
          style={styles.camera}
          onCameraReady={() => setIsCameraReady(true)}
          facing="back"
        >
          {/* Semi-transparent overlay to help user frame the shot */}
          <View style={styles.cameraOverlay}>
            <View style={styles.frameGuide} />
            <Text style={styles.frameLabel}>Photograph loose bricks in a pile</Text>
          </View>

          {/* Floating capture button */}
          <View style={styles.floatingButtonContainer}>
            <TouchableOpacity
              style={styles.captureButton}
              onPress={handlePileCapture}
              disabled={isLoading}
            >
              <Ionicons
                name={isLoading ? 'hourglass' : 'camera'}
                size={32}
                color={C.white}
              />
              <Text style={styles.captureButtonText}>
                {isLoading ? 'Scanning...' : 'Capture Pile'}
              </Text>
            </TouchableOpacity>
          </View>

          {/* Loading overlay */}
          {isLoading && (
            <View style={styles.loadingOverlay}>
              <Animated.View style={{ opacity: loadingPulse }}>
                <View style={styles.loadingCard}>
                  <ActivityIndicator size="large" color={C.red} />
                  <Text style={styles.loadingStatusText}>{loadingStatus}</Text>
                </View>
              </Animated.View>
            </View>
          )}
        </CameraView>
      </View>
    );
  }

  // ─── Error state ──────────────────────────────────────────────────────────
  if (hasError) {
    return (
      <View style={[gs.screenBg, styles.errorContainer]}>
        <StatusBar barStyle="dark-content" backgroundColor={C.bg} />
        <View style={styles.errorContent}>
          <Ionicons name="alert-circle" size={48} color={C.red} />
          <Text style={styles.errorTitle}>Detection Failed</Text>
          <Text style={styles.errorMessage}>{errorMsg}</Text>
          <TouchableOpacity
            style={[gs.btnPrimary, styles.errorButton]}
            onPress={() => {
              resetScan();
              navigation.goBack();
            }}
          >
            <Text style={gs.btnPrimaryText}>Go Back</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  // ─── Result view ──────────────────────────────────────────────────────────
  return (
    <View style={[gs.screenBg, styles.resultContainer]}>
      <StatusBar barStyle="dark-content" backgroundColor={C.bg} />

      {/* Header: total count */}
      <View style={styles.headerCard}>
        <Text style={styles.headerTitle}>{totalBricksDetected} bricks detected</Text>
        <Text style={styles.headerSubtitle}>{pileResults.length} unique part types</Text>
      </View>

      {/* ScrollView: list of results */}
      <ScrollView
        style={styles.resultsList}
        contentContainerStyle={styles.resultsListContent}
        showsVerticalScrollIndicator={false}
      >
        {pileResults.map((result, index) => (
          <PileResultRow
            key={`${result.partNum}-${index}`}
            result={result}
            index={index}
            onAddRow={() => handleAddRowToInventory(result)}
          />
        ))}
      </ScrollView>

      {/* Action buttons */}
      <View style={styles.actionButtons}>
        <TouchableOpacity
          style={[gs.btnPrimary, styles.addAllButton]}
          onPress={handleAddAllToInventory}
        >
          <Text style={gs.btnPrimaryText}>Add All to Inventory</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[gs.btnSecondary, styles.scanAgainButton]}
          onPress={() => {
            resetScan();
          }}
        >
          <Text style={gs.btnSecondaryText}>Scan Another Pile</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

// ─── PileResultRow component ──────────────────────────────────────────────────

interface PileResultRowProps {
  result: PileResult;
  index: number;
  onAddRow: () => void;
}

const PileResultRow: React.FC<PileResultRowProps> = ({ result, index, onAddRow }) => {
  const slideAnim = useRef(new Animated.Value(300)).current;

  useEffect(() => {
    Animated.spring(slideAnim, {
      toValue: 0,
      useNativeDriver: true,
      speed: 12,
      bounciness: 8,
      delay: index * 50,
    }).start();
  }, [index]);

  const hasImage = result.cropImageBase64 && result.cropImageBase64.length > 0;

  return (
    <Animated.View
      style={[
        styles.resultRow,
        { transform: [{ translateX: slideAnim }] },
      ]}
    >
      {/* Thumbnail */}
      <View style={styles.thumbnailContainer}>
        {hasImage ? (
          <Image
            source={{ uri: `data:image/jpeg;base64,${result.cropImageBase64}` }}
            style={styles.thumbnail}
          />
        ) : (
          <View style={[styles.thumbnail, styles.thumbnailPlaceholder]}>
            <Ionicons name="cube" size={32} color={C.textMuted} />
          </View>
        )}
      </View>

      {/* Part info */}
      <View style={styles.partInfo}>
        <Text style={styles.partName} numberOfLines={1}>
          {result.partName}
        </Text>
        <Text style={styles.partNum}>{result.partNum}</Text>
        {result.colorName && <Text style={styles.colorName}>{result.colorName}</Text>}
      </View>

      {/* Count badge */}
      <View style={styles.countBadge}>
        <Text style={styles.countText}>×{result.count}</Text>
      </View>

      {/* Confidence bar */}
      <View style={styles.confidenceCol}>
        <ConfidenceBar value={result.confidence} />
        <Text style={styles.confidenceLabel}>
          {Math.round(result.confidence * 100)}%
        </Text>
      </View>

      {/* Add button */}
      <TouchableOpacity style={styles.addButton} onPress={onAddRow}>
        <Ionicons name="add-circle" size={28} color={C.red} />
      </TouchableOpacity>
    </Animated.View>
  );
};

// ─── ConfidenceBar sub-component ──────────────────────────────────────────────

const ConfidenceBar: React.FC<{ value: number }> = ({ value }) => {
  const anim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.spring(anim, {
      toValue: value,
      useNativeDriver: false,
      speed: 14,
      bounciness: 2,
    }).start();
  }, [value]);

  const width = anim.interpolate({
    inputRange: [0, 1],
    outputRange: ['0%', '100%'],
  });

  return (
    <View style={styles.confidenceTrack}>
      <Animated.View
        style={[
          styles.confidenceFill,
          { width, backgroundColor: C.green },
        ]}
      />
    </View>
  );
};

// ─── Styles ───────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: C.white,
  },
  camera: {
    flex: 1,
  },
  cameraOverlay: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.2)',
  },
  frameGuide: {
    width: 280,
    height: 280,
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.6)',
    borderRadius: R.lg,
    marginBottom: S.lg,
  },
  frameLabel: {
    ...T.bodySmall,
    color: 'rgba(255, 255, 255, 0.8)',
    textAlign: 'center',
    paddingHorizontal: S.lg,
  },
  floatingButtonContainer: {
    position: 'absolute',
    bottom: S.xxl,
    left: 0,
    right: 0,
    alignItems: 'center',
    zIndex: 10,
  },
  captureButton: {
    backgroundColor: C.red,
    borderRadius: R.full,
    width: 80,
    height: 80,
    alignItems: 'center',
    justifyContent: 'center',
    ...shadow(3),
  },
  captureButtonText: {
    ...T.caption,
    color: C.white,
    marginTop: S.xs,
    fontWeight: '600',
  },
  loadingOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 20,
  },
  loadingCard: {
    backgroundColor: C.white,
    borderRadius: R.lg,
    paddingVertical: S.xl,
    paddingHorizontal: S.lg,
    alignItems: 'center',
    ...shadow(2),
    minWidth: 200,
  },
  loadingStatusText: {
    ...T.body,
    marginTop: S.md,
    textAlign: 'center',
    color: C.textSub,
  },
  errorContainer: {
    ...gs.emptyState,
  },
  errorContent: {
    alignItems: 'center',
  },
  errorTitle: {
    ...gs.emptyTitle,
  },
  errorMessage: {
    ...gs.emptySubtitle,
  },
  errorButton: {
    marginTop: S.lg,
    minWidth: 200,
  },
  resultContainer: {
    flex: 1,
    paddingTop: S.md,
  },
  headerCard: {
    backgroundColor: C.card,
    marginHorizontal: S.md,
    marginBottom: S.md,
    paddingVertical: S.lg,
    paddingHorizontal: S.md,
    borderRadius: R.lg,
    ...shadow(1),
  },
  headerTitle: {
    ...T.h3,
    color: C.text,
  },
  headerSubtitle: {
    ...T.bodySmall,
    color: C.textMuted,
    marginTop: S.xs,
  },
  resultsList: {
    flex: 1,
    marginHorizontal: S.md,
  },
  resultsListContent: {
    paddingBottom: S.md,
    gap: S.md,
  },
  resultRow: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: C.card,
    borderRadius: R.lg,
    padding: S.md,
    gap: S.md,
    ...shadow(1),
  },
  thumbnailContainer: {
    width: 60,
    height: 60,
  },
  thumbnail: {
    width: '100%',
    height: '100%',
    borderRadius: R.md,
    backgroundColor: C.bgDark,
  },
  thumbnailPlaceholder: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  partInfo: {
    flex: 1,
  },
  partName: {
    ...T.body,
    fontWeight: '600',
    color: C.text,
  },
  partNum: {
    ...T.caption,
    color: C.textMuted,
    marginTop: S.xs,
  },
  colorName: {
    ...T.caption,
    color: C.textSub,
    marginTop: S.xs,
  },
  countBadge: {
    backgroundColor: C.yellow,
    borderRadius: R.full,
    paddingHorizontal: S.sm,
    paddingVertical: S.xs,
    minWidth: 44,
    alignItems: 'center',
    justifyContent: 'center',
  },
  countText: {
    ...T.label,
    color: C.text,
    fontWeight: '700',
  },
  confidenceCol: {
    width: 50,
    alignItems: 'center',
    gap: S.xs,
  },
  confidenceTrack: {
    width: '100%',
    height: 5,
    borderRadius: 3,
    backgroundColor: C.bgDark,
    overflow: 'hidden',
  },
  confidenceFill: {
    height: '100%',
    borderRadius: 3,
  },
  confidenceLabel: {
    ...T.caption,
    color: C.textMuted,
    fontSize: 10,
  },
  addButton: {
    padding: S.sm,
  },
  actionButtons: {
    gap: S.md,
    paddingHorizontal: S.md,
    paddingVertical: S.md,
    backgroundColor: C.white,
  },
  addAllButton: {
    marginBottom: 0,
  },
  scanAgainButton: {
    marginBottom: 0,
  },
});

export default PileScanScreen;
