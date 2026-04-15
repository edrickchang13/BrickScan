import React, { useEffect, useRef } from 'react';
import { Animated, StyleSheet, Text, View } from 'react-native';
import { C, R, S, T } from '@/constants/theme';

export interface ScanProgressProps {
  percent: number;          // 0-100
  stageLabel: string;       // human-readable e.g. "Asking Gemini…"
  topPrediction?: {
    partNum: string;
    partName?: string;
    confidence: number;
    source?: string;
  };
}

const STAGE_FRIENDLY: Record<string, string> = {
  decode: 'Preparing image…',
  brickognize_start: 'Querying Brickognize…',
  brickognize_done: 'Brickognize result in',
  gemini_start: 'Asking Gemini…',
  gemini_done: 'Gemini result in',
  local_models: 'Running local models…',
  merge: 'Merging predictions…',
  tta: 'Stabilising with rotation…',
  multipiece: 'Looking for additional pieces…',
  persist: 'Saving scan…',
  enrich: 'Looking up part details…',
  done: 'Done',
};

export const friendlyStageLabel = (stage: string, fallback?: string): string =>
  STAGE_FRIENDLY[stage] || fallback || stage;

export const ScanProgress: React.FC<ScanProgressProps> = ({ percent, stageLabel, topPrediction }) => {
  const animatedWidth = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const target = percent < 0 ? 0 : Math.min(100, percent);
    Animated.timing(animatedWidth, {
      toValue: target,
      duration: 250,
      useNativeDriver: false,
    }).start();
  }, [percent, animatedWidth]);

  const widthInterpolation = animatedWidth.interpolate({
    inputRange: [0, 100],
    outputRange: ['0%', '100%'],
  });

  return (
    <View style={styles.wrap}>
      <View style={styles.barTrack}>
        <Animated.View style={[styles.barFill, { width: widthInterpolation }]} />
      </View>
      <View style={styles.row}>
        <Text style={styles.stageText} numberOfLines={1}>{stageLabel}</Text>
        <Text style={styles.percentText}>{percent < 0 ? '' : `${Math.round(percent)}%`}</Text>
      </View>

      {topPrediction && (
        <View style={styles.partialCard}>
          <Text style={styles.partialLabel}>Partial guess</Text>
          <Text style={styles.partialPart} numberOfLines={1}>
            {topPrediction.partName || topPrediction.partNum}{' '}
            <Text style={styles.partialConf}>· {Math.round(topPrediction.confidence * 100)}%</Text>
          </Text>
          {topPrediction.source && (
            <Text style={styles.partialSource}>via {topPrediction.source}</Text>
          )}
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  wrap: {
    width: '100%',
    paddingHorizontal: S.md,
  },
  barTrack: {
    height: 6,
    borderRadius: R.full,
    backgroundColor: C.bgDark,
    overflow: 'hidden',
  },
  barFill: {
    height: '100%',
    backgroundColor: C.red,
    borderRadius: R.full,
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: S.sm,
  },
  stageText: {
    ...T.bodySmall,
    flex: 1,
    marginRight: S.sm,
  },
  percentText: {
    ...T.caption,
    color: C.textSub,
    fontVariant: ['tabular-nums'],
  },
  partialCard: {
    marginTop: S.md,
    padding: S.sm,
    borderRadius: R.md,
    backgroundColor: C.cardAlt,
    borderWidth: 1,
    borderColor: C.border,
  },
  partialLabel: {
    ...T.caption,
    textTransform: 'uppercase' as const,
    color: C.textMuted,
    marginBottom: 2,
  },
  partialPart: {
    ...T.h4,
    color: C.text,
  },
  partialConf: {
    ...T.bodySmall,
    color: C.textSub,
  },
  partialSource: {
    ...T.caption,
    color: C.textMuted,
    marginTop: 2,
  },
});
