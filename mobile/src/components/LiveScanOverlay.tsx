/**
 * LiveScanOverlay
 *
 * Displays real-time confidence accumulation during video scan mode.
 * Shows top candidate bricks with animated bars that fill as more
 * frames agree. Locks the top result once confidence crosses the
 * threshold and plays a visual confirmation.
 *
 * Usage:
 *   <LiveScanOverlay
 *     votes={votesMap}           // Map<partNum, {name, totalWeight, frameCount}>
 *     totalFrames={frameCount}
 *     lockedResult={lockedResult}
 *     threshold={0.75}
 *     onLocked={(result) => navigation.navigate('ScanResultScreen', ...)}
 *   />
 */

import React, { useEffect, useRef, useMemo } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Animated,
  TouchableOpacity,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { C, R, S, shadow } from '@/constants/theme';

// ─── Types ───────────────────────────────────────────────────────────────────

export interface VoteEntry {
  partNum: string;
  name: string;
  totalWeight: number;   // sum of confidence scores across all agreeing frames
  frameCount: number;    // how many frames voted for this part
  colorId?: number;
  colorName?: string;
}

export interface LockedResult {
  partNum: string;
  name: string;
  confidence: number;    // 0-1, fraction of frames that agreed
  frameCount: number;
  colorId?: number;
  colorName?: string;
}

interface Props {
  votes: Map<string, VoteEntry>;
  totalFrames: number;
  lockedResult: LockedResult | null;
  threshold?: number;          // agreement fraction to trigger lock (default 0.75)
  onAccept?: (result: LockedResult) => void;
  onContinue?: () => void;     // keep scanning for more confidence
  mode?: 'vote' | 'multiview'; // scan mode (default 'vote')
}

const MAX_SHOWN = 3;
const DEFAULT_THRESHOLD = 0.75;
const MIN_FRAMES_TO_LOCK = 6;    // require at least N frames before locking

// ─── Sub-component: animated confidence bar ──────────────────────────────────

const ConfidenceBar: React.FC<{
  value: number;         // 0-1
  isLeader: boolean;
  isLocked: boolean;
  delay?: number;
}> = ({ value, isLeader, isLocked, delay = 0 }) => {
  const anim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.spring(anim, {
      toValue: value,
      useNativeDriver: false,
      speed: 14,
      bounciness: 2,
      delay,
    }).start();
  }, [value]);

  const width = anim.interpolate({
    inputRange: [0, 1],
    outputRange: ['0%', '100%'],
  });

  const barColor = isLocked
    ? C.green
    : isLeader
    ? C.red
    : 'rgba(255,255,255,0.35)';

  return (
    <View style={barStyles.track}>
      <Animated.View
        style={[
          barStyles.fill,
          { width, backgroundColor: barColor },
        ]}
      />
    </View>
  );
};

const barStyles = StyleSheet.create({
  track: {
    height: 5,
    borderRadius: 3,
    backgroundColor: 'rgba(255,255,255,0.12)',
    overflow: 'hidden',
    flex: 1,
  },
  fill: {
    height: '100%',
    borderRadius: 3,
  },
});

// ─── Sub-component: single candidate row ─────────────────────────────────────

const CandidateRow: React.FC<{
  entry: VoteEntry;
  totalFrames: number;
  rank: number;
  isLocked: boolean;
}> = ({ entry, totalFrames, rank, isLocked }) => {
  const confidence = totalFrames > 0 ? entry.frameCount / totalFrames : 0;
  const pct = Math.round(confidence * 100);
  const isLeader = rank === 0;

  return (
    <View style={[rowStyles.row, isLeader && rowStyles.rowLeader]}>
      {/* Rank badge */}
      <View style={[rowStyles.rankBadge, isLeader && rowStyles.rankBadgeLeader]}>
        {isLocked && isLeader ? (
          <Ionicons name="checkmark" size={11} color={C.white} />
        ) : (
          <Text style={[rowStyles.rankText, isLeader && rowStyles.rankTextLeader]}>
            {rank + 1}
          </Text>
        )}
      </View>

      {/* Part info + bar */}
      <View style={rowStyles.info}>
        <View style={rowStyles.nameRow}>
          <Text style={[rowStyles.partNum, isLeader && rowStyles.partNumLeader]}
                numberOfLines={1}>
            {entry.name || entry.partNum}
          </Text>
          <Text style={[rowStyles.pct, isLeader && rowStyles.pctLeader]}>
            {pct}%
          </Text>
        </View>
        <View style={rowStyles.barRow}>
          <ConfidenceBar
            value={confidence}
            isLeader={isLeader}
            isLocked={isLocked && isLeader}
            delay={rank * 60}
          />
          <Text style={rowStyles.frameCount}>
            {entry.frameCount}/{totalFrames}f
          </Text>
        </View>
      </View>
    </View>
  );
};

const rowStyles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingVertical: 6,
  },
  rowLeader: {},
  rankBadge: {
    width: 22,
    height: 22,
    borderRadius: 11,
    backgroundColor: 'rgba(255,255,255,0.15)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  rankBadgeLeader: {
    backgroundColor: C.red,
  },
  rankText: {
    color: 'rgba(255,255,255,0.7)',
    fontSize: 11,
    fontWeight: '700',
  },
  rankTextLeader: {
    color: C.white,
  },
  info: {
    flex: 1,
    gap: 4,
  },
  nameRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'baseline',
  },
  partNum: {
    color: 'rgba(255,255,255,0.75)',
    fontSize: 12,
    fontWeight: '600',
    flex: 1,
    marginRight: 8,
  },
  partNumLeader: {
    color: C.white,
    fontSize: 13,
    fontWeight: '800',
  },
  pct: {
    color: 'rgba(255,255,255,0.6)',
    fontSize: 12,
    fontWeight: '700',
    minWidth: 34,
    textAlign: 'right',
  },
  pctLeader: {
    color: C.yellow,
    fontSize: 14,
  },
  barRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  frameCount: {
    color: 'rgba(255,255,255,0.35)',
    fontSize: 9,
    fontWeight: '600',
    minWidth: 32,
    textAlign: 'right',
  },
});

// ─── Lock pulse animation ─────────────────────────────────────────────────────

const LockPulse: React.FC = () => {
  const scale = useRef(new Animated.Value(0)).current;
  const opacity = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.sequence([
      Animated.parallel([
        Animated.spring(scale, {
          toValue: 1,
          useNativeDriver: true,
          speed: 6,
          bounciness: 10,
        }),
        Animated.timing(opacity, {
          toValue: 1,
          duration: 200,
          useNativeDriver: true,
        }),
      ]),
    ]).start();
  }, []);

  return (
    <Animated.View
      style={[lockStyles.badge, { transform: [{ scale }], opacity }]}
    >
      <Ionicons name="checkmark-circle" size={18} color={C.green} />
      <Text style={lockStyles.text}>IDENTIFIED</Text>
    </Animated.View>
  );
};

const lockStyles = StyleSheet.create({
  badge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: 'rgba(22, 163, 74, 0.2)',
    borderWidth: 1,
    borderColor: C.green,
    borderRadius: R.full,
    paddingHorizontal: 10,
    paddingVertical: 4,
    alignSelf: 'flex-start',
  },
  text: {
    color: C.green,
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 0.8,
  },
});

// ─── Main component ──────────────────────────────────────────────────────────

export const LiveScanOverlay: React.FC<Props> = ({
  votes,
  totalFrames,
  lockedResult,
  threshold = DEFAULT_THRESHOLD,
  onAccept,
  onContinue,
  mode = 'vote',
}) => {
  const slideAnim = useRef(new Animated.Value(120)).current;

  // Slide up on mount
  useEffect(() => {
    Animated.spring(slideAnim, {
      toValue: 0,
      useNativeDriver: true,
      speed: 14,
      bounciness: 4,
    }).start();
  }, []);

  // Sort candidates by frame count descending
  const topCandidates = useMemo(() => {
    return Array.from(votes.values())
      .sort((a, b) => b.frameCount - a.frameCount)
      .slice(0, MAX_SHOWN);
  }, [votes]);

  const isLocked = lockedResult !== null;
  const hasEnoughData = totalFrames >= MIN_FRAMES_TO_LOCK;
  const isMultiview = mode === 'multiview';
  const targetFrames = isMultiview ? 8 : MIN_FRAMES_TO_LOCK;

  if (isMultiview) {
    // Multiview mode: show progress indicator instead of vote percentages
    if (totalFrames === 0) {
      return (
        <Animated.View
          style={[styles.card, { transform: [{ translateY: slideAnim }] }]}
        >
          <View style={styles.scanningRow}>
            <View style={styles.scanDot} />
            <Text style={styles.scanningText}>Collecting views… point camera at a brick</Text>
          </View>
        </Animated.View>
      );
    }

    if (totalFrames < targetFrames && !isLocked) {
      return (
        <Animated.View
          style={[styles.card, { transform: [{ translateY: slideAnim }] }]}
        >
          <View style={styles.header}>
            <View style={styles.headerLeft}>
              <Ionicons name="sparkles" size={14} color={C.yellow} />
              <Text style={styles.headerTitle}>Collecting views…</Text>
            </View>
            <Text style={styles.framesText}>{totalFrames}/{targetFrames}</Text>
          </View>
          <View style={styles.divider} />
          <View style={styles.multiviewProgressContainer}>
            <View style={styles.progressBar}>
              <View
                style={[
                  styles.progressFill,
                  { width: `${(totalFrames / targetFrames) * 100}%` },
                ]}
              />
            </View>
            <Text style={styles.progressText}>
              {totalFrames} of {targetFrames} views
            </Text>
          </View>
        </Animated.View>
      );
    }

    if (isLocked && totalFrames >= targetFrames) {
      return (
        <Animated.View
          style={[styles.card, { transform: [{ translateY: slideAnim }] }]}
        >
          <View style={styles.header}>
            <View style={styles.headerLeft}>
              <Ionicons name="checkmark-circle" size={14} color={C.green} />
              <Text style={styles.headerTitle}>Analyzing…</Text>
            </View>
          </View>
          <View style={styles.divider} />
          <View style={styles.multiviewResultContainer}>
            <Text style={styles.multiviewResultText}>
              Processing {totalFrames} views with attention pooling
            </Text>
          </View>
        </Animated.View>
      );
    }
  }

  // Vote mode (original behavior)
  if (topCandidates.length === 0 && totalFrames < 2) {
    // Not enough data yet — show scanning indicator
    return (
      <Animated.View
        style={[styles.card, { transform: [{ translateY: slideAnim }] }]}
      >
        <View style={styles.scanningRow}>
          <View style={styles.scanDot} />
          <Text style={styles.scanningText}>Scanning… point camera at a brick</Text>
        </View>
      </Animated.View>
    );
  }

  return (
    <Animated.View
      style={[styles.card, { transform: [{ translateY: slideAnim }] }]}
    >
      {/* Header row */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <Ionicons name="sparkles" size={14} color={isLocked ? C.green : C.yellow} />
          <Text style={styles.headerTitle}>
            {isLocked ? 'Match Found' : 'Scanning…'}
          </Text>
        </View>
        <View style={styles.headerRight}>
          {isLocked && <LockPulse />}
          {!isLocked && hasEnoughData && (
            <Text style={styles.framesText}>{totalFrames} frames</Text>
          )}
          {!hasEnoughData && (
            <Text style={styles.framesTextDim}>need {MIN_FRAMES_TO_LOCK - totalFrames} more frames</Text>
          )}
        </View>
      </View>

      {/* Divider */}
      <View style={styles.divider} />

      {/* Candidate list */}
      {topCandidates.map((entry, rank) => (
        <CandidateRow
          key={entry.partNum}
          entry={entry}
          totalFrames={totalFrames}
          rank={rank}
          isLocked={isLocked && rank === 0}
        />
      ))}

      {/* Action buttons when locked */}
      {isLocked && lockedResult && (
        <>
          <View style={styles.divider} />
          <View style={styles.actions}>
            <TouchableOpacity
              style={styles.continueBtn}
              onPress={onContinue}
              activeOpacity={0.75}
            >
              <Ionicons name="refresh" size={15} color="rgba(255,255,255,0.7)" />
              <Text style={styles.continueBtnText}>Keep Scanning</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.acceptBtn}
              onPress={() => onAccept?.(lockedResult)}
              activeOpacity={0.8}
            >
              <Ionicons name="checkmark" size={16} color={C.white} />
              <Text style={styles.acceptBtnText}>Accept Result</Text>
            </TouchableOpacity>
          </View>
        </>
      )}
    </Animated.View>
  );
};

// ─── Styles ───────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  card: {
    backgroundColor: 'rgba(10, 10, 10, 0.88)',
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 14,
    gap: 0,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
    ...shadow(3),
    // Note: backdropFilter is web-only — omitted for React Native / Xcode compatibility
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  headerTitle: {
    color: C.white,
    fontSize: 13,
    fontWeight: '700',
  },
  headerRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  framesText: {
    color: 'rgba(255,255,255,0.5)',
    fontSize: 11,
    fontWeight: '600',
  },
  framesTextDim: {
    color: 'rgba(255,255,255,0.3)',
    fontSize: 10,
    fontWeight: '500',
    fontStyle: 'italic',
  },
  divider: {
    height: 1,
    backgroundColor: 'rgba(255,255,255,0.08)',
    marginVertical: 8,
  },
  scanningRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingVertical: 4,
  },
  scanDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: C.yellow,
  },
  scanningText: {
    color: 'rgba(255,255,255,0.65)',
    fontSize: 13,
    fontWeight: '500',
  },
  actions: {
    flexDirection: 'row',
    gap: 10,
    marginTop: 4,
  },
  continueBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingVertical: 10,
    borderRadius: R.md,
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.15)',
  },
  continueBtnText: {
    color: 'rgba(255,255,255,0.7)',
    fontSize: 13,
    fontWeight: '600',
  },
  acceptBtn: {
    flex: 2,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingVertical: 10,
    borderRadius: R.md,
    backgroundColor: C.green,
  },
  acceptBtnText: {
    color: C.white,
    fontSize: 13,
    fontWeight: '700',
  },
  multiviewProgressContainer: {
    gap: 10,
    paddingVertical: 4,
  },
  progressBar: {
    height: 6,
    borderRadius: 3,
    backgroundColor: 'rgba(255,255,255,0.12)',
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: C.yellow,
    borderRadius: 3,
  },
  progressText: {
    color: 'rgba(255,255,255,0.6)',
    fontSize: 12,
    fontWeight: '600',
    textAlign: 'center',
  },
  multiviewResultContainer: {
    paddingVertical: 8,
    alignItems: 'center',
  },
  multiviewResultText: {
    color: 'rgba(255,255,255,0.75)',
    fontSize: 13,
    fontWeight: '500',
    textAlign: 'center',
    fontStyle: 'italic',
  },
});
