/**
 * DetectedBricksDrawer — top-right panel showing bricks accumulated during a
 * continuous-scan session.
 *
 * Two states:
 *   Collapsed: a chip with count + preview thumbnail of most-recent lock.
 *   Expanded:  full list with part #, name, color, confidence, lock status,
 *              and per-row remove action.
 *
 * Locks are indicated by a green checkmark. In-progress tracks show their
 * current best-guess with a spinner and live confidence %.
 */
import React, { useCallback } from 'react';
import {
  View, Text, Image, TouchableOpacity, ScrollView, StyleSheet,
  ActivityIndicator, Animated,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { C, R, S, T, shadow } from '@/constants/theme';

export type ContinuousBrickTrack = {
  /** Stable local ID for the track. */
  id: string;
  partNum: string;
  partName: string;
  colorName?: string;
  colorHex?: string;
  /** Best-guess thumbnail URL (or base64 data URI). */
  thumbnailUrl?: string;
  /** Consecutive scan cycles with this part_num as the top match for THIS bbox. */
  consecutiveAgreements: number;
  /** EMA of confidence across sightings. */
  fusedConfidence: number;
  /** When the track was first seen (ms epoch). */
  firstSeenAt: number;
  /** When it crossed the lock threshold. Null if not locked yet. */
  lockedAt: number | null;
};

interface Props {
  tracks: ContinuousBrickTrack[];
  expanded: boolean;
  onToggle: () => void;
  onRemove: (id: string) => void;
  onClear: () => void;
}

export const DetectedBricksDrawer: React.FC<Props> = ({
  tracks, expanded, onToggle, onRemove, onClear,
}) => {
  const lockedCount = tracks.filter(t => t.lockedAt !== null).length;
  const pendingCount = tracks.length - lockedCount;

  if (!expanded) {
    return (
      <TouchableOpacity style={styles.chip} onPress={onToggle} activeOpacity={0.85}>
        <Ionicons
          name={lockedCount > 0 ? 'cube' : 'scan-outline'}
          size={16}
          color={C.white}
          style={{ marginRight: 6 }}
        />
        <Text style={styles.chipText}>
          {lockedCount > 0
            ? `${lockedCount}${pendingCount ? ` + ${pendingCount}` : ''}`
            : pendingCount > 0 ? `seeing ${pendingCount}` : 'scan bricks'}
        </Text>
      </TouchableOpacity>
    );
  }

  return (
    <View style={[styles.drawer, shadow(2)]}>
      <View style={styles.drawerHeader}>
        <Text style={styles.drawerTitle}>
          {lockedCount} locked{pendingCount > 0 ? ` · ${pendingCount} pending` : ''}
        </Text>
        <View style={{ flexDirection: 'row', alignItems: 'center' }}>
          {tracks.length > 0 && (
            <TouchableOpacity onPress={onClear} hitSlop={10} style={styles.headerBtn}>
              <Text style={styles.clearText}>Clear</Text>
            </TouchableOpacity>
          )}
          <TouchableOpacity onPress={onToggle} hitSlop={10} style={styles.headerBtn}>
            <Ionicons name="close" size={20} color={C.text} />
          </TouchableOpacity>
        </View>
      </View>

      {tracks.length === 0 ? (
        <View style={styles.emptyState}>
          <Ionicons name="scan-outline" size={28} color={C.textMuted} />
          <Text style={styles.emptyText}>
            Point the camera at some bricks — they'll appear here as I recognise them.
          </Text>
        </View>
      ) : (
        <ScrollView style={styles.list} showsVerticalScrollIndicator={false}>
          {tracks
            .slice()
            .sort((a, b) => (b.lockedAt ?? b.firstSeenAt) - (a.lockedAt ?? a.firstSeenAt))
            .map(track => (
              <TrackRow
                key={track.id}
                track={track}
                onRemove={() => onRemove(track.id)}
              />
            ))}
        </ScrollView>
      )}
    </View>
  );
};

const TrackRow: React.FC<{
  track: ContinuousBrickTrack;
  onRemove: () => void;
}> = ({ track, onRemove }) => {
  const isLocked = track.lockedAt !== null;
  return (
    <View style={[styles.row, isLocked && styles.rowLocked]}>
      <View style={styles.thumbWrap}>
        {track.thumbnailUrl ? (
          <Image source={{ uri: track.thumbnailUrl }} style={styles.thumb} />
        ) : (
          <View style={[styles.thumb, styles.thumbPlaceholder]}>
            <Ionicons name="cube-outline" size={20} color={C.textMuted} />
          </View>
        )}
        {isLocked && (
          <View style={styles.lockBadge}>
            <Ionicons name="checkmark" size={10} color={C.white} />
          </View>
        )}
      </View>

      <View style={styles.rowInfo}>
        <Text style={styles.partNum} numberOfLines={1}>#{track.partNum}</Text>
        <Text style={styles.partName} numberOfLines={1}>
          {track.partName}
        </Text>
        {track.colorName && (
          <Text style={styles.colorText} numberOfLines={1}>{track.colorName}</Text>
        )}
      </View>

      <View style={styles.rowRight}>
        {isLocked ? (
          <Text style={styles.lockedPct}>{Math.round(track.fusedConfidence * 100)}%</Text>
        ) : (
          <View style={styles.pendingChip}>
            <ActivityIndicator size="small" color={C.red} />
            <Text style={styles.pendingPct}>
              {Math.round(track.fusedConfidence * 100)}%
            </Text>
          </View>
        )}
        <TouchableOpacity onPress={onRemove} hitSlop={8} style={styles.removeBtn}>
          <Ionicons name="close-circle" size={18} color={C.textMuted} />
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  chip: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.72)',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: R.full,
    minWidth: 90,
  },
  chipText: { color: C.white, fontWeight: '700', fontSize: 13 },

  drawer: {
    width: 320,
    maxHeight: 480,
    backgroundColor: C.white,
    borderRadius: R.lg,
    overflow: 'hidden',
  },
  drawerHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: S.md,
    paddingVertical: S.sm,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: C.border,
    backgroundColor: C.cardAlt,
  },
  drawerTitle: { fontSize: 14, fontWeight: '700', color: C.text },
  headerBtn: { paddingHorizontal: 6, paddingVertical: 2 },
  clearText: { color: C.red, fontSize: 13, fontWeight: '600' },

  list: { maxHeight: 420 },
  emptyState: {
    alignItems: 'center',
    padding: S.lg,
  },
  emptyText: {
    fontSize: 12,
    color: C.textMuted,
    textAlign: 'center',
    marginTop: S.sm,
    lineHeight: 18,
  },

  row: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: S.md,
    paddingVertical: S.sm,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: C.border,
  },
  rowLocked: { backgroundColor: C.greenLight },

  thumbWrap: { position: 'relative' },
  thumb: {
    width: 42, height: 42,
    borderRadius: R.sm,
    backgroundColor: C.bgDark,
  },
  thumbPlaceholder: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  lockBadge: {
    position: 'absolute',
    right: -4,
    bottom: -4,
    width: 18, height: 18,
    borderRadius: 9,
    backgroundColor: C.green,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    borderColor: C.white,
  },

  rowInfo: { flex: 1, marginLeft: S.sm, marginRight: S.xs },
  partNum: { fontSize: 13, fontWeight: '700', color: C.text },
  partName: { fontSize: 11, color: C.textSub, marginTop: 1 },
  colorText: { fontSize: 10, color: C.textMuted, marginTop: 1 },

  rowRight: { alignItems: 'flex-end' },
  lockedPct: {
    fontSize: 12,
    fontWeight: '700',
    color: C.green,
  },
  pendingChip: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  pendingPct: {
    fontSize: 11,
    color: C.red,
    marginLeft: 4,
    fontWeight: '600',
  },
  removeBtn: { marginTop: 4 },
});
