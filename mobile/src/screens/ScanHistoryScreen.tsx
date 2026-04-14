import React, { useCallback, useState } from 'react';
import {
  View, Text, FlatList, Image, TouchableOpacity,
  Alert, StyleSheet, Platform, StatusBar,
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import { loadRecentScans, clearRecentScans } from '@/utils/storageUtils';
import { C, R, S, shadow } from '@/constants/theme';
import { getFeedbackStats } from '@/services/feedbackApi';
import { FeedbackStatsBar } from '@/components/FeedbackStatsBar';

interface StoredScan {
  id: string;
  partNum: string;
  partName: string;
  colorId?: string;
  colorName?: string;
  colorHex?: string;
  confidence: number;
  imageUrl?: string;
  thumbnailUrl?: string;
  timestamp: number;
  scanMode?: 'photo' | 'video' | 'multi';
  source?: string;
  framesAnalyzed?: number;
  agreementScore?: number;
  allPredictions?: Array<{
    partNum: string;
    partName: string;
    colorId?: string;
    colorName?: string;
    colorHex?: string;
    confidence: number;
    imageUrl?: string;
    source?: string;
  }>;
}

interface ScanHistoryScreenProps {
  onSelectScan: (scan: StoredScan) => void;
}

function timeAgo(ts: number): string {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);
  const days = Math.floor(diff / 86400000);
  if (mins < 1) return 'Just now';
  if (mins < 60) return `${mins}m ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days < 7) return `${days}d ago`;
  return new Date(ts).toLocaleDateString();
}

function confidenceColor(c: number) {
  if (c >= 0.80) return C.green;
  if (c >= 0.50) return '#F59E0B';
  return C.red;
}

const ScanModeIcon: React.FC<{ mode?: string; source?: string }> = ({ mode, source }) => {
  if (mode === 'video' || source?.includes('video')) {
    return (
      <View style={styles.modeTag}>
        <Ionicons name="videocam" size={10} color="#6366F1" />
        <Text style={styles.modeTagText}>Video</Text>
      </View>
    );
  }
  if (mode === 'multi') {
    return (
      <View style={[styles.modeTag, styles.modeTagMulti]}>
        <Ionicons name="grid" size={10} color="#059669" />
        <Text style={[styles.modeTagText, { color: '#059669' }]}>Multi</Text>
      </View>
    );
  }
  return null;
};

const HistoryImage: React.FC<{ thumbnailUrl?: string; imageUrl?: string }> = ({ thumbnailUrl, imageUrl }) => {
  const [err, setErr] = useState(false);
  const [loading, setLoading] = useState(true);

  // Use thumbnailUrl first, fallback to imageUrl
  const url = thumbnailUrl || imageUrl;

  if (url && !err) {
    return (
      <Image
        source={{ uri: url }}
        style={styles.itemImg}
        resizeMode="cover"
        onError={() => setErr(true)}
        onLoad={() => setLoading(false)}
      />
    );
  }
  return (
    <View style={[styles.itemImg, styles.itemImgPlaceholder]}>
      <Text style={styles.placeholderEmoji}>🧱</Text>
    </View>
  );
};

export const ScanHistoryScreen: React.FC<ScanHistoryScreenProps> = ({ onSelectScan }) => {
  const [scans, setScans] = useState<StoredScan[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [corrections, setCorrections] = useState<number | null>(null);

  useFocusEffect(
    useCallback(() => {
      let active = true;
      setIsLoading(true);
      loadRecentScans()
        .then((data) => { if (active) setScans(data as StoredScan[]); })
        .catch(console.error)
        .finally(() => { if (active) setIsLoading(false); });
      return () => { active = false; };
    }, [])
  );

  const handleClear = () => {
    Alert.alert(
      'Clear History',
      'Remove all scan history? This cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear All', style: 'destructive',
          onPress: async () => {
            await clearRecentScans();
            setScans([]);
          },
        },
      ]
    );
  };

  // Group scans by date bucket
  const grouped = React.useMemo(() => {
    const today = new Date(); today.setHours(0, 0, 0, 0);
    const yesterday = new Date(today); yesterday.setDate(today.getDate() - 1);

    const buckets: { [key: string]: StoredScan[] } = {};
    scans.forEach(scan => {
      const d = new Date(scan.timestamp); d.setHours(0, 0, 0, 0);
      let label = d.getTime() === today.getTime()
        ? 'Today'
        : d.getTime() === yesterday.getTime()
        ? 'Yesterday'
        : d.toLocaleDateString('en-US', { month: 'long', day: 'numeric' });
      if (!buckets[label]) buckets[label] = [];
      buckets[label].push(scan);
    });

    return Object.entries(buckets);
  }, [scans]);

  // Flatten grouped data for FlatList with section headers
  const flatData = React.useMemo(() => {
    const result: Array<{ type: 'header'; label: string } | { type: 'item'; scan: StoredScan }> = [];
    grouped.forEach(([label, items]) => {
      result.push({ type: 'header', label });
      items.forEach(scan => result.push({ type: 'item', scan }));
    });
    return result;
  }, [grouped]);

  if (isLoading) {
    return (
      <View style={styles.root}>
        <StatusBar barStyle="dark-content" />
        <View style={styles.header}>
          <Text style={styles.headerTitle}>Scan History</Text>
        </View>
        <View style={styles.centered}>
          <Ionicons name="time-outline" size={48} color={C.border} />
          <Text style={styles.loadingText}>Loading history…</Text>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.root}>
      <StatusBar barStyle="dark-content" />

      {/* Header */}
      <View style={styles.header}>
        <View>
          <Text style={styles.headerTitle}>Scan History</Text>
          <Text style={styles.headerSub}>{scans.length} scans recorded</Text>
        </View>
        {scans.length > 0 && (
          <TouchableOpacity style={styles.clearBtn} onPress={handleClear} activeOpacity={0.7}>
            <Ionicons name="trash-outline" size={16} color={C.red} />
            <Text style={styles.clearBtnText}>Clear</Text>
          </TouchableOpacity>
        )}
      </View>

      {/* Stats row if we have scans */}
      {scans.length > 0 && (
        <View style={styles.statsRow}>
          <View style={styles.statItem}>
            <Ionicons name="camera-outline" size={16} color={C.textMuted} />
            <Text style={styles.statLabel}>
              {scans.filter(s => !s.scanMode || s.scanMode === 'photo').length} photos
            </Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.statItem}>
            <Ionicons name="videocam-outline" size={16} color="#6366F1" />
            <Text style={styles.statLabel}>
              {scans.filter(s => s.scanMode === 'video' || s.source?.includes('video')).length} video
            </Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.statItem}>
            <Ionicons name="grid-outline" size={16} color="#059669" />
            <Text style={styles.statLabel}>
              {scans.filter(s => s.scanMode === 'multi').length} multi
            </Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.statItem}>
            <Ionicons name="checkmark-circle-outline" size={16} color={C.green} />
            <Text style={styles.statLabel}>
              {scans.filter(s => s.confidence >= 0.8).length} high conf.
            </Text>
          </View>
        </View>
      )}

      <FeedbackStatsBar corrections={corrections} />
      <FlatList
        style={{ flex: 1 }}
        data={flatData}
        keyExtractor={(item, i) =>
          item.type === 'header' ? `header-${item.label}` : `scan-${item.scan.id}-${i}`
        }
        contentContainerStyle={[
          styles.listContent,
          flatData.length === 0 && { flex: 1 },
        ]}
        ListEmptyComponent={
          <View style={styles.emptyState}>
            <View style={styles.emptyIcon}>
              <Ionicons name="scan-outline" size={40} color={C.textMuted} />
            </View>
            <Text style={styles.emptyTitle}>No Scan History</Text>
            <Text style={styles.emptySub}>
              Scan LEGO pieces using Photo, Video, or Multi mode. Your history appears here.
            </Text>
          </View>
        }
        renderItem={({ item }) => {
          if (item.type === 'header') {
            return (
              <View style={styles.sectionHeader}>
                <Text style={styles.sectionHeaderText}>{item.label}</Text>
              </View>
            );
          }

          const { scan } = item;
          const confPct = Math.round(scan.confidence * 100);
          const confColor = confidenceColor(scan.confidence);

          return (
            <TouchableOpacity
              style={styles.scanCard}
              onPress={() => onSelectScan(scan)}
              activeOpacity={0.78}
            >
              <HistoryImage thumbnailUrl={scan.thumbnailUrl} imageUrl={scan.imageUrl} />

              <View style={styles.scanInfo}>
                <View style={styles.scanNameRow}>
                  <Text style={styles.scanName} numberOfLines={1}>{scan.partName}</Text>
                  <ScanModeIcon mode={scan.scanMode} source={scan.source} />
                </View>

                <Text style={styles.scanNum}>#{scan.partNum}</Text>

                <View style={styles.scanMetaRow}>
                  {scan.colorName ? (
                    <View style={styles.colorChip}>
                      <View style={[styles.colorDot, { backgroundColor: scan.colorHex || '#ccc' }]} />
                      <Text style={styles.colorText} numberOfLines={1}>{scan.colorName}</Text>
                    </View>
                  ) : null}
                  <Text style={styles.timeText}>{timeAgo(scan.timestamp)}</Text>
                </View>
              </View>

              <View style={[styles.confBadge, { backgroundColor: confColor + '18' }]}>
                <Text style={[styles.confText, { color: confColor }]}>{confPct}%</Text>
              </View>
            </TouchableOpacity>
          );
        }}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: C.bg },

  header: {
    backgroundColor: C.white,
    paddingTop: Platform.OS === 'ios' ? 56 : 24,
    paddingBottom: S.md,
    paddingHorizontal: S.md,
    borderBottomWidth: 1, borderBottomColor: C.border,
    flexDirection: 'row', alignItems: 'flex-end', justifyContent: 'space-between',
  },
  headerTitle: { fontSize: 26, fontWeight: '800', color: C.text, letterSpacing: -0.5 },
  headerSub: { fontSize: 12, color: C.textMuted, marginTop: 2 },
  clearBtn: {
    flexDirection: 'row', alignItems: 'center', gap: 4,
    backgroundColor: '#FFF0F0', paddingHorizontal: 12, paddingVertical: 7,
    borderRadius: R.full, borderWidth: 1, borderColor: '#FFCCCC',
  },
  clearBtnText: { fontSize: 13, fontWeight: '700', color: C.red },

  statsRow: {
    flexDirection: 'row', alignItems: 'center',
    backgroundColor: C.white, borderBottomWidth: 1, borderBottomColor: C.border,
    paddingHorizontal: S.md, paddingVertical: 10,
  },
  statItem: { flex: 1, flexDirection: 'row', alignItems: 'center', gap: 5, justifyContent: 'center' },
  statLabel: { fontSize: 11, fontWeight: '600', color: C.textSub },
  statDivider: { width: 1, height: 16, backgroundColor: C.border },

  listContent: { padding: S.md, gap: 6, paddingBottom: 32 },

  sectionHeader: {
    paddingHorizontal: 4, paddingTop: 8, paddingBottom: 6,
  },
  sectionHeaderText: {
    fontSize: 11, fontWeight: '700', color: C.textMuted, letterSpacing: 0.8, textTransform: 'uppercase',
  },

  scanCard: {
    flexDirection: 'row', alignItems: 'center',
    backgroundColor: C.white, borderRadius: R.lg,
    padding: 12, gap: 12,
    ...shadow(1),
  },
  itemImg: {
    width: 60, height: 60, borderRadius: 8,
    backgroundColor: C.bg,
  },
  itemImgPlaceholder: {
    alignItems: 'center', justifyContent: 'center',
  },
  placeholderEmoji: {
    fontSize: 28,
  },

  scanInfo: { flex: 1, gap: 3 },
  scanNameRow: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  scanName: { fontSize: 14, fontWeight: '700', color: C.text, flex: 1 },
  scanNum: {
    fontSize: 11, color: C.textMuted,
    fontFamily: Platform.OS === 'ios' ? 'Courier New' : 'monospace',
  },

  modeTag: {
    flexDirection: 'row', alignItems: 'center', gap: 3,
    backgroundColor: '#EEF2FF', paddingHorizontal: 7, paddingVertical: 2,
    borderRadius: R.full,
  },
  modeTagMulti: { backgroundColor: '#ECFDF5' },
  modeTagText: { fontSize: 9, fontWeight: '700', color: '#6366F1' },

  scanMetaRow: { flexDirection: 'row', alignItems: 'center', gap: 8, flexWrap: 'wrap' },
  colorChip: {
    flexDirection: 'row', alignItems: 'center', gap: 4,
  },
  colorDot: { width: 9, height: 9, borderRadius: 5, borderWidth: 1, borderColor: 'rgba(0,0,0,0.1)' },
  colorText: { fontSize: 11, color: C.textMuted, maxWidth: 80 },
  timeText: { fontSize: 11, color: C.textMuted },

  confBadge: {
    paddingHorizontal: 10, paddingVertical: 5, borderRadius: R.full,
  },
  confText: { fontSize: 12, fontWeight: '700' },

  centered: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 12 },
  loadingText: { fontSize: 14, color: C.textMuted },

  emptyState: { flex: 1, alignItems: 'center', justifyContent: 'center', paddingHorizontal: S.xl },
  emptyIcon: {
    width: 80, height: 80, borderRadius: 20, backgroundColor: C.bgDark,
    alignItems: 'center', justifyContent: 'center', marginBottom: S.md,
  },
  emptyTitle: { fontSize: 18, fontWeight: '700', color: C.text, marginBottom: S.sm, textAlign: 'center' },
  emptySub: { fontSize: 14, color: C.textSub, textAlign: 'center', lineHeight: 20 },
});
