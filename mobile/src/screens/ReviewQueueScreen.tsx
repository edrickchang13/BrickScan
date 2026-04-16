/**
 * ReviewQueueScreen — the Mattheij bootstrap-loop UI.
 *
 * Fetches the server's "pending-review" queue of low-confidence scans and
 * lets the user rapidly triage each one: "yep, that's correct" or "nope, fix
 * it". Every correction becomes training data for the next retrain cycle.
 *
 * The screen is deliberately minimal — a card stack with big image, big
 * predicted label, and two big buttons — so reviewers can blitz a dozen
 * scans in under a minute.
 */
import React, { useCallback, useEffect, useState } from 'react';
import {
  View, Text, Image, TouchableOpacity, FlatList, StyleSheet,
  ActivityIndicator, RefreshControl, Alert,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { C, R, S, shadow } from '@/constants/theme';
import {
  getPendingReview, submitFeedback,
  type PendingReviewItem,
} from '@/services/feedbackApi';

export const ReviewQueueScreen: React.FC = () => {
  const [items, setItems] = useState<PendingReviewItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [processedCount, setProcessedCount] = useState(0);

  const load = useCallback(async (isRefresh = false) => {
    isRefresh ? setRefreshing(true) : setLoading(true);
    try {
      const { items: next } = await getPendingReview(20, 0.65);
      setItems(next);
    } catch (e: any) {
      Alert.alert('Failed to load review queue', e?.message ?? 'Unknown error');
    } finally {
      isRefresh ? setRefreshing(false) : setLoading(false);
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  const confirmCorrect = useCallback(async (item: PendingReviewItem) => {
    try {
      await submitFeedback({
        scanId: item.scanId,
        predictedPartNum: item.predictedPartNum,
        correctPartNum: item.predictedPartNum,  // user confirmed prediction was right
        confidence: item.confidence,
        source: item.source ?? 'unknown',
      });
      setItems(prev => prev.filter(x => x.scanId !== item.scanId));
      setProcessedCount(c => c + 1);
    } catch (e: any) {
      Alert.alert('Could not save confirmation', e?.message ?? 'Unknown error');
    }
  }, []);

  const flagWrong = useCallback((item: PendingReviewItem) => {
    Alert.prompt(
      'Correct part number?',
      `The model predicted #${item.predictedPartNum}. What's the actual part number?`,
      async (input) => {
        const correct = (input ?? '').trim();
        if (!correct) return;
        try {
          await submitFeedback({
            scanId: item.scanId,
            predictedPartNum: item.predictedPartNum,
            correctPartNum: correct,
            confidence: item.confidence,
            source: item.source ?? 'unknown',
          });
          setItems(prev => prev.filter(x => x.scanId !== item.scanId));
          setProcessedCount(c => c + 1);
        } catch (e: any) {
          Alert.alert('Could not save correction', e?.message ?? 'Unknown error');
        }
      },
      'plain-text',
    );
  }, []);

  if (loading && items.length === 0) {
    return (
      <View style={styles.loading}>
        <ActivityIndicator color={C.red} size="large" />
        <Text style={styles.loadingText}>Loading scans that need review…</Text>
      </View>
    );
  }

  if (items.length === 0 && !loading) {
    return (
      <View style={styles.empty}>
        <Ionicons name="checkmark-done-circle" size={64} color={C.green} />
        <Text style={styles.emptyTitle}>All caught up</Text>
        <Text style={styles.emptySub}>
          No low-confidence scans waiting for review.{'\n'}
          Your model quality thanks you.
        </Text>
        <TouchableOpacity onPress={() => load(true)} style={styles.refreshBtn}>
          <Ionicons name="refresh" size={16} color={C.white} />
          <Text style={styles.refreshText}>Refresh</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <FlatList
      contentContainerStyle={styles.listContent}
      data={items}
      keyExtractor={item => item.scanId}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={() => load(true)} tintColor={C.red} />
      }
      ListHeaderComponent={
        <View style={styles.header}>
          <Text style={styles.headerTitle}>Review Queue</Text>
          <Text style={styles.headerSub}>
            {items.length} scans where the model wasn't sure.{'\n'}
            Confirming or correcting each one directly improves the next model.
          </Text>
          {processedCount > 0 && (
            <View style={styles.progressPill}>
              <Ionicons name="trending-up" size={14} color={C.green} />
              <Text style={styles.progressText}>+{processedCount} reviewed this session</Text>
            </View>
          )}
        </View>
      }
      renderItem={({ item }) => (
        <View style={[styles.card, shadow(1)]}>
          {item.thumbnailUrl ? (
            <Image source={{ uri: item.thumbnailUrl }} style={styles.thumb} />
          ) : (
            <View style={[styles.thumb, styles.thumbPlaceholder]}>
              <Ionicons name="image-outline" size={40} color={C.textMuted} />
            </View>
          )}
          <View style={styles.info}>
            <Text style={styles.predictedNum}>#{item.predictedPartNum}</Text>
            {item.predictedPartName && (
              <Text style={styles.predictedName} numberOfLines={2}>{item.predictedPartName}</Text>
            )}
            <Text style={styles.confidence}>
              {Math.round(item.confidence * 100)}% confident
              {item.source ? ` • ${item.source}` : ''}
            </Text>
          </View>
          <View style={styles.actions}>
            <TouchableOpacity
              style={[styles.actionBtn, styles.confirmBtn]}
              onPress={() => confirmCorrect(item)}
              activeOpacity={0.8}
            >
              <Ionicons name="checkmark" size={18} color={C.white} />
              <Text style={styles.actionText}>Correct</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.actionBtn, styles.flagBtn]}
              onPress={() => flagWrong(item)}
              activeOpacity={0.8}
            >
              <Ionicons name="close" size={18} color={C.white} />
              <Text style={styles.actionText}>Fix</Text>
            </TouchableOpacity>
          </View>
        </View>
      )}
    />
  );
};

const styles = StyleSheet.create({
  loading: {
    flex: 1, alignItems: 'center', justifyContent: 'center',
    backgroundColor: C.bg,
  },
  loadingText: { marginTop: 12, color: C.textMuted },
  empty: {
    flex: 1, alignItems: 'center', justifyContent: 'center',
    padding: S.xl, backgroundColor: C.bg,
  },
  emptyTitle: { fontSize: 20, fontWeight: '700', color: C.text, marginTop: S.md },
  emptySub: { fontSize: 14, color: C.textMuted, textAlign: 'center', marginTop: S.sm },
  refreshBtn: {
    marginTop: S.lg,
    flexDirection: 'row',
    backgroundColor: C.red,
    paddingVertical: 10, paddingHorizontal: 16,
    borderRadius: R.sm,
    alignItems: 'center',
  },
  refreshText: { color: C.white, marginLeft: 6, fontWeight: '600' },
  listContent: { padding: S.md, paddingBottom: S.xl },
  header: { marginBottom: S.lg },
  headerTitle: { fontSize: 22, fontWeight: '700', color: C.text },
  headerSub: { fontSize: 13, color: C.textMuted, marginTop: S.xs, lineHeight: 18 },
  progressPill: {
    marginTop: S.sm,
    flexDirection: 'row',
    alignSelf: 'flex-start',
    backgroundColor: C.greenLight,
    paddingVertical: 4, paddingHorizontal: 10,
    borderRadius: R.sm,
    alignItems: 'center',
  },
  progressText: { color: C.green, fontWeight: '600', marginLeft: 4, fontSize: 12 },
  card: {
    flexDirection: 'row',
    backgroundColor: C.white,
    borderRadius: R.md,
    padding: S.md,
    marginBottom: S.md,
  },
  thumb: {
    width: 80, height: 80, borderRadius: R.sm,
    backgroundColor: '#f0f0f0',
  },
  thumbPlaceholder: { alignItems: 'center', justifyContent: 'center' },
  info: { flex: 1, marginLeft: S.md, justifyContent: 'center' },
  predictedNum: { fontSize: 15, fontWeight: '700', color: C.text },
  predictedName: { fontSize: 13, color: C.text, marginTop: 2 },
  confidence: { fontSize: 11, color: C.textMuted, marginTop: 4 },
  actions: { justifyContent: 'center' },
  actionBtn: {
    flexDirection: 'row',
    paddingVertical: 8, paddingHorizontal: 12,
    borderRadius: R.sm,
    alignItems: 'center',
    marginBottom: 6,
  },
  confirmBtn: { backgroundColor: C.green },
  flagBtn: { backgroundColor: C.red },
  actionText: { color: C.white, fontWeight: '600', marginLeft: 4, fontSize: 12 },
});
