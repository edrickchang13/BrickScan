/**
 * FeedbackStatsScreen — model-improvement dashboard.
 *
 * Reads GET /api/local-inventory/feedback/stats and renders:
 *   • Top-1 + Top-3 rolling 30-day accuracy
 *   • Weekly accuracy trend (text rows — can swap for a chart library later)
 *   • Per-source accuracy breakdown (Brickognize / Gemini / k-NN / TTA / …)
 *   • Top 10 confusion pairs
 *   • "Retrain on Spark" button (triggers POST /scan/admin/trigger-retrain)
 *
 * Accessed from the header of ScanScreen.
 */

import React, { useCallback, useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  RefreshControl,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { ScanStackParamList } from '@/types';
import { C, R, S, shadow, T } from '@/constants/theme';
import { getFeedbackStats, type FeedbackStats } from '@/services/feedbackApi';
import { apiClient } from '@/services/api';

type Props = NativeStackScreenProps<ScanStackParamList, 'FeedbackStatsScreen'>;

const SOURCE_LABELS: Record<string, string> = {
  brickognize: 'Brickognize',
  'brickognize+gemini': 'Brickognize + Gemini',
  gemini: 'Gemini',
  contrastive_knn: 'Contrastive k-NN',
  distilled_model: 'Distilled Model',
  tta_local: 'TTA (local)',
  local_model: 'Legacy ONNX',
  unknown: 'Unknown source',
};

export const FeedbackStatsScreen: React.FC<Props> = ({ navigation }) => {
  const [stats, setStats] = useState<FeedbackStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [retraining, setRetraining] = useState(false);

  const load = useCallback(async () => {
    setError(null);
    try {
      const data = await getFeedbackStats();
      setStats(data);
    } catch (e: any) {
      setError(e?.message || 'Failed to load stats');
    }
  }, []);

  useEffect(() => {
    (async () => {
      setLoading(true);
      await load();
      setLoading(false);
    })();
  }, [load]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await load();
    setRefreshing(false);
  }, [load]);

  const triggerRetrain = useCallback(async () => {
    Alert.alert(
      'Retrain on Spark?',
      'This will export your feedback corrections to the DGX Spark training box and fine-tune the contrastive model. Takes ~30 minutes. Works only when Spark is online.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Retrain',
          onPress: async () => {
            setRetraining(true);
            try {
              const resp = await apiClient.healthCheck(); // soft check reachable
              if (!resp) throw new Error('Backend unreachable');
              // @ts-ignore — endpoint exists on backend but not in apiClient interface
              await (apiClient as any).client?.post('/api/scan/admin/trigger-retrain');
              Alert.alert('Triggered', 'Retrain job queued. Watch Spark logs for progress.');
            } catch (e: any) {
              Alert.alert('Retrain failed', e?.message || 'Unknown error — is Spark online?');
            } finally {
              setRetraining(false);
            }
          },
        },
      ],
    );
  }, []);

  if (loading) {
    return (
      <View style={styles.center}>
        <StatusBar barStyle="dark-content" />
        <ActivityIndicator size="large" color={C.red} />
      </View>
    );
  }

  return (
    <View style={styles.root}>
      <StatusBar barStyle="dark-content" />
      <View style={styles.header}>
        <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()}>
          <Ionicons name="arrow-back" size={22} color={C.text} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Model Accuracy</Text>
        <TouchableOpacity
          style={styles.backBtn}
          onPress={() => navigation.navigate('ReviewQueueScreen')}
          accessibilityLabel="Review uncertain scans"
        >
          <Ionicons name="checkmark-done-circle-outline" size={22} color={C.red} />
        </TouchableOpacity>
      </View>

      <ScrollView
        contentContainerStyle={styles.content}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
      >
        {error && (
          <View style={styles.errorBanner}>
            <Ionicons name="alert-circle" size={16} color={C.red} />
            <Text style={styles.errorText}>{error}</Text>
          </View>
        )}

        {stats && (
          <>
            {/* ── Top-line accuracy ─────────────────────────────────────────── */}
            <View style={styles.heroCard}>
              <Text style={styles.heroLabel}>Rolling 30-day accuracy</Text>
              <View style={styles.heroNumbers}>
                <View style={styles.heroStat}>
                  <Text style={styles.heroPct}>{pct(stats.top1Accuracy)}</Text>
                  <Text style={styles.heroSub}>top-1</Text>
                </View>
                <View style={styles.heroDivider} />
                <View style={styles.heroStat}>
                  <Text style={styles.heroPct}>{pct(stats.top3Accuracy)}</Text>
                  <Text style={styles.heroSub}>top-3</Text>
                </View>
              </View>
              <Text style={styles.heroSample}>
                Based on {stats.totalCorrections + stats.agreementCount} scans with feedback
              </Text>
            </View>

            {/* ── Weekly trend ───────────────────────────────────────────────── */}
            <Text style={styles.sectionLabel}>WEEKLY TREND</Text>
            <View style={styles.card}>
              {stats.accuracyTrend.length === 0
                ? <Text style={styles.emptyText}>No snapshots yet. Run the weekly snapshot endpoint.</Text>
                : stats.accuracyTrend.map((p, i) => (
                    <View key={p.weekEnding} style={[styles.trendRow, i > 0 && styles.trendRowBorder]}>
                      <Text style={styles.trendDate}>{p.weekEnding}</Text>
                      <View style={styles.trendBars}>
                        <TrendBar label="top-1" value={p.top1Accuracy} color={C.green} />
                        <TrendBar label="top-3" value={p.top3Accuracy} color="#6366F1" />
                      </View>
                      <Text style={styles.trendN}>n={p.sampleSize}</Text>
                    </View>
                  ))
              }
            </View>

            {/* ── Per-source accuracy ────────────────────────────────────────── */}
            <Text style={styles.sectionLabel}>ACCURACY BY MODEL</Text>
            <View style={styles.card}>
              {stats.bySource.length === 0
                ? <Text style={styles.emptyText}>Not enough data yet.</Text>
                : stats.bySource.map((s, i) => (
                    <View key={s.source} style={[styles.sourceRow, i > 0 && styles.sourceRowBorder]}>
                      <View style={{ flex: 1 }}>
                        <Text style={styles.sourceName}>
                          {SOURCE_LABELS[s.source] ?? s.source}
                        </Text>
                        <Text style={styles.sourceMeta}>
                          {s.correct} / {s.count} correct
                        </Text>
                      </View>
                      <Text style={[
                        styles.sourceAcc,
                        s.accuracy >= 0.8 ? { color: C.green }
                          : s.accuracy >= 0.5 ? { color: '#D97706' }
                          : { color: C.red },
                      ]}>
                        {pct(s.accuracy)}
                      </Text>
                    </View>
                  ))
              }
            </View>

            {/* ── Confusion pairs ────────────────────────────────────────────── */}
            <Text style={styles.sectionLabel}>MOST-CONFUSED PAIRS</Text>
            <View style={styles.card}>
              {stats.topConfusedPairs.length === 0
                ? <Text style={styles.emptyText}>No confusion data yet — keep scanning!</Text>
                : stats.topConfusedPairs.slice(0, 10).map((p, i) => (
                    <View key={`${p.predictedPartNum}-${p.correctPartNum}`} style={[styles.pairRow, i > 0 && styles.pairRowBorder]}>
                      <Text style={styles.pairNum}>#{p.predictedPartNum}</Text>
                      <Ionicons name="arrow-forward" size={14} color={C.textMuted} />
                      <Text style={styles.pairNum}>#{p.correctPartNum}</Text>
                      <View style={{ flex: 1 }} />
                      <View style={styles.pairCountPill}>
                        <Text style={styles.pairCountText}>{p.count}×</Text>
                      </View>
                    </View>
                  ))
              }
            </View>

            {/* ── Training trigger ────────────────────────────────────────────── */}
            <Text style={styles.sectionLabel}>TRAINING</Text>
            <View style={styles.card}>
              <Text style={styles.trainText}>
                {stats.pendingTraining} correction{stats.pendingTraining === 1 ? '' : 's'} ready to feed into the next retrain.
              </Text>
              <TouchableOpacity
                style={[styles.retrainBtn, retraining && styles.retrainBtnDisabled]}
                onPress={triggerRetrain}
                disabled={retraining}
                activeOpacity={0.8}
              >
                {retraining
                  ? <ActivityIndicator color={C.white} />
                  : (
                    <>
                      <Ionicons name="rocket-outline" size={16} color={C.white} />
                      <Text style={styles.retrainText}>Retrain on Spark</Text>
                    </>
                  )
                }
              </TouchableOpacity>
              <Text style={styles.trainHint}>
                Requires DGX Spark to be online. If offline, corrections stay queued until it returns.
              </Text>
            </View>
          </>
        )}
      </ScrollView>
    </View>
  );
};

// ─────────────────────────────────────────────────────────────────────────────

const pct = (v: number): string => `${Math.round((v || 0) * 100)}%`;

const TrendBar: React.FC<{ label: string; value: number; color: string }> = ({ label, value, color }) => {
  const pctW = `${Math.max(2, Math.min(100, Math.round(value * 100)))}%` as const;
  return (
    <View style={styles.trendBarWrap}>
      <Text style={styles.trendBarLabel}>{label}</Text>
      <View style={styles.trendBarTrack}>
        <View style={[styles.trendBarFill, { width: pctW, backgroundColor: color }]} />
      </View>
      <Text style={styles.trendBarVal}>{pct(value)}</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: C.bg },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: C.bg },

  header: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingHorizontal: S.md, paddingVertical: S.sm,
    backgroundColor: C.white, borderBottomWidth: 1, borderBottomColor: C.border,
  },
  backBtn: { width: 38, height: 38, alignItems: 'center', justifyContent: 'center' },
  headerTitle: { ...T.h3 },

  content: { padding: S.md, gap: S.md, paddingBottom: 40 },

  errorBanner: {
    flexDirection: 'row', alignItems: 'center', gap: 8,
    backgroundColor: '#FEF2F2', borderWidth: 1, borderColor: '#FECACA',
    padding: S.md, borderRadius: R.md,
  },
  errorText: { color: C.red, fontSize: 13, flex: 1 },

  // Hero
  heroCard: {
    backgroundColor: C.white, borderRadius: R.lg, padding: S.lg, ...shadow(1),
    alignItems: 'center', gap: S.sm,
  },
  heroLabel: { ...T.caption, textTransform: 'uppercase' as const, color: C.textMuted },
  heroNumbers: { flexDirection: 'row', alignItems: 'center', gap: S.lg },
  heroStat: { alignItems: 'center', minWidth: 100 },
  heroPct: { fontSize: 42, fontWeight: '800', color: C.text, letterSpacing: -1 },
  heroSub: { ...T.label, color: C.textSub },
  heroDivider: { width: 1, height: 50, backgroundColor: C.border },
  heroSample: { ...T.caption, color: C.textMuted },

  // Section
  sectionLabel: { ...T.caption, paddingHorizontal: 4 },
  card: { backgroundColor: C.white, borderRadius: R.lg, ...shadow(1) },
  emptyText: { ...T.bodySmall, textAlign: 'center', padding: S.lg, color: C.textMuted },

  // Trend
  trendRow: { padding: S.md, gap: 6 },
  trendRowBorder: { borderTopWidth: 1, borderTopColor: C.border },
  trendDate: { ...T.label },
  trendBars: { gap: 4 },
  trendBarWrap: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  trendBarLabel: { ...T.caption, width: 42 },
  trendBarTrack: {
    flex: 1, height: 6, backgroundColor: C.bgDark, borderRadius: 3, overflow: 'hidden',
  },
  trendBarFill: { height: '100%', borderRadius: 3 },
  trendBarVal: { ...T.caption, fontVariant: ['tabular-nums'], width: 36, textAlign: 'right' },
  trendN: { ...T.caption, color: C.textMuted },

  // Source
  sourceRow: { flexDirection: 'row', alignItems: 'center', padding: S.md, gap: S.sm },
  sourceRowBorder: { borderTopWidth: 1, borderTopColor: C.border },
  sourceName: { ...T.h4, color: C.text },
  sourceMeta: { ...T.caption, color: C.textSub, marginTop: 2 },
  sourceAcc: { fontSize: 20, fontWeight: '800', fontVariant: ['tabular-nums'] },

  // Pair
  pairRow: { flexDirection: 'row', alignItems: 'center', padding: S.md, gap: 8 },
  pairRowBorder: { borderTopWidth: 1, borderTopColor: C.border },
  pairNum: { ...T.bodySmall, fontWeight: '600', color: C.text },
  pairCountPill: {
    backgroundColor: '#FEF3C7', paddingHorizontal: 8, paddingVertical: 2,
    borderRadius: 999,
  },
  pairCountText: { fontSize: 11, fontWeight: '700', color: '#B45309' },

  // Train
  trainText: { ...T.body, padding: S.md, paddingBottom: 0 },
  trainHint: { ...T.caption, paddingHorizontal: S.md, paddingBottom: S.md, color: C.textMuted },
  retrainBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8,
    backgroundColor: C.red, paddingVertical: 14, marginHorizontal: S.md, marginTop: S.sm,
    borderRadius: R.md,
  },
  retrainBtnDisabled: { backgroundColor: '#FCA5A5' },
  retrainText: { ...T.h4, color: C.white },
});
