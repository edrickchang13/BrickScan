/**
 * CollectionAnalyticsScreen — Brickify-style breakdown of the user's local
 * inventory by part category, LEGO theme, and decade.
 *
 * Reads from GET /api/inventory/analytics (computed entirely server-side
 * from the Rebrickable bulk catalogue + the user's local SQLite inventory).
 *
 * Three sections, each with a horizontal-bar visualisation:
 *   - "Buildable Sets" (top 5 from /buildable-sets, conf ≥ 80%)
 *   - "By Part Category"
 *   - "By Theme"
 *   - "By Decade" (timeline)
 *
 * No charts library — we draw bars with plain Views to keep the bundle small
 * and avoid native deps. Looks clean enough at the typical inventory scale.
 */
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  View, Text, ScrollView, RefreshControl, StyleSheet, TouchableOpacity,
  ActivityIndicator, Platform, StatusBar,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { ScanStackParamList } from '@/types';
import { C, R, S, T, shadow } from '@/constants/theme';
import {
  apiClient,
  type InventoryAnalyticsResponse,
  type BuildableSet,
} from '@/services/api';

type Props = NativeStackScreenProps<ScanStackParamList, 'CollectionAnalyticsScreen'>;

export const CollectionAnalyticsScreen: React.FC<Props> = ({ navigation }) => {
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [data, setData] = useState<InventoryAnalyticsResponse | null>(null);
  const [buildable, setBuildable] = useState<BuildableSet[]>([]);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (isRefresh = false) => {
    isRefresh ? setRefreshing(true) : setLoading(true);
    setError(null);
    try {
      const [analytics, builds] = await Promise.all([
        apiClient.getInventoryAnalytics(),
        apiClient.getBuildableSets({
          colorMatch: 'loose',
          minCompletion: 0.80,
          limit: 5,
        }),
      ]);
      setData(analytics);
      setBuildable(builds.sets);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? e?.message ?? 'Failed to load analytics');
    } finally {
      isRefresh ? setRefreshing(false) : setLoading(false);
    }
  }, []);

  useEffect(() => { load(false); }, [load]);

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator color={C.red} size="large" />
        <Text style={styles.loadingText}>Crunching collection numbers…</Text>
      </View>
    );
  }
  if (error || !data) {
    return (
      <View style={styles.center}>
        <Ionicons name="alert-circle-outline" size={32} color={C.red} />
        <Text style={styles.errorTitle}>Couldn't load analytics</Text>
        <Text style={styles.errorBody}>{error ?? 'Unknown error'}</Text>
        <TouchableOpacity style={styles.retryBtn} onPress={() => load(false)}>
          <Text style={styles.retryText}>Retry</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <ScrollView
      style={styles.root}
      contentContainerStyle={styles.contentContainer}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={() => load(true)} tintColor={C.red} />}
    >
      <StatusBar barStyle="dark-content" />

      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} hitSlop={10} style={styles.backBtn}>
          <Ionicons name="chevron-back" size={24} color={C.text} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Collection Insights</Text>
        <View style={{ width: 32 }} />
      </View>

      {/* Hero: total bricks + distinct */}
      <View style={[styles.hero, shadow(1)]}>
        <View style={styles.heroBlock}>
          <Text style={styles.heroNumber}>{data.total_quantity.toLocaleString()}</Text>
          <Text style={styles.heroLabel}>bricks total</Text>
        </View>
        <View style={styles.heroDivider} />
        <View style={styles.heroBlock}>
          <Text style={styles.heroNumber}>{data.distinct_parts.toLocaleString()}</Text>
          <Text style={styles.heroLabel}>distinct parts</Text>
        </View>
      </View>

      {!data.catalog_loaded && (
        <View style={styles.warnBanner}>
          <Ionicons name="warning-outline" size={14} color={C.white} />
          <Text style={styles.warnText}>
            Rebrickable catalogue not loaded on the backend — analytics may be empty.
          </Text>
        </View>
      )}

      {/* Buildable Sets */}
      <SectionHeader
        icon="construct-outline"
        title="Sets you can build"
        subtitle={buildable.length > 0
          ? `${buildable.length} sets at 80%+ completion`
          : 'No sets at 80%+ yet — keep scanning'}
      />
      {buildable.length === 0 ? (
        <Text style={styles.emptyText}>
          As you build up inventory, sets you can complete will appear here.
        </Text>
      ) : (
        <View style={[styles.card, shadow(1)]}>
          {buildable.map((s, i) => (
            <View
              key={s.set_num}
              style={[
                styles.buildableRow,
                i < buildable.length - 1 && styles.rowDivider,
              ]}
            >
              <View style={styles.buildableInfo}>
                <Text style={styles.buildableName} numberOfLines={1}>
                  {s.name}
                </Text>
                <Text style={styles.buildableMeta}>
                  #{s.set_num}{s.year ? ` · ${s.year}` : ''} · {s.matched_pairs}/{s.total_pairs} parts
                </Text>
              </View>
              <View style={styles.completionPill}>
                <Text style={styles.completionPct}>
                  {Math.round(s.distinct_completion * 100)}%
                </Text>
              </View>
            </View>
          ))}
        </View>
      )}

      {/* By Part Category */}
      <SectionHeader
        icon="cube-outline"
        title="By Part Category"
        subtitle={`${data.by_part_category.length} categories`}
      />
      <BarList
        items={data.by_part_category.slice(0, 12).map(c => ({
          label: c.cat_name,
          value: c.total_quantity,
          sub: `${c.distinct_parts} distinct`,
        }))}
        accent={C.red}
      />

      {/* By Theme */}
      <SectionHeader
        icon="planet-outline"
        title="By Theme"
        subtitle={`${data.by_theme.length} themes`}
      />
      <BarList
        items={data.by_theme.slice(0, 12).map(t => ({
          label: t.theme_name,
          value: t.total_quantity,
          sub: `${t.distinct_parts} parts`,
        }))}
        accent={C.blue ?? '#2563EB'}
      />

      {/* By Decade */}
      <SectionHeader
        icon="time-outline"
        title="By Decade"
        subtitle="Imputed from set appearances"
      />
      <BarList
        items={data.by_year_decade.map(d => ({
          label: `${d.decade}s`,
          value: d.total_quantity,
          sub: `${d.distinct_parts} parts`,
        }))}
        accent={C.green ?? '#16A34A'}
      />

      <View style={{ height: 40 }} />
    </ScrollView>
  );
};

// ── Sub-components ─────────────────────────────────────────────────────────

const SectionHeader: React.FC<{ icon: keyof typeof Ionicons.glyphMap; title: string; subtitle?: string }> = ({
  icon, title, subtitle,
}) => (
  <View style={styles.sectionHeader}>
    <View style={styles.sectionTitleRow}>
      <Ionicons name={icon} size={18} color={C.text} />
      <Text style={styles.sectionTitle}>{title}</Text>
    </View>
    {subtitle ? <Text style={styles.sectionSubtitle}>{subtitle}</Text> : null}
  </View>
);

const BarList: React.FC<{
  items: { label: string; value: number; sub: string }[];
  accent: string;
}> = ({ items, accent }) => {
  const max = useMemo(
    () => (items.length ? items.reduce((m, it) => Math.max(m, it.value), 0) : 1),
    [items],
  );
  if (items.length === 0) {
    return (
      <View style={[styles.card, shadow(1), { padding: S.lg }]}>
        <Text style={styles.emptyText}>No data — start scanning bricks to populate this.</Text>
      </View>
    );
  }
  return (
    <View style={[styles.card, shadow(1)]}>
      {items.map((it, i) => {
        const pct = max > 0 ? (it.value / max) : 0;
        return (
          <View key={it.label + i} style={styles.barRow}>
            <View style={styles.barTop}>
              <Text style={styles.barLabel} numberOfLines={1}>{it.label}</Text>
              <Text style={styles.barValue}>{it.value.toLocaleString()}</Text>
            </View>
            <View style={styles.barTrack}>
              <View style={[styles.barFill, { width: `${Math.max(2, pct * 100)}%`, backgroundColor: accent }]} />
            </View>
            <Text style={styles.barSub}>{it.sub}</Text>
          </View>
        );
      })}
    </View>
  );
};

// ── Styles ─────────────────────────────────────────────────────────────────

const TOP_PAD = Platform.OS === 'ios' ? 52 : 32;

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: C.bg },
  contentContainer: { paddingBottom: 40 },
  center: {
    flex: 1, alignItems: 'center', justifyContent: 'center', padding: S.xl,
    backgroundColor: C.bg,
  },
  loadingText: { marginTop: S.md, color: C.textMuted },
  errorTitle: { fontSize: 18, fontWeight: '700', color: C.text, marginTop: S.md },
  errorBody: { color: C.textSub, textAlign: 'center', marginTop: S.sm },
  retryBtn: {
    marginTop: S.md, paddingVertical: 10, paddingHorizontal: 20,
    backgroundColor: C.red, borderRadius: R.sm,
  },
  retryText: { color: C.white, fontWeight: '700' },

  header: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingTop: TOP_PAD, paddingHorizontal: S.md, paddingBottom: S.md,
    backgroundColor: C.bg,
  },
  backBtn: { padding: 4 },
  headerTitle: { fontSize: 18, fontWeight: '700', color: C.text },

  hero: {
    flexDirection: 'row',
    backgroundColor: C.white,
    marginHorizontal: S.md,
    borderRadius: R.lg,
    padding: S.lg,
    marginBottom: S.md,
  },
  heroBlock: { flex: 1, alignItems: 'center' },
  heroDivider: { width: StyleSheet.hairlineWidth, backgroundColor: C.border, marginHorizontal: S.md },
  heroNumber: { fontSize: 28, fontWeight: '800', color: C.text },
  heroLabel: { fontSize: 12, color: C.textMuted, marginTop: 2 },

  warnBanner: {
    flexDirection: 'row', alignItems: 'center',
    backgroundColor: 'rgba(234, 88, 12, 0.92)',
    marginHorizontal: S.md, marginBottom: S.sm,
    paddingVertical: 8, paddingHorizontal: 12,
    borderRadius: R.sm,
  },
  warnText: { color: C.white, fontSize: 11, marginLeft: 6, flex: 1 },

  sectionHeader: { paddingHorizontal: S.md, paddingTop: S.md, paddingBottom: S.xs },
  sectionTitleRow: { flexDirection: 'row', alignItems: 'center' },
  sectionTitle: { fontSize: 15, fontWeight: '700', color: C.text, marginLeft: 6 },
  sectionSubtitle: { fontSize: 11, color: C.textMuted, marginTop: 2 },

  card: {
    backgroundColor: C.white,
    marginHorizontal: S.md,
    borderRadius: R.lg,
    overflow: 'hidden',
  },
  emptyText: { color: C.textMuted, fontSize: 12, textAlign: 'center', padding: S.md },

  buildableRow: {
    flexDirection: 'row', alignItems: 'center',
    paddingVertical: S.sm, paddingHorizontal: S.md,
  },
  rowDivider: { borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: C.border },
  buildableInfo: { flex: 1, marginRight: S.sm },
  buildableName: { fontSize: 14, fontWeight: '600', color: C.text },
  buildableMeta: { fontSize: 11, color: C.textMuted, marginTop: 1 },
  completionPill: {
    backgroundColor: C.greenLight,
    paddingHorizontal: 10, paddingVertical: 4,
    borderRadius: R.full,
  },
  completionPct: { color: C.green, fontWeight: '700', fontSize: 12 },

  barRow: { paddingVertical: S.sm, paddingHorizontal: S.md, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: C.border },
  barTop: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 },
  barLabel: { flex: 1, fontSize: 13, color: C.text, fontWeight: '500' },
  barValue: { fontSize: 13, color: C.text, fontWeight: '700' },
  barTrack: {
    height: 6, borderRadius: 3,
    backgroundColor: C.bgDark,
    overflow: 'hidden',
  },
  barFill: { height: 6, borderRadius: 3 },
  barSub: { fontSize: 10, color: C.textMuted, marginTop: 3 },
});
