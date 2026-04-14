import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
  View, Text, TextInput, FlatList, TouchableOpacity,
  RefreshControl, ScrollView, StyleSheet, Platform, StatusBar,
  ActivityIndicator, Animated, Dimensions,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/services/api';
import { SetCard } from '@/components/SetCard';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { SetsStackParamList, SetCompletionResult } from '@/types';
import { C, R, S, shadow } from '@/constants/theme';

type Props = NativeStackScreenProps<SetsStackParamList, 'SetsScreen'>;

const THEMES = [
  'All', 'Star Wars', 'City', 'Technic', 'Creator', 'Friends',
  'Architecture', 'Ninjago', 'Harry Potter', 'Minecraft', 'Disney',
];

export const SetsScreen: React.FC<Props> = ({ navigation }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTheme, setSelectedTheme] = useState('All');
  const [activeTab, setActiveTab] = useState<'browse' | 'completion'>('browse');

  // Poll backend download status until sets are ready
  const { data: setsStatus } = useQuery({
    queryKey: ['sets-status'],
    queryFn: () => apiClient.getSetsStatus(),
    refetchInterval: (data: any) => (data?.sets_ready ? false : 3000),
    staleTime: 0,
  });

  const setsReady = setsStatus?.sets_ready ?? false;
  const isDownloading = !setsReady;

  const { data: results = [], isLoading, error, refetch } = useQuery({
    queryKey: ['sets', searchQuery, selectedTheme, setsReady],
    queryFn: () =>
      apiClient.searchSets(
        searchQuery,
        selectedTheme === 'All' ? undefined : selectedTheme,
      ),
    staleTime: 5 * 60 * 1000,
    enabled: setsReady && activeTab === 'browse',
  });

  const { data: completionResults = [], isLoading: completionLoading, error: completionError, refetch: refetchCompletion } = useQuery({
    queryKey: ['sets-completion-scan'],
    queryFn: () => apiClient.scanInventoryForSets(),
    staleTime: 5 * 60 * 1000,
    enabled: activeTab === 'completion',
  });

  const handleTheme = useCallback((theme: string) => {
    setSelectedTheme(theme);
  }, []);

  // ─── Header rendered above the list items ────────────────────────────────────
  const ListHeader = (
    <View>
      <StatusBar barStyle="dark-content" />

      {/* Page title */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Sets</Text>
        {setsStatus?.sets_count ? (
          <Text style={styles.headerSub}>
            {setsStatus.sets_count.toLocaleString()} sets available
          </Text>
        ) : results.length > 0 ? (
          <Text style={styles.headerSub}>{results.length} results</Text>
        ) : null}
      </View>

      {/* Downloading banner */}
      {isDownloading && (
        <View style={styles.downloadBanner}>
          <ActivityIndicator size="small" color={C.white} style={{ marginRight: 8 }} />
          <Text style={styles.downloadBannerText}>Downloading LEGO set database…</Text>
        </View>
      )}

      {/* Search row */}
      <View style={styles.searchRow}>
        <View style={styles.searchWrap}>
          <Ionicons name="search" size={16} color={C.textMuted} style={{ marginRight: 8 }} />
          <TextInput
            style={styles.searchInput}
            placeholder="Name, number, or keyword…"
            placeholderTextColor={C.textMuted}
            value={searchQuery}
            onChangeText={setSearchQuery}
            returnKeyType="search"
            onSubmitEditing={() => refetch()}
          />
          {searchQuery.length > 0 && (
            <TouchableOpacity onPress={() => setSearchQuery('')} hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}>
              <Ionicons name="close-circle" size={16} color={C.textMuted} />
            </TouchableOpacity>
          )}
        </View>
        <TouchableOpacity style={styles.searchBtn} onPress={() => refetch()} activeOpacity={0.8}>
          <Ionicons name="search" size={18} color={C.white} />
        </TouchableOpacity>
      </View>

      {/* Theme filter pills */}
      <View style={styles.pillRow}>
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.pillContent}
          bounces={false}
        >
          {THEMES.map((theme) => {
            const active = selectedTheme === theme;
            return (
              <TouchableOpacity
                key={theme}
                style={[styles.pill, active && styles.pillActive]}
                onPress={() => handleTheme(theme)}
                activeOpacity={0.75}
              >
                <Text style={[styles.pillText, active && styles.pillTextActive]}>
                  {theme}
                </Text>
              </TouchableOpacity>
            );
          })}
        </ScrollView>
      </View>
    </View>
  );

  // ─── Empty / loading state ────────────────────────────────────────────────────
  const EmptyState = (
    <View style={styles.emptyState}>
      <View style={styles.emptyIcon}>
        {isDownloading
          ? <ActivityIndicator size="large" color={C.red} />
          : error
          ? <Ionicons name="alert-circle-outline" size={40} color={C.red} />
          : <Ionicons name="layers-outline" size={40} color={C.textMuted} />}
      </View>
      <Text style={styles.emptyTitle}>
        {isDownloading
          ? 'Loading set database…'
          : error
          ? 'Failed to load sets'
          : isLoading
          ? 'Fetching sets…'
          : searchQuery
          ? 'No sets found'
          : 'Browse LEGO Sets'}
      </Text>
      <Text style={styles.emptySub}>
        {isDownloading
          ? 'Downloading the full Rebrickable catalog. This only happens once.'
          : error
          ? (error as any)?.message || 'Please try again later'
          : isLoading
          ? 'Just a moment…'
          : searchQuery
          ? 'Try different keywords or browse by theme'
          : 'Loading recent sets…'}
      </Text>
    </View>
  );

  // ─── Tab header ───────────────────────────────────────────────────────────────
  const TabHeader = (
    <View style={styles.tabContainer}>
      <TouchableOpacity
        style={[styles.tab, activeTab === 'browse' && styles.tabActive]}
        onPress={() => setActiveTab('browse')}
      >
        <Text style={[styles.tabText, activeTab === 'browse' && styles.tabTextActive]}>
          Browse Sets
        </Text>
      </TouchableOpacity>
      <TouchableOpacity
        style={[styles.tab, activeTab === 'completion' && styles.tabActive]}
        onPress={() => setActiveTab('completion')}
      >
        <Text style={[styles.tabText, activeTab === 'completion' && styles.tabTextActive]}>
          My Completion
        </Text>
      </TouchableOpacity>
    </View>
  );

  // ─── Render ───────────────────────────────────────────────────────────────────
  if (activeTab === 'completion') {
    return (
      <FlatList
        style={styles.root}
        data={completionResults}
        keyExtractor={(item) => item.set_num}
        contentContainerStyle={styles.listContent}
        ListHeaderComponent={
          <View>
            <StatusBar barStyle="dark-content" />
            <View style={styles.header}>
              <Text style={styles.headerTitle}>Sets</Text>
              <Text style={styles.headerSub}>Your completion progress</Text>
            </View>
            {TabHeader}
          </View>
        }
        ListEmptyComponent={
          <View style={styles.emptyState}>
            <View style={styles.emptyIcon}>
              {completionLoading
                ? <ActivityIndicator size="large" color={C.red} />
                : completionError
                ? <Ionicons name="alert-circle-outline" size={40} color={C.red} />
                : <Ionicons name="checkmark-circle-outline" size={40} color={C.textMuted} />}
            </View>
            <Text style={styles.emptyTitle}>
              {completionLoading
                ? 'Loading sets…'
                : completionError
                ? 'Failed to load'
                : 'No sets at 30%+ completion'}
            </Text>
            <Text style={styles.emptySub}>
              {completionLoading
                ? 'Scanning your inventory…'
                : completionError
                ? (completionError as any)?.message || 'Please try again'
                : 'Scan more pieces to unlock sets!'}
            </Text>
          </View>
        }
        renderItem={({ item }) => (
          <SetCompletionCard
            item={item}
            onPress={() => navigation.navigate('SetDetailScreen', { setNum: item.set_num })}
          />
        )}
        refreshControl={
          <RefreshControl refreshing={completionLoading} onRefresh={() => refetchCompletion()} tintColor={C.red} />
        }
        scrollIndicatorInsets={{ right: 1 }}
      />
    );
  }

  return (
    <FlatList
      style={styles.root}
      data={results}
      keyExtractor={(item) => item.setNum}
      numColumns={2}
      columnWrapperStyle={styles.columnWrap}
      contentContainerStyle={styles.listContent}
      ListHeaderComponent={
        <View>
          {ListHeader}
          {TabHeader}
        </View>
      }
      ListEmptyComponent={EmptyState}
      renderItem={({ item }) => (
        <SetCard
          setNum={item.setNum}
          name={item.name}
          year={item.year}
          numParts={item.numParts}
          imgUrl={item.imageUrl}
          theme={item.theme}
          onPress={() => navigation.navigate('SetDetailScreen', { setNum: item.setNum })}
        />
      )}
      refreshControl={
        <RefreshControl refreshing={isLoading} onRefresh={() => refetch()} tintColor={C.red} />
      }
      keyboardShouldPersistTaps="handled"
      keyboardDismissMode="on-drag"
    />
  );
};

// ─── Set Completion Card Component ────────────────────────────────────────────
interface SetCompletionCardProps {
  item: SetCompletionResult;
  onPress: () => void;
}

const SetCompletionCard: React.FC<SetCompletionCardProps> = ({ item, onPress }) => {
  const progressAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.spring(progressAnim, {
      toValue: item.completion_pct,
      useNativeDriver: false,
      friction: 7,
      tension: 40,
    }).start();
  }, [item.completion_pct, progressAnim]);

  const progressWidth = progressAnim.interpolate({
    inputRange: [0, 100],
    outputRange: ['0%', '100%'],
  });

  const progressColor = item.completion_pct >= 90 ? C.green : item.completion_pct >= 60 ? C.yellow : C.orange;

  return (
    <TouchableOpacity style={styles.completionCard} onPress={onPress} activeOpacity={0.7}>
      <View style={styles.completionHeader}>
        <View style={styles.completionTitleWrap}>
          <Text style={styles.completionTitle} numberOfLines={2}>{item.set_name}</Text>
          <Text style={styles.completionSetNum}>{item.set_num}</Text>
        </View>
        <Text style={styles.completionPercent}>{item.completion_pct}%</Text>
      </View>

      <View style={styles.progressBarBg}>
        <Animated.View
          style={[
            styles.progressBarFill,
            {
              width: progressWidth,
              backgroundColor: progressColor,
            },
          ]}
        />
      </View>

      <Text style={styles.completionSubtext}>
        {item.have_parts} / {item.total_parts} parts · {item.missing.length} missing
      </Text>
    </TouchableOpacity>
  );
};

// ─── Styles ───────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: C.bg },

  // ── Header ──────────────────────────────────────────────────────────────────
  header: {
    backgroundColor: C.white,
    paddingTop: Platform.OS === 'ios' ? 56 : 24,
    paddingHorizontal: S.md,
    paddingBottom: S.md,
    borderBottomWidth: 1,
    borderBottomColor: C.border,
  },
  headerTitle: {
    fontSize: 28, fontWeight: '800', color: C.text, letterSpacing: -0.5,
  },
  headerSub: { fontSize: 13, color: C.textMuted, marginTop: 2 },

  // ── Download banner ──────────────────────────────────────────────────────────
  downloadBanner: {
    flexDirection: 'row', alignItems: 'center',
    backgroundColor: C.red, paddingHorizontal: S.md, paddingVertical: 10,
  },
  downloadBannerText: { fontSize: 13, fontWeight: '600', color: C.white },

  // ── Search ───────────────────────────────────────────────────────────────────
  searchRow: {
    flexDirection: 'row', alignItems: 'center', gap: 8,
    paddingHorizontal: S.md, paddingVertical: 10,
    backgroundColor: C.white,
    borderBottomWidth: 1, borderBottomColor: C.border,
  },
  searchWrap: {
    flex: 1, flexDirection: 'row', alignItems: 'center',
    backgroundColor: C.bg, borderRadius: R.full,
    paddingHorizontal: 12, paddingVertical: 9,
  },
  searchInput: { flex: 1, fontSize: 14, color: C.text, padding: 0, margin: 0 },
  searchBtn: {
    width: 42, height: 42, borderRadius: R.full,
    backgroundColor: C.red, alignItems: 'center', justifyContent: 'center',
  },

  // ── Pills ────────────────────────────────────────────────────────────────────
  pillRow: {
    backgroundColor: C.white,
    borderBottomWidth: 1, borderBottomColor: C.border,
  },
  pillContent: {
    paddingHorizontal: S.md,
    paddingVertical: 10,
    flexDirection: 'row',
    gap: 8,
    alignItems: 'center',
  },
  pill: {
    paddingHorizontal: 16,
    paddingVertical: 7,
    borderRadius: R.full,
    borderWidth: 1.5,
    borderColor: C.border,
    backgroundColor: C.white,
  },
  pillActive: {
    backgroundColor: C.red,
    borderColor: C.red,
  },
  pillText: {
    fontSize: 13,
    fontWeight: '600',
    color: C.textSub,
    includeFontPadding: false,
  },
  pillTextActive: { color: C.white },

  // ── List ─────────────────────────────────────────────────────────────────────
  columnWrap: { gap: 10, paddingHorizontal: 12 },
  listContent: { paddingTop: 12, paddingBottom: 36, gap: 10 },

  // ── Empty state ───────────────────────────────────────────────────────────────
  emptyState: {
    alignItems: 'center', paddingTop: 80, paddingHorizontal: S.xl,
  },
  emptyIcon: {
    width: 80, height: 80, borderRadius: 20, backgroundColor: C.cardAlt,
    alignItems: 'center', justifyContent: 'center', marginBottom: S.md,
  },
  emptyTitle: {
    fontSize: 17, fontWeight: '700', color: C.text, marginBottom: 6, textAlign: 'center',
  },
  emptySub: {
    fontSize: 14, color: C.textSub, textAlign: 'center', lineHeight: 20,
  },

  // ── Tabs ────────────────────────────────────────────────────────────────────
  tabContainer: {
    flexDirection: 'row', gap: 8,
    paddingHorizontal: S.md, paddingVertical: 10,
    backgroundColor: C.white,
    borderBottomWidth: 1, borderBottomColor: C.border,
  },
  tab: {
    flex: 1,
    paddingVertical: 10,
    paddingHorizontal: S.md,
    alignItems: 'center',
    borderBottomWidth: 2,
    borderBottomColor: C.transparent,
  },
  tabActive: {
    borderBottomColor: C.red,
  },
  tabText: {
    fontSize: 14, fontWeight: '600', color: C.textMuted,
  },
  tabTextActive: {
    color: C.red,
  },

  // ── Completion Card ────────────────────────────────────────────────────────────
  completionCard: {
    marginHorizontal: S.md, marginBottom: S.md,
    paddingHorizontal: S.md, paddingVertical: S.md,
    backgroundColor: C.card,
    borderRadius: R.lg,
    ...shadow(1),
  },
  completionHeader: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start',
    marginBottom: S.sm,
  },
  completionTitleWrap: {
    flex: 1, marginRight: S.md,
  },
  completionTitle: {
    fontSize: 15, fontWeight: '700', color: C.text, marginBottom: 2,
  },
  completionSetNum: {
    fontSize: 12, fontWeight: '500', color: C.textMuted,
  },
  completionPercent: {
    fontSize: 18, fontWeight: '800', color: C.red,
  },
  progressBarBg: {
    height: 8, backgroundColor: C.bgDark, borderRadius: R.full, overflow: 'hidden',
    marginBottom: S.sm,
  },
  progressBarFill: {
    height: 8, borderRadius: R.full,
  },
  completionSubtext: {
    fontSize: 12, color: C.textMuted, fontWeight: '500',
  },
});
