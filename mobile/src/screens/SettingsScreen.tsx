import React, { useState, useEffect } from 'react';
import {
  View, Text, TouchableOpacity, ScrollView, Alert, Switch,
  ActivityIndicator, StyleSheet, Platform, StatusBar,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Ionicons } from '@expo/vector-icons';
import { clearRecentScans, clearAllCache } from '@/utils/storageUtils';
import { apiClient } from '@/services/api';
import { C, R, S, shadow } from '@/constants/theme';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { ProfileStackParamList } from '@/types';

type Props = NativeStackScreenProps<ProfileStackParamList, 'SettingsScreen'>;

const SCAN_MODE_KEY = 'brickscan_default_scan_mode';
const LOCAL_ONLY_KEY = 'brickscan_local_only';

type ScanMode = 'photo' | 'video' | 'multi';

const SCAN_MODES: { value: ScanMode; label: string; icon: string; desc: string }[] = [
  { value: 'photo', label: 'Photo', icon: 'camera', desc: 'Single photo scan' },
  { value: 'video', label: 'Video', icon: 'videocam', desc: 'Multi-frame video scan' },
  { value: 'multi', label: 'Multi', icon: 'apps', desc: 'Detect multiple pieces' },
];

export const SettingsScreen: React.FC<Props> = ({ navigation }) => {
  const [defaultScanMode, setDefaultScanMode] = useState<ScanMode>('photo');
  const [localOnly, setLocalOnly] = useState(false);
  const [isClearingHistory, setIsClearingHistory] = useState(false);
  const [isClearingCache, setIsClearingCache] = useState(false);
  const [apiUrl] = useState(() => apiClient.getBaseUrl());
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  useEffect(() => {
    // Load stored settings
    Promise.all([
      AsyncStorage.getItem(SCAN_MODE_KEY),
      AsyncStorage.getItem(LOCAL_ONLY_KEY),
    ]).then(([mode, local]) => {
      if (mode === 'photo' || mode === 'video' || mode === 'multi') {
        setDefaultScanMode(mode);
      }
      setLocalOnly(local === 'true');
    });

    // Check API status
    apiClient.healthCheck().then((ok) => setApiStatus(ok ? 'online' : 'offline'));
  }, []);

  const handleSetScanMode = async (mode: ScanMode) => {
    setDefaultScanMode(mode);
    await AsyncStorage.setItem(SCAN_MODE_KEY, mode);
  };

  const handleToggleLocalOnly = async (value: boolean) => {
    setLocalOnly(value);
    await AsyncStorage.setItem(LOCAL_ONLY_KEY, value ? 'true' : 'false');
  };

  const handleClearHistory = () => {
    Alert.alert(
      'Clear Scan History',
      'This will delete all your recent scan history. Your inventory will not be affected.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear History',
          style: 'destructive',
          onPress: async () => {
            setIsClearingHistory(true);
            try {
              await clearRecentScans();
              Alert.alert('Done', 'Scan history cleared.');
            } catch {
              Alert.alert('Error', 'Failed to clear history.');
            } finally {
              setIsClearingHistory(false);
            }
          },
        },
      ],
    );
  };

  const handleClearCache = () => {
    Alert.alert(
      'Clear All Cache',
      'This will clear all cached data (scan history, build checks, wishlist). Your inventory on the server will not be affected.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear Cache',
          style: 'destructive',
          onPress: async () => {
            setIsClearingCache(true);
            try {
              await clearAllCache();
              Alert.alert('Done', 'All local cache cleared.');
            } catch {
              Alert.alert('Error', 'Failed to clear cache.');
            } finally {
              setIsClearingCache(false);
            }
          },
        },
      ],
    );
  };

  const statusColor = apiStatus === 'online' ? C.green : apiStatus === 'offline' ? C.red : '#F59E0B';
  const statusLabel = apiStatus === 'online' ? 'Online' : apiStatus === 'offline' ? 'Offline' : 'Checking…';

  return (
    <ScrollView style={styles.root} contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
      <StatusBar barStyle="dark-content" />

      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()}>
          <Ionicons name="arrow-back" size={22} color={C.text} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Settings</Text>
        <View style={{ width: 38 }} />
      </View>

      {/* API Status */}
      <Text style={styles.sectionLabel}>CONNECTION</Text>
      <View style={styles.card}>
        <View style={styles.settingRow}>
          <View style={[styles.settingIcon, { backgroundColor: statusColor + '18' }]}>
            <Ionicons name="wifi" size={20} color={statusColor} />
          </View>
          <View style={styles.settingInfo}>
            <Text style={styles.settingTitle}>Backend API</Text>
            <Text style={styles.settingDesc} numberOfLines={1}>{apiUrl}</Text>
          </View>
          <View style={[styles.statusPill, { backgroundColor: statusColor + '18' }]}>
            <View style={[styles.statusDot, { backgroundColor: statusColor }]} />
            <Text style={[styles.statusText, { color: statusColor }]}>{statusLabel}</Text>
          </View>
        </View>
      </View>

      {/* Default Scan Mode */}
      <Text style={styles.sectionLabel}>DEFAULT SCAN MODE</Text>
      <View style={styles.card}>
        {SCAN_MODES.map((mode, i) => (
          <TouchableOpacity
            key={mode.value}
            style={[
              styles.modeRow,
              i < SCAN_MODES.length - 1 && styles.rowBorder,
              defaultScanMode === mode.value && styles.modeRowSelected,
            ]}
            onPress={() => handleSetScanMode(mode.value)}
            activeOpacity={0.75}
          >
            <View style={[
              styles.settingIcon,
              { backgroundColor: defaultScanMode === mode.value ? C.red + '18' : C.bg },
            ]}>
              <Ionicons
                name={mode.icon as any}
                size={20}
                color={defaultScanMode === mode.value ? C.red : C.textMuted}
              />
            </View>
            <View style={styles.settingInfo}>
              <Text style={[
                styles.settingTitle,
                defaultScanMode === mode.value && { color: C.red },
              ]}>
                {mode.label}
              </Text>
              <Text style={styles.settingDesc}>{mode.desc}</Text>
            </View>
            {defaultScanMode === mode.value && (
              <Ionicons name="checkmark-circle" size={22} color={C.red} />
            )}
          </TouchableOpacity>
        ))}
      </View>

      {/* AI Settings */}
      <Text style={styles.sectionLabel}>AI</Text>
      <View style={styles.card}>
        <View style={styles.settingRow}>
          <View style={[styles.settingIcon, { backgroundColor: '#EEF2FF' }]}>
            <Ionicons name="hardware-chip-outline" size={20} color="#6366F1" />
          </View>
          <View style={styles.settingInfo}>
            <Text style={styles.settingTitle}>Local Model Only</Text>
            <Text style={styles.settingDesc}>Skip cloud AI fallback (faster, less accurate)</Text>
          </View>
          <Switch
            value={localOnly}
            onValueChange={handleToggleLocalOnly}
            trackColor={{ false: C.border, true: C.red + '80' }}
            thumbColor={localOnly ? C.red : C.textMuted}
          />
        </View>
      </View>

      {/* Data management */}
      <Text style={styles.sectionLabel}>DATA</Text>
      <View style={styles.card}>
        <TouchableOpacity
          style={[styles.settingRow, styles.rowBorder]}
          onPress={handleClearHistory}
          disabled={isClearingHistory}
          activeOpacity={0.75}
        >
          <View style={[styles.settingIcon, { backgroundColor: '#FEF3C7' }]}>
            <Ionicons name="time-outline" size={20} color="#D97706" />
          </View>
          <View style={styles.settingInfo}>
            <Text style={styles.settingTitle}>Clear Scan History</Text>
            <Text style={styles.settingDesc}>Remove all recent scan records</Text>
          </View>
          {isClearingHistory
            ? <ActivityIndicator size="small" color={C.textMuted} />
            : <Ionicons name="chevron-forward" size={18} color={C.textMuted} />
          }
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.settingRow}
          onPress={handleClearCache}
          disabled={isClearingCache}
          activeOpacity={0.75}
        >
          <View style={[styles.settingIcon, { backgroundColor: '#FFF0F0' }]}>
            <Ionicons name="trash-outline" size={20} color={C.red} />
          </View>
          <View style={styles.settingInfo}>
            <Text style={[styles.settingTitle, { color: C.red }]}>Clear All Cache</Text>
            <Text style={styles.settingDesc}>History, wishlists, build checks</Text>
          </View>
          {isClearingCache
            ? <ActivityIndicator size="small" color={C.red} />
            : <Ionicons name="chevron-forward" size={18} color={C.textMuted} />
          }
        </TouchableOpacity>
      </View>

      <Text style={styles.footer}>
        Settings are saved automatically and persist across sessions.
      </Text>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: C.bg },
  content: { paddingBottom: 48 },

  header: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    backgroundColor: C.white,
    paddingTop: Platform.OS === 'ios' ? 56 : 24,
    paddingBottom: S.md, paddingHorizontal: S.md,
    borderBottomWidth: 1, borderBottomColor: C.border,
    marginBottom: S.md,
  },
  backBtn: {
    width: 38, height: 38, borderRadius: R.full,
    backgroundColor: C.bg, alignItems: 'center', justifyContent: 'center',
  },
  headerTitle: { fontSize: 17, fontWeight: '700', color: C.text },

  sectionLabel: {
    fontSize: 11, fontWeight: '700', color: C.textMuted,
    letterSpacing: 0.8, paddingHorizontal: S.md,
    marginBottom: 6, marginTop: 6,
  },
  card: {
    backgroundColor: C.white, borderRadius: R.lg,
    marginHorizontal: S.md, marginBottom: S.md,
    overflow: 'hidden', ...shadow(1),
  },

  settingRow: { flexDirection: 'row', alignItems: 'center', padding: S.md, gap: S.md },
  modeRow: { flexDirection: 'row', alignItems: 'center', padding: S.md, gap: S.md },
  modeRowSelected: { backgroundColor: '#FFF5F5' },
  rowBorder: { borderBottomWidth: 1, borderBottomColor: C.border },
  settingIcon: { width: 40, height: 40, borderRadius: 10, alignItems: 'center', justifyContent: 'center' },
  settingInfo: { flex: 1 },
  settingTitle: { fontSize: 15, fontWeight: '600', color: C.text },
  settingDesc: { fontSize: 12, color: C.textMuted, marginTop: 1 },

  statusPill: {
    flexDirection: 'row', alignItems: 'center', gap: 5,
    paddingHorizontal: 10, paddingVertical: 4, borderRadius: R.full,
  },
  statusDot: { width: 6, height: 6, borderRadius: 3 },
  statusText: { fontSize: 12, fontWeight: '700' },

  footer: {
    textAlign: 'center', fontSize: 11, color: C.textMuted,
    paddingHorizontal: S.xl, paddingVertical: S.md, lineHeight: 16,
  },
});
