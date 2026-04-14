import React, { useState } from 'react';
import {
  View, Text, TouchableOpacity, ScrollView, Alert,
  ActivityIndicator, StyleSheet, Platform, StatusBar, Share,
} from 'react-native';
import { useQuery } from '@tanstack/react-query';
import { Ionicons } from '@expo/vector-icons';
import { useAuthStore } from '@/store/authStore';
import { useInventoryStore } from '@/store/inventoryStore';
import { apiClient } from '@/services/api';
import Constants from 'expo-constants';
import { C, R, S, shadow } from '@/constants/theme';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { ProfileStackParamList } from '@/types';

type Props = NativeStackScreenProps<ProfileStackParamList, 'ProfileScreen'>;

export const ProfileScreen: React.FC<Props> = ({ navigation }) => {
  const user = useAuthStore((s) => s.user);
  const logout = useAuthStore((s) => s.logout);
  const clearInventory = useInventoryStore((s) => s.clearInventory);
  const [isExporting, setIsExporting] = useState(false);

  const { data: stats, isLoading } = useQuery({
    queryKey: ['profile-stats'],
    queryFn: async () => {
      const inv = await apiClient.getInventory();
      return {
        totalParts: inv.reduce((s: number, i: any) => s + i.quantity, 0),
        uniqueParts: inv.length,
      };
    },
    staleTime: 5 * 60 * 1000,
  });

  const handleExport = async () => {
    setIsExporting(true);
    try {
      const inv = await apiClient.getInventory();
      const exportDate = new Date().toISOString().split('T')[0];

      // Header row — comprehensive columns
      const header = [
        'Part Number',
        'Part Name',
        'Color Name',
        'Color ID',
        'Color Hex',
        'Quantity',
        'Image URL',
        'Date Added',
      ].join(',');

      const rows = inv.map((item: any) => {
        const escape = (v: string) => `"${String(v || '').replace(/"/g, '""')}"`;
        return [
          escape(item.partNum),
          escape(item.partName),
          escape(item.colorName),
          escape(item.colorId),
          escape(item.colorHex),
          item.quantity,
          escape(item.imageUrl || ''),
          escape(item.createdAt ? item.createdAt.split('T')[0] : exportDate),
        ].join(',');
      });

      // Summary footer
      const totalParts = inv.reduce((s: number, i: any) => s + i.quantity, 0);
      rows.push('');
      rows.push(`"Exported","${exportDate}","Total Pieces","${totalParts}","Unique Parts","${inv.length}"`);

      const csv = [header, ...rows].join('\n');
      await Share.share({
        message: csv,
        title: `BrickScan Inventory — ${exportDate}`,
      });
    } catch (e: any) {
      Alert.alert('Error', e?.message || 'Export failed');
    } finally {
      setIsExporting(false);
    }
  };

  const handleLogout = () => {
    Alert.alert('Sign Out', 'Are you sure?', [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Sign Out', style: 'destructive', onPress: async () => {
        try { await logout(); clearInventory(); } catch {}
      }},
    ]);
  };

  const emailInitial = (user?.email ?? 'U')[0].toUpperCase();
  const appVersion = Constants.expoConfig?.version ?? '1.0.0';

  return (
    <ScrollView style={styles.root} contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
      <StatusBar barStyle="dark-content" />

      {/* Hero card */}
      <View style={styles.heroCard}>
        <View style={styles.avatar}>
          <Text style={styles.avatarText}>{emailInitial}</Text>
        </View>
        <View style={styles.heroInfo}>
          <Text style={styles.heroEmail} numberOfLines={1}>{user?.email ?? 'User'}</Text>
          <View style={styles.heroBadge}>
            <Ionicons name="shield-checkmark" size={11} color={C.green} />
            <Text style={styles.heroBadgeText}>Verified</Text>
          </View>
        </View>
      </View>

      {/* Stats */}
      <View style={styles.statsRow}>
        <View style={[styles.statCard, styles.statCardRed]}>
          <Text style={styles.statValue}>
            {isLoading ? '—' : (stats?.totalParts ?? 0).toLocaleString()}
          </Text>
          <Text style={styles.statLabel}>Total Pieces</Text>
        </View>
        <View style={[styles.statCard, styles.statCardYellow]}>
          <Text style={[styles.statValue, { color: C.black }]}>
            {isLoading ? '—' : (stats?.uniqueParts ?? 0).toLocaleString()}
          </Text>
          <Text style={[styles.statLabel, { color: 'rgba(0,0,0,0.55)' }]}>Unique Parts</Text>
        </View>
      </View>

      {/* Actions */}
      <Text style={styles.sectionLabel}>ACTIONS</Text>
      <View style={styles.actionCard}>
        <TouchableOpacity
          style={[styles.actionRow, { borderBottomWidth: 1, borderBottomColor: C.border }]}
          onPress={() => navigation.navigate('SettingsScreen')}
        >
          <View style={[styles.actionIcon, { backgroundColor: '#EEF2FF' }]}>
            <Ionicons name="settings-outline" size={20} color="#6366F1" />
          </View>
          <View style={styles.actionInfo}>
            <Text style={styles.actionTitle}>Settings</Text>
            <Text style={styles.actionSub}>Scan mode, cache, API config</Text>
          </View>
          <Ionicons name="chevron-forward" size={18} color={C.textMuted} />
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.actionRow, { borderBottomWidth: 1, borderBottomColor: C.border }]}
          onPress={handleExport}
          disabled={isExporting}
        >
          <View style={[styles.actionIcon, { backgroundColor: '#E6F9F0' }]}>
            <Ionicons name="download-outline" size={20} color={C.green} />
          </View>
          <View style={styles.actionInfo}>
            <Text style={styles.actionTitle}>Export Inventory</Text>
            <Text style={styles.actionSub}>Share as CSV with all fields</Text>
          </View>
          {isExporting
            ? <ActivityIndicator size="small" color={C.green} />
            : <Ionicons name="chevron-forward" size={18} color={C.textMuted} />
          }
        </TouchableOpacity>

        <TouchableOpacity style={styles.actionRow} onPress={handleLogout}>
          <View style={[styles.actionIcon, { backgroundColor: '#FFF0F0' }]}>
            <Ionicons name="log-out-outline" size={20} color={C.red} />
          </View>
          <View style={styles.actionInfo}>
            <Text style={[styles.actionTitle, { color: C.red }]}>Sign Out</Text>
            <Text style={styles.actionSub}>Log out of your account</Text>
          </View>
          <Ionicons name="chevron-forward" size={18} color={C.textMuted} />
        </TouchableOpacity>
      </View>

      {/* About */}
      <Text style={styles.sectionLabel}>ABOUT</Text>
      <View style={styles.actionCard}>
        {[
          { label: 'Version', value: `v${appVersion}` },
          { label: 'Platform', value: Platform.OS === 'ios' ? 'iOS' : 'Android' },
          { label: 'Data', value: 'Rebrickable' },
        ].map((row, i, arr) => (
          <View
            key={row.label}
            style={[styles.infoRow, i < arr.length - 1 && { borderBottomWidth: 1, borderBottomColor: C.border }]}
          >
            <Text style={styles.infoLabel}>{row.label}</Text>
            <Text style={styles.infoValue}>{row.value}</Text>
          </View>
        ))}
      </View>

      <Text style={styles.footer}>
        BrickScan — AI-powered LEGO inventory{'\n'}v{appVersion}
      </Text>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: C.bg },
  content: { paddingBottom: 48 },

  // Hero
  heroCard: {
    flexDirection: 'row', alignItems: 'center', gap: S.md,
    backgroundColor: C.white,
    paddingTop: Platform.OS === 'ios' ? 56 : 24,
    paddingBottom: S.lg, paddingHorizontal: S.md,
    borderBottomWidth: 1, borderBottomColor: C.border,
  },
  avatar: {
    width: 64, height: 64, borderRadius: 32,
    backgroundColor: C.red, alignItems: 'center', justifyContent: 'center',
    ...shadow(2),
  },
  avatarText: { fontSize: 26, fontWeight: '900', color: C.white },
  heroInfo: { flex: 1 },
  heroEmail: { fontSize: 16, fontWeight: '700', color: C.text, marginBottom: 4 },
  heroBadge: {
    flexDirection: 'row', alignItems: 'center', gap: 4,
    backgroundColor: C.greenLight, paddingHorizontal: 8, paddingVertical: 3,
    borderRadius: R.full, alignSelf: 'flex-start',
  },
  heroBadgeText: { fontSize: 11, fontWeight: '600', color: C.green },

  // Stats
  statsRow: { flexDirection: 'row', gap: 12, padding: S.md },
  statCard: {
    flex: 1, borderRadius: R.lg, padding: S.md, ...shadow(1),
  },
  statCardRed: { backgroundColor: C.red },
  statCardYellow: { backgroundColor: C.yellow },
  statValue: { fontSize: 32, fontWeight: '900', color: C.white, letterSpacing: -1 },
  statLabel: { fontSize: 12, fontWeight: '600', color: 'rgba(255,255,255,0.75)', marginTop: 2 },

  // Sections
  sectionLabel: {
    fontSize: 11, fontWeight: '700', color: C.textMuted,
    letterSpacing: 0.8, paddingHorizontal: S.md, marginBottom: 6, marginTop: 6,
  },
  actionCard: {
    backgroundColor: C.white, borderRadius: R.lg,
    marginHorizontal: S.md, marginBottom: S.md, ...shadow(1), overflow: 'hidden',
  },
  actionRow: { flexDirection: 'row', alignItems: 'center', padding: S.md, gap: S.md },
  actionIcon: { width: 40, height: 40, borderRadius: 10, alignItems: 'center', justifyContent: 'center' },
  actionInfo: { flex: 1 },
  actionTitle: { fontSize: 15, fontWeight: '600', color: C.text },
  actionSub: { fontSize: 12, color: C.textMuted, marginTop: 1 },

  // Info rows
  infoRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', padding: S.md },
  infoLabel: { fontSize: 14, color: C.textSub },
  infoValue: { fontSize: 14, fontWeight: '600', color: C.text },

  footer: {
    textAlign: 'center', fontSize: 11, color: C.textMuted,
    lineHeight: 16, paddingHorizontal: S.xl, paddingVertical: S.md,
  },
});
