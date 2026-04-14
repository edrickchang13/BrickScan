import React, { useEffect, useState } from 'react';
import {
  View, Text, Image, TouchableOpacity, FlatList,
  Alert, ActivityIndicator, StyleSheet, Platform,
} from 'react-native';
import { useQuery } from '@tanstack/react-query';
import { Ionicons } from '@expo/vector-icons';
import { apiClient } from '@/services/api';
import { isInWishlist, addToWishlist, removeFromWishlist } from '@/utils/storageUtils';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { SetsStackParamList } from '@/types';
import { C, R, S, shadow } from '@/constants/theme';

type Props = NativeStackScreenProps<SetsStackParamList, 'SetDetailScreen'>;

export const SetDetailScreen: React.FC<Props> = ({ route, navigation }) => {
  const { setNum } = route.params;
  const [isWishlisted, setIsWishlisted] = useState(false);

  const { data: setDetail, isLoading } = useQuery({
    queryKey: ['set', setNum],
    queryFn: () => apiClient.getSet(setNum),
    staleTime: 0,
    // Keep polling until parts list is available
    refetchInterval: (query: any) => (query.state.data?.partsLoading ? 4000 : false),
  });

  useEffect(() => {
    if (setDetail) {
      navigation.setOptions({ title: setDetail.name });
    }
    // Check persistent wishlist state
    isInWishlist(setNum).then(setIsWishlisted).catch(() => {});
  }, [setDetail]);

  const handleBuildCheck = () => navigation.navigate('BuildCheckScreen', { setNum });

  const handleToggleWishlist = async () => {
    try {
      if (isWishlisted) {
        await removeFromWishlist(setNum);
        setIsWishlisted(false);
        Alert.alert('', 'Removed from wishlist');
      } else {
        if (setDetail) {
          await addToWishlist({
            setNum,
            setName: setDetail.name,
            year: setDetail.year,
            theme: setDetail.theme ?? '',
            partCount: setDetail.numParts ?? 0,
            imageUrl: setDetail.imageUrl,
          });
        }
        setIsWishlisted(true);
        Alert.alert('', 'Added to wishlist');
      }
    } catch {
      Alert.alert('Error', 'Could not update wishlist');
    }
  };

  if (isLoading) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color={C.red} />
        <Text style={styles.loadingText}>Loading set…</Text>
      </View>
    );
  }

  if (!setDetail) {
    return (
      <View style={styles.centered}>
        <Ionicons name="alert-circle-outline" size={48} color={C.textMuted} />
        <Text style={styles.emptyTitle}>Set not found</Text>
      </View>
    );
  }

  return (
    <FlatList
      style={styles.root}
      data={setDetail.parts}
      keyExtractor={(item) => `${item.partNum}-${item.colorId}`}
      contentContainerStyle={{ paddingBottom: 32 }}
      ListHeaderComponent={
        <View>
          {/* Hero image */}
          <View style={styles.heroWrap}>
            {setDetail.imageUrl
              ? <Image source={{ uri: setDetail.imageUrl }} style={styles.heroImg} resizeMode="contain" />
              : (
                <View style={styles.heroPlaceholder}>
                  <Ionicons name="layers-outline" size={64} color={C.textMuted} />
                </View>
              )
            }
          </View>

          {/* Set info */}
          <View style={styles.infoSection}>
            {/* Title row */}
            <View style={styles.titleRow}>
              <View style={styles.titleWrap}>
                <Text style={styles.setName}>{setDetail.name}</Text>
                <View style={styles.setMeta}>
                  <View style={styles.metaBadge}>
                    <Ionicons name="barcode-outline" size={12} color={C.textMuted} />
                    <Text style={styles.metaBadgeText}>{setNum}</Text>
                  </View>
                  <View style={styles.metaBadge}>
                    <Ionicons name="calendar-outline" size={12} color={C.textMuted} />
                    <Text style={styles.metaBadgeText}>{setDetail.year}</Text>
                  </View>
                  {setDetail.theme && (
                    <View style={[styles.metaBadge, styles.themeBadge]}>
                      <Text style={styles.themeBadgeText}>{setDetail.theme}</Text>
                    </View>
                  )}
                </View>
              </View>
              <TouchableOpacity
                style={[styles.wishlistBtn, isWishlisted && styles.wishlistBtnActive]}
                onPress={handleToggleWishlist}
                activeOpacity={0.8}
              >
                <Ionicons
                  name={isWishlisted ? 'heart' : 'heart-outline'}
                  size={20}
                  color={isWishlisted ? C.white : C.red}
                />
              </TouchableOpacity>
            </View>

            {/* Stats row */}
            <View style={styles.statsRow}>
              <View style={styles.statCard}>
                <Text style={styles.statValue}>{setDetail.numParts?.toLocaleString()}</Text>
                <Text style={styles.statLabel}>Total Pieces</Text>
              </View>
              <View style={[styles.statCard, styles.statCardAlt]}>
                <Text style={[styles.statValue, styles.statValueAlt]}>{setDetail.parts?.length ?? 0}</Text>
                <Text style={[styles.statLabel, styles.statLabelAlt]}>Unique Parts</Text>
              </View>
            </View>

            {/* Build check CTA */}
            <TouchableOpacity style={styles.buildBtn} onPress={handleBuildCheck} activeOpacity={0.85}>
              <View style={styles.buildBtnIcon}>
                <Ionicons name="construct-outline" size={20} color={C.red} />
              </View>
              <View style={styles.buildBtnText}>
                <Text style={styles.buildBtnTitle}>Check Build Progress</Text>
                <Text style={styles.buildBtnSub}>See which pieces you already own</Text>
              </View>
              <Ionicons name="chevron-forward" size={18} color={C.textMuted} />
            </TouchableOpacity>
          </View>

          {/* Parts section header */}
          <View style={styles.partsHeader}>
            <Text style={styles.partsHeaderTitle}>Parts List</Text>
            <View style={styles.partsHeaderBadge}>
              <Text style={styles.partsHeaderCount}>{setDetail.parts?.length ?? 0}</Text>
            </View>
          </View>
        </View>
      }
      renderItem={({ item, index }) => (
        <View style={[
          styles.partRow,
          index === 0 && styles.partRowFirst,
        ]}>
          {item.imageUrl
            ? <Image source={{ uri: item.imageUrl }} style={styles.partImg} resizeMode="contain" />
            : <View style={[styles.partImg, styles.partImgPlaceholder]}><Ionicons name="cube-outline" size={20} color={C.textMuted} /></View>
          }
          <View style={styles.partInfo}>
            <Text style={styles.partName} numberOfLines={2}>{item.partName}</Text>
            <Text style={styles.partNum}>#{item.partNum}</Text>
          </View>
          <View style={styles.partRight}>
            <View style={[styles.colorSwatch, { backgroundColor: item.colorHex || '#ccc' }]} />
            <View style={styles.qtyBadge}>
              <Text style={styles.qtyText}>×{item.quantity}</Text>
            </View>
          </View>
        </View>
      )}
      ItemSeparatorComponent={() => <View style={styles.separator} />}
    />
  );
};

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: C.bg },

  centered: {
    flex: 1, alignItems: 'center', justifyContent: 'center',
    backgroundColor: C.bg, gap: 12,
  },
  loadingText: { fontSize: 14, color: C.textMuted, marginTop: 8 },
  emptyTitle: { fontSize: 16, fontWeight: '600', color: C.textSub },

  // Hero
  heroWrap: {
    height: 260, backgroundColor: C.white,
    borderBottomWidth: 1, borderBottomColor: C.border,
  },
  heroImg: { width: '100%', height: '100%' },
  heroPlaceholder: {
    flex: 1, alignItems: 'center', justifyContent: 'center',
    backgroundColor: C.bg,
  },

  // Info section
  infoSection: {
    backgroundColor: C.white, padding: S.md,
    borderBottomWidth: 1, borderBottomColor: C.border,
    gap: 16,
  },
  titleRow: { flexDirection: 'row', alignItems: 'flex-start', gap: 12 },
  titleWrap: { flex: 1 },
  setName: { fontSize: 22, fontWeight: '800', color: C.text, letterSpacing: -0.3, marginBottom: 8 },
  setMeta: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  metaBadge: {
    flexDirection: 'row', alignItems: 'center', gap: 4,
    backgroundColor: C.bg, paddingHorizontal: 8, paddingVertical: 4,
    borderRadius: R.full, borderWidth: 1, borderColor: C.border,
  },
  metaBadgeText: { fontSize: 11, color: C.textSub, fontWeight: '600' },
  themeBadge: { backgroundColor: '#FFF5E6', borderColor: '#FFD700' },
  themeBadgeText: { fontSize: 11, color: '#92610A', fontWeight: '700' },

  wishlistBtn: {
    width: 44, height: 44, borderRadius: R.full,
    backgroundColor: '#FFF0F0', alignItems: 'center', justifyContent: 'center',
    borderWidth: 1.5, borderColor: '#FFCCCC',
  },
  wishlistBtnActive: { backgroundColor: C.red, borderColor: C.red },

  // Stats
  statsRow: { flexDirection: 'row', gap: 10 },
  statCard: {
    flex: 1, backgroundColor: C.red, borderRadius: R.lg,
    padding: S.md, ...shadow(1),
  },
  statCardAlt: { backgroundColor: C.yellow },
  statValue: { fontSize: 28, fontWeight: '900', color: C.white, letterSpacing: -0.5 },
  statValueAlt: { color: C.black },
  statLabel: { fontSize: 11, fontWeight: '600', color: 'rgba(255,255,255,0.75)', marginTop: 2 },
  statLabelAlt: { color: 'rgba(0,0,0,0.5)' },

  // Build btn
  buildBtn: {
    flexDirection: 'row', alignItems: 'center', gap: S.md,
    backgroundColor: C.bg, borderRadius: R.lg,
    padding: S.md, borderWidth: 1.5, borderColor: '#FFCCCC',
  },
  buildBtnIcon: {
    width: 44, height: 44, borderRadius: 12,
    backgroundColor: '#FFF0F0', alignItems: 'center', justifyContent: 'center',
  },
  buildBtnText: { flex: 1 },
  buildBtnTitle: { fontSize: 15, fontWeight: '700', color: C.text },
  buildBtnSub: { fontSize: 12, color: C.textMuted, marginTop: 1 },

  // Parts header
  partsHeader: {
    flexDirection: 'row', alignItems: 'center', gap: 8,
    paddingHorizontal: S.md, paddingTop: S.md, paddingBottom: S.sm,
    backgroundColor: C.bg,
  },
  partsHeaderTitle: { fontSize: 16, fontWeight: '800', color: C.text },
  partsHeaderBadge: {
    backgroundColor: C.red, borderRadius: R.full,
    paddingHorizontal: 8, paddingVertical: 2,
  },
  partsHeaderCount: { fontSize: 12, fontWeight: '700', color: C.white },

  // Part rows
  partRow: {
    flexDirection: 'row', alignItems: 'center', gap: S.sm,
    backgroundColor: C.white, padding: S.md,
    marginHorizontal: S.md, borderRadius: 0,
  },
  partRowFirst: { borderTopLeftRadius: R.lg, borderTopRightRadius: R.lg, ...shadow(1) },
  partImg: { width: 48, height: 48, borderRadius: 8 },
  partImgPlaceholder: {
    backgroundColor: C.bg, alignItems: 'center', justifyContent: 'center',
    borderWidth: 1, borderColor: C.border,
  },
  partInfo: { flex: 1 },
  partName: { fontSize: 13, fontWeight: '600', color: C.text, lineHeight: 18 },
  partNum: { fontSize: 11, color: C.textMuted, marginTop: 2, fontFamily: Platform.OS === 'ios' ? 'Courier New' : 'monospace' },
  partRight: { alignItems: 'flex-end', gap: 6 },
  colorSwatch: {
    width: 18, height: 18, borderRadius: 4,
    borderWidth: 1, borderColor: 'rgba(0,0,0,0.12)',
  },
  qtyBadge: {
    backgroundColor: C.bg, paddingHorizontal: 8, paddingVertical: 3,
    borderRadius: R.full, borderWidth: 1, borderColor: C.border,
  },
  qtyText: { fontSize: 12, fontWeight: '700', color: C.textSub },
  separator: { height: 1, backgroundColor: C.border, marginHorizontal: S.md },
});
