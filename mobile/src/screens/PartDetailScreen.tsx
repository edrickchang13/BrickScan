import React, { useState } from 'react';
import {
  View, Text, Image, TouchableOpacity, ScrollView, Linking,
  Alert, ActivityIndicator, StyleSheet, Platform, StatusBar, Modal,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useQuery } from '@tanstack/react-query';
import { useInventoryStore } from '@/store/inventoryStore';
import { LoadingOverlay } from '@/components/LoadingOverlay';
import { apiClient } from '@/services/api';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { ScanStackParamList } from '@/types';
import { C, R, S, shadow } from '@/constants/theme';

type Props = NativeStackScreenProps<ScanStackParamList, 'PartDetailScreen'>;

const REBRICKABLE_URL = (partNum: string) =>
  `https://rebrickable.com/parts/${encodeURIComponent(partNum)}/`;
const BRICKLINK_URL = (partNum: string) =>
  `https://www.bricklink.com/v2/catalog/catalogitem.page?P=${encodeURIComponent(partNum)}`;
const BRICKLINK_IMG = (partNum: string) =>
  `https://img.bricklink.com/ItemImage/PN/11/${partNum}.png`;

export const PartDetailScreen: React.FC<Props> = ({ route, navigation }) => {
  const { partNum, partName, imageUrl, colorId, colorName, colorHex, confidence } = route.params;

  const [imgError, setImgError] = useState(false);
  const [quantity, setQuantity] = useState(1);
  const [showQtyModal, setShowQtyModal] = useState(false);
  const [isAdding, setIsAdding] = useState(false);

  const addItem = useInventoryStore((s) => s.addItem);

  const { data: partInfo, isLoading } = useQuery({
    queryKey: ['part-info', partNum],
    queryFn: () => apiClient.getPartInfo(partNum),
    staleTime: 10 * 60 * 1000,
  });

  const displayName = partInfo?.partName || partName || partNum;
  const category = partInfo?.categoryName;
  const finalImageUrl = imageUrl || BRICKLINK_IMG(partNum);

  const handleAddToInventory = async () => {
    setShowQtyModal(false);
    setIsAdding(true);
    try {
      await addItem(partNum, colorId || '', quantity, colorName || '', colorHex || '');
      Alert.alert(
        'Added!',
        `${quantity}× ${displayName} saved to inventory`,
        [
          { text: 'Scan Another', onPress: () => navigation.navigate('ScanScreen') },
          { text: 'Done', style: 'cancel' },
        ],
      );
    } catch (e: any) {
      Alert.alert('Error', e?.message || 'Failed to add to inventory');
    } finally {
      setIsAdding(false);
    }
  };

  const openLink = async (url: string, label: string) => {
    try {
      const supported = await Linking.canOpenURL(url);
      if (supported) {
        await Linking.openURL(url);
      } else {
        Alert.alert('Cannot open', `Could not open ${label}`);
      }
    } catch {
      Alert.alert('Error', `Failed to open ${label}`);
    }
  };

  const confidencePct = confidence != null ? Math.round(confidence * 100) : null;
  const confColor =
    confidencePct == null ? C.textMuted :
    confidencePct >= 80 ? C.green :
    confidencePct >= 50 ? '#F59E0B' : C.red;

  return (
    <View style={styles.root}>
      <StatusBar barStyle="dark-content" />
      <LoadingOverlay visible={isAdding} message="Adding to inventory…" />

      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()}>
          <Ionicons name="arrow-back" size={22} color={C.text} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Part Details</Text>
        <View style={{ width: 38 }} />
      </View>

      <ScrollView style={styles.scroll} contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>

        {/* Hero image */}
        <View style={styles.heroCard}>
          <View style={styles.imageWrap}>
            {!imgError ? (
              <Image
                source={{ uri: finalImageUrl }}
                style={styles.heroImage}
                resizeMode="contain"
                onError={() => setImgError(true)}
              />
            ) : (
              <View style={styles.imagePlaceholder}>
                <Ionicons name="cube-outline" size={64} color={C.textMuted} />
              </View>
            )}
          </View>

          {isLoading ? (
            <ActivityIndicator color={C.red} style={{ marginVertical: 12 }} />
          ) : (
            <Text style={styles.partName}>{displayName}</Text>
          )}

          <View style={styles.partNumRow}>
            <Ionicons name="barcode-outline" size={14} color={C.textMuted} />
            <Text style={styles.partNum}>#{partNum}</Text>
          </View>

          {/* Badges row */}
          <View style={styles.badgesRow}>
            {category ? (
              <View style={styles.categoryBadge}>
                <Ionicons name="grid-outline" size={12} color="#6366F1" />
                <Text style={styles.categoryText}>{category}</Text>
              </View>
            ) : null}

            {colorName ? (
              <View style={styles.colorBadge}>
                <View style={[styles.colorDot, { backgroundColor: colorHex || '#ccc' }]} />
                <Text style={styles.colorText}>{colorName}</Text>
              </View>
            ) : null}

            {confidencePct != null ? (
              <View style={[styles.confBadge, { backgroundColor: confColor + '18' }]}>
                <Ionicons
                  name={confidencePct >= 50 ? 'checkmark-circle' : 'help-circle'}
                  size={12}
                  color={confColor}
                />
                <Text style={[styles.confText, { color: confColor }]}>{confidencePct}% match</Text>
              </View>
            ) : null}
          </View>
        </View>

        {/* External links */}
        <Text style={styles.sectionLabel}>FIND THIS PART</Text>
        <View style={styles.linksCard}>
          <TouchableOpacity
            style={[styles.linkRow, { borderBottomWidth: 1, borderBottomColor: C.border }]}
            onPress={() => openLink(REBRICKABLE_URL(partNum), 'Rebrickable')}
            activeOpacity={0.75}
          >
            <View style={[styles.linkIcon, { backgroundColor: '#FFF5F5' }]}>
              <Ionicons name="globe-outline" size={20} color={C.red} />
            </View>
            <View style={styles.linkInfo}>
              <Text style={styles.linkTitle}>Rebrickable</Text>
              <Text style={styles.linkSub}>View part catalog page</Text>
            </View>
            <Ionicons name="open-outline" size={16} color={C.textMuted} />
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.linkRow}
            onPress={() => openLink(BRICKLINK_URL(partNum), 'BrickLink')}
            activeOpacity={0.75}
          >
            <View style={[styles.linkIcon, { backgroundColor: '#EFF6FF' }]}>
              <Ionicons name="cart-outline" size={20} color="#3B82F6" />
            </View>
            <View style={styles.linkInfo}>
              <Text style={styles.linkTitle}>BrickLink</Text>
              <Text style={styles.linkSub}>Buy or sell this part</Text>
            </View>
            <Ionicons name="open-outline" size={16} color={C.textMuted} />
          </TouchableOpacity>
        </View>

        {/* Part details info */}
        <Text style={styles.sectionLabel}>PART INFO</Text>
        <View style={styles.infoCard}>
          {[
            { label: 'Part Number', value: partNum },
            { label: 'Category', value: category || '—' },
            { label: 'Color', value: colorName || '—' },
            { label: 'Color ID', value: colorId || '—' },
          ].map((row, i, arr) => (
            <View
              key={row.label}
              style={[
                styles.infoRow,
                i < arr.length - 1 && { borderBottomWidth: 1, borderBottomColor: C.border },
              ]}
            >
              <Text style={styles.infoLabel}>{row.label}</Text>
              <Text style={styles.infoValue} numberOfLines={1}>{row.value}</Text>
            </View>
          ))}
        </View>

        {/* Add to inventory CTA */}
        <TouchableOpacity
          style={styles.primaryBtn}
          onPress={() => setShowQtyModal(true)}
          activeOpacity={0.85}
        >
          <Ionicons name="add-circle-outline" size={20} color={C.white} style={{ marginRight: 8 }} />
          <Text style={styles.primaryBtnText}>Add to Inventory</Text>
        </TouchableOpacity>

      </ScrollView>

      {/* Quantity Modal */}
      <Modal visible={showQtyModal} transparent animationType="slide">
        <TouchableOpacity style={styles.modalBg} activeOpacity={1} onPress={() => setShowQtyModal(false)}>
          <TouchableOpacity activeOpacity={1} style={styles.qtySheet}>
            <View style={styles.sheetHandle} />
            <Text style={styles.qtyTitle}>How many pieces?</Text>
            <Text style={styles.qtyPartLabel}>{displayName}</Text>

            <View style={styles.qtyRow}>
              <TouchableOpacity style={styles.qtyBtn} onPress={() => setQuantity(Math.max(1, quantity - 1))}>
                <Ionicons name="remove" size={24} color={C.text} />
              </TouchableOpacity>
              <View style={styles.qtyDisplay}>
                <Text style={styles.qtyNumber}>{quantity}</Text>
              </View>
              <TouchableOpacity style={styles.qtyBtn} onPress={() => setQuantity(quantity + 1)}>
                <Ionicons name="add" size={24} color={C.text} />
              </TouchableOpacity>
            </View>

            <TouchableOpacity style={styles.primaryBtn} onPress={handleAddToInventory} activeOpacity={0.85}>
              <Ionicons name="checkmark-circle-outline" size={20} color={C.white} style={{ marginRight: 8 }} />
              <Text style={styles.primaryBtnText}>Add {quantity} {quantity === 1 ? 'Piece' : 'Pieces'}</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[styles.secondaryBtn, { marginTop: 0 }]}
              onPress={() => setShowQtyModal(false)}
            >
              <Text style={styles.secondaryBtnText}>Cancel</Text>
            </TouchableOpacity>
          </TouchableOpacity>
        </TouchableOpacity>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: C.bg },

  header: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    backgroundColor: C.white,
    paddingTop: Platform.OS === 'ios' ? 56 : 24,
    paddingBottom: S.md, paddingHorizontal: S.md,
    borderBottomWidth: 1, borderBottomColor: C.border,
  },
  backBtn: {
    width: 38, height: 38, borderRadius: R.full,
    backgroundColor: C.bg, alignItems: 'center', justifyContent: 'center',
  },
  headerTitle: { fontSize: 17, fontWeight: '700', color: C.text },

  scroll: { flex: 1 },
  content: { padding: S.md, gap: 12, paddingBottom: 48 },

  // Hero card
  heroCard: {
    backgroundColor: C.white, borderRadius: R.xl,
    padding: S.lg, alignItems: 'center', ...shadow(2),
  },
  imageWrap: {
    width: 200, height: 200, borderRadius: R.lg,
    backgroundColor: C.bg, alignItems: 'center', justifyContent: 'center',
    marginBottom: S.md,
  },
  heroImage: { width: 180, height: 180 },
  imagePlaceholder: { alignItems: 'center', justifyContent: 'center' },
  partName: {
    fontSize: 22, fontWeight: '800', color: C.text,
    textAlign: 'center', marginBottom: 6, lineHeight: 28,
  },
  partNumRow: { flexDirection: 'row', alignItems: 'center', gap: 6, marginBottom: S.md },
  partNum: {
    fontSize: 13, color: C.textMuted,
    fontFamily: Platform.OS === 'ios' ? 'Courier New' : 'monospace',
  },
  badgesRow: {
    flexDirection: 'row', flexWrap: 'wrap', gap: 8,
    justifyContent: 'center', alignItems: 'center',
  },
  categoryBadge: {
    flexDirection: 'row', alignItems: 'center', gap: 5,
    backgroundColor: '#EEF2FF', paddingHorizontal: 12, paddingVertical: 5,
    borderRadius: R.full,
  },
  categoryText: { fontSize: 12, fontWeight: '600', color: '#6366F1' },
  colorBadge: {
    flexDirection: 'row', alignItems: 'center', gap: 6,
    backgroundColor: C.bg, paddingHorizontal: 12, paddingVertical: 5,
    borderRadius: R.full, borderWidth: 1, borderColor: C.border,
  },
  colorDot: { width: 10, height: 10, borderRadius: 5, borderWidth: 1, borderColor: 'rgba(0,0,0,0.1)' },
  colorText: { fontSize: 12, fontWeight: '600', color: C.textSub },
  confBadge: {
    flexDirection: 'row', alignItems: 'center', gap: 5,
    paddingHorizontal: 12, paddingVertical: 5, borderRadius: R.full,
  },
  confText: { fontSize: 12, fontWeight: '700' },

  // Section label
  sectionLabel: {
    fontSize: 11, fontWeight: '700', color: C.textMuted,
    letterSpacing: 0.8, paddingHorizontal: 2,
  },

  // Links card
  linksCard: {
    backgroundColor: C.white, borderRadius: R.xl,
    overflow: 'hidden', ...shadow(1),
  },
  linkRow: { flexDirection: 'row', alignItems: 'center', padding: S.md, gap: S.md },
  linkIcon: { width: 40, height: 40, borderRadius: 10, alignItems: 'center', justifyContent: 'center' },
  linkInfo: { flex: 1 },
  linkTitle: { fontSize: 15, fontWeight: '600', color: C.text },
  linkSub: { fontSize: 12, color: C.textMuted, marginTop: 1 },

  // Info card
  infoCard: {
    backgroundColor: C.white, borderRadius: R.xl,
    overflow: 'hidden', ...shadow(1),
  },
  infoRow: {
    flexDirection: 'row', justifyContent: 'space-between',
    alignItems: 'center', padding: S.md,
  },
  infoLabel: { fontSize: 14, color: C.textSub },
  infoValue: { fontSize: 14, fontWeight: '600', color: C.text, maxWidth: '55%', textAlign: 'right' },

  // Buttons
  primaryBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    backgroundColor: C.red, borderRadius: R.md, paddingVertical: 15, marginTop: 4,
  },
  primaryBtnText: { color: C.white, fontSize: 16, fontWeight: '700' },
  secondaryBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    backgroundColor: C.white, borderRadius: R.md,
    paddingVertical: 14, marginTop: 4,
    borderWidth: 1.5, borderColor: C.border, ...shadow(1),
  },
  secondaryBtnText: { color: C.text, fontSize: 15, fontWeight: '600' },

  // Qty modal
  modalBg: { flex: 1, backgroundColor: 'rgba(0,0,0,0.5)', justifyContent: 'flex-end' },
  qtySheet: {
    backgroundColor: C.white, borderTopLeftRadius: 24, borderTopRightRadius: 24,
    padding: S.lg, paddingBottom: Platform.OS === 'ios' ? 40 : S.lg, gap: 12,
  },
  sheetHandle: {
    width: 40, height: 4, borderRadius: 2, backgroundColor: C.border,
    alignSelf: 'center', marginBottom: S.sm,
  },
  qtyTitle: { fontSize: 20, fontWeight: '800', color: C.text },
  qtyPartLabel: { fontSize: 13, color: C.textMuted, marginTop: -4 },
  qtyRow: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    backgroundColor: C.bg, borderRadius: R.md, padding: 8,
  },
  qtyBtn: {
    width: 52, height: 52, borderRadius: 14,
    backgroundColor: C.white, alignItems: 'center', justifyContent: 'center', ...shadow(1),
  },
  qtyDisplay: { flex: 1, alignItems: 'center' },
  qtyNumber: { fontSize: 40, fontWeight: '900', color: C.red },
});
