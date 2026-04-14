import React, { useState, useMemo } from 'react';
import {
  View, Text, TextInput, FlatList, TouchableOpacity,
  Alert, RefreshControl, Modal, ScrollView, StyleSheet, StatusBar, Platform,
  Image,
} from 'react-native';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { Ionicons } from '@expo/vector-icons';
import { useInventoryStore } from '@/store/inventoryStore';
import { PartCard } from '@/components/PartCard';
import { LoadingOverlay } from '@/components/LoadingOverlay';
import type { InventoryItem, InventoryStackParamList } from '@/types';
import { C, R, S, shadow } from '@/constants/theme';

type InventoryNavProp = NativeStackNavigationProp<InventoryStackParamList, 'InventoryScreen'>;

export const InventoryScreen: React.FC = () => {
  const navigation = useNavigation<InventoryNavProp>();
  const items = useInventoryStore((s) => s.items);
  const isLoading = useInventoryStore((s) => s.isLoading);
  const error = useInventoryStore((s) => s.error);
  const fetchInventory = useInventoryStore((s) => s.fetchInventory);
  const removeItem = useInventoryStore((s) => s.removeItem);
  const updateItem = useInventoryStore((s) => s.updateItem);

  const [searchQuery, setSearchQuery] = useState('');
  const [selectedColor, setSelectedColor] = useState<string | null>(null);
  const [showFilterModal, setShowFilterModal] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedItem, setSelectedItem] = useState<InventoryItem | null>(null);
  const [itemQuantity, setItemQuantity] = useState(0);
  type SortMode = 'nameAZ' | 'qtyHigh' | 'qtyLow';
  const [sortMode, setSortMode] = useState<SortMode>('nameAZ');
  const cycleSortMode = () => {
    setSortMode(s => s === 'nameAZ' ? 'qtyHigh' : s === 'qtyHigh' ? 'qtyLow' : 'nameAZ');
  };
  const sortLabel: Record<SortMode, string> = { nameAZ: 'A→Z', qtyHigh: 'Qty ↓', qtyLow: 'Qty ↑' };

  useFocusEffect(React.useCallback(() => { fetchInventory().catch(console.error); }, []));

  const onRefresh = async () => {
    setRefreshing(true);
    try {
      await fetchInventory();
    } catch (err: any) {
      console.error('Refresh failed:', err?.message || err);
    } finally {
      setRefreshing(false);
    }
  };

  const handleDelete = (item: InventoryItem) => {
    Alert.alert('Remove Part', `Remove ${item.partName} from your inventory?`, [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Remove', style: 'destructive', onPress: async () => {
        try { await removeItem(item.id); } catch (e: any) { Alert.alert('Error', e?.message); }
      }},
    ]);
  };

  const handleUpdateQty = async (item: InventoryItem) => {
    if (itemQuantity <= 0) { Alert.alert('Invalid', 'Quantity must be at least 1'); return; }
    try {
      await updateItem(item.id, itemQuantity);
      setSelectedItem(null);
    } catch (e: any) { Alert.alert('Error', e?.message); }
  };

  // Build color list dynamically from actual inventory — only colors you own
  const inventoryColors = useMemo(() => {
    const seen = new Map<string, string>();
    items.forEach((it) => {
      if (it.colorName && !seen.has(it.colorName)) {
        seen.set(it.colorName, it.colorHex || '#ccc');
      }
    });
    return Array.from(seen.entries())
      .map(([name, hex]) => ({ name, hex }))
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [items]);

  const filtered = items
    .filter((it) => {
      const q = searchQuery.toLowerCase();
      const matchSearch = it.partName.toLowerCase().includes(q) || it.partNum.toLowerCase().includes(q);
      return matchSearch && (!selectedColor || it.colorName === selectedColor);
    })
    .sort((a, b) => {
      if (sortMode === 'nameAZ') return a.partName.localeCompare(b.partName);
      if (sortMode === 'qtyHigh') return b.quantity - a.quantity;
      return a.quantity - b.quantity;
    });

  const totalPieces = filtered.reduce((s, it) => s + it.quantity, 0);

  return (
    <View style={styles.root}>
      <StatusBar barStyle="dark-content" />
      <LoadingOverlay visible={isLoading} message="Loading inventory…" />

      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>My Inventory</Text>
        <Text style={styles.headerSub}>{totalPieces.toLocaleString()} pieces · {filtered.length} types</Text>
      </View>

      {/* Search + filter */}
      <View style={styles.searchRow}>
        <View style={styles.searchInputWrap}>
          <Ionicons name="search" size={16} color={C.textMuted} style={styles.searchIcon} />
          <TextInput
            style={styles.searchInput}
            placeholder="Search by name or number…"
            placeholderTextColor={C.textMuted}
            value={searchQuery}
            onChangeText={setSearchQuery}
          />
          {searchQuery.length > 0 && (
            <TouchableOpacity onPress={() => setSearchQuery('')} style={{ padding: 4 }}>
              <Ionicons name="close-circle" size={16} color={C.textMuted} />
            </TouchableOpacity>
          )}
        </View>
        <TouchableOpacity
          style={styles.sortBtn}
          onPress={cycleSortMode}
          activeOpacity={0.7}
        >
          <Text style={styles.sortBtnText}>{sortLabel[sortMode]}</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.filterBtn, !!selectedColor && styles.filterBtnActive]}
          onPress={() => setShowFilterModal(true)}
          activeOpacity={0.7}
        >
          <Ionicons name="funnel" size={18} color={selectedColor ? C.white : C.red} />
        </TouchableOpacity>
      </View>

      {selectedColor && (
        <View style={styles.activeFilterRow}>
          <View style={styles.filterChip}>
            <View style={[styles.colorDot, {
              backgroundColor: inventoryColors.find(c => c.name === selectedColor)?.hex || '#ccc'
            }]} />
            <Text style={styles.filterChipText}>{selectedColor}</Text>
            <TouchableOpacity onPress={() => setSelectedColor(null)}>
              <Ionicons name="close" size={14} color={C.red} />
            </TouchableOpacity>
          </View>
        </View>
      )}

      {filtered.length > 0 && (
        <View style={styles.hintRow}>
          <Ionicons name="information-circle-outline" size={13} color={C.textMuted} />
          <Text style={styles.hintText}>Tap to edit quantity · Hold to delete</Text>
        </View>
      )}

      <FlatList
        style={{ flex: 1 }}
        data={filtered}
        keyExtractor={(item) => item.id}
        numColumns={2}
        columnWrapperStyle={styles.columnWrap}
        contentContainerStyle={styles.listContent}
        ListEmptyComponent={
          error && filtered.length === 0 ? (
            <View style={styles.emptyState}>
              <View style={styles.emptyIcon}>
                <Ionicons name="alert-circle-outline" size={40} color={C.red} />
              </View>
              <Text style={styles.emptyTitle}>Failed to load inventory</Text>
              <Text style={styles.emptySub}>{error}</Text>
            </View>
          ) : (
            <View style={styles.emptyState}>
              <View style={styles.emptyIcon}>
                <Ionicons name="cube-outline" size={40} color={C.textMuted} />
              </View>
              <Text style={styles.emptyTitle}>No pieces yet</Text>
              <Text style={styles.emptySub}>
                Scan LEGO pieces with the camera to build your collection
              </Text>
            </View>
          )
        }
        renderItem={({ item }) => (
          <TouchableOpacity
            style={styles.cardWrap}
            onPress={() => { setSelectedItem(item); setItemQuantity(item.quantity); }}
            onLongPress={() => handleDelete(item)}
            activeOpacity={0.80}
          >
            <PartCard
              partNum={item.partNum}
              name={item.partName}
              colorName={item.colorName}
              colorHex={item.colorHex}
              quantity={item.quantity}
              imageUrl={item.imageUrl}
              size="medium"
            />
          </TouchableOpacity>
        )}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={C.red} />}
      />

      {/* Filter Modal */}
      <Modal visible={showFilterModal} transparent animationType="slide">
        <View style={styles.modalBg}>
          <View style={styles.bottomSheet}>
            <View style={styles.sheetHandle} />
            <Text style={styles.sheetTitle}>Filter by Color</Text>
            <ScrollView showsVerticalScrollIndicator={false} style={{ maxHeight: 380 }}>
              <TouchableOpacity
                style={[styles.colorRow, selectedColor === null && styles.colorRowActive]}
                onPress={() => { setSelectedColor(null); setShowFilterModal(false); }}
              >
                <Text style={[styles.colorRowText, selectedColor === null && styles.colorRowTextActive]}>
                  All Colors
                </Text>
                {selectedColor === null && <Ionicons name="checkmark" size={18} color={C.red} />}
              </TouchableOpacity>

              {inventoryColors.length === 0 ? (
                <Text style={[styles.colorRowText, { padding: 14, color: C.textMuted }]}>
                  No colors in inventory yet
                </Text>
              ) : inventoryColors.map((color) => (
                <TouchableOpacity
                  key={color.name}
                  style={[styles.colorRow, selectedColor === color.name && styles.colorRowActive]}
                  onPress={() => { setSelectedColor(color.name); setShowFilterModal(false); }}
                  activeOpacity={0.7}
                >
                  <View style={[styles.colorSwatch, { backgroundColor: color.hex }]} />
                  <Text style={[styles.colorRowText, selectedColor === color.name && styles.colorRowTextActive]}>
                    {color.name}
                  </Text>
                  {selectedColor === color.name && <Ionicons name="checkmark" size={18} color={C.red} />}
                </TouchableOpacity>
              ))}
            </ScrollView>
          </View>
        </View>
      </Modal>

      {/* Quantity & Details Modal */}
      <Modal visible={selectedItem !== null} transparent animationType="fade">
        <TouchableOpacity
          style={styles.qtyModalBg}
          activeOpacity={1}
          onPress={() => setSelectedItem(null)}
        >
          <TouchableOpacity activeOpacity={1} style={styles.qtyCard} onPress={() => {}}>
            {/* Part image */}
            {selectedItem?.imageUrl && (
              <View style={styles.detailImage}>
                <Image
                  source={{ uri: selectedItem.imageUrl }}
                  style={{ width: '100%', height: '100%' }}
                  resizeMode="contain"
                />
              </View>
            )}

            {/* Part info */}
            <View style={styles.detailContent}>
              <Text style={styles.qtyTitle}>{selectedItem?.partName}</Text>
              <Text style={styles.qtyPartNum}>#{selectedItem?.partNum}</Text>

              {/* Color badge */}
              {selectedItem && (
                <View style={styles.colorBadge}>
                  <View style={[styles.colorBadgeDot, { backgroundColor: selectedItem.colorHex }]} />
                  <Text style={styles.colorBadgeText}>{selectedItem.colorName}</Text>
                </View>
              )}

              {/* Quantity controls */}
              <View style={styles.qtySection}>
                <Text style={styles.qtySectionLabel}>Quantity</Text>
                <View style={styles.qtyRow}>
                  <TouchableOpacity
                    style={styles.qtyBtn}
                    onPress={() => setItemQuantity(Math.max(1, itemQuantity - 1))}
                    activeOpacity={0.7}
                  >
                    <Ionicons name="remove" size={22} color={C.text} />
                  </TouchableOpacity>
                  <View style={styles.qtyDisplay}>
                    <Text style={styles.qtyNumber}>{itemQuantity}</Text>
                  </View>
                  <TouchableOpacity
                    style={styles.qtyBtn}
                    onPress={() => setItemQuantity(itemQuantity + 1)}
                    activeOpacity={0.7}
                  >
                    <Ionicons name="add" size={22} color={C.text} />
                  </TouchableOpacity>
                </View>
              </View>

              {/* Action buttons */}
              <View style={styles.buttonRow}>
                <TouchableOpacity
                  style={styles.qtyConfirmBtn}
                  onPress={() => selectedItem && handleUpdateQty(selectedItem)}
                  activeOpacity={0.85}
                >
                  <Ionicons name="checkmark-circle" size={18} color={C.white} style={{ marginRight: 6 }} />
                  <Text style={styles.qtyConfirmText}>Save Changes</Text>
                </TouchableOpacity>

                <TouchableOpacity
                  style={styles.qtyDeleteBtn}
                  onPress={() => { handleDelete(selectedItem!); setSelectedItem(null); }}
                  activeOpacity={0.85}
                >
                  <Ionicons name="trash-outline" size={18} color={C.red} />
                </TouchableOpacity>
              </View>

              {/* View full part details */}
              <TouchableOpacity
                style={styles.qtyDetailBtn}
                onPress={() => {
                  if (!selectedItem) return;
                  setSelectedItem(null);
                  navigation.navigate('PartDetailScreen', {
                    partNum: selectedItem.partNum,
                    partName: selectedItem.partName,
                    imageUrl: selectedItem.imageUrl,
                    colorId: selectedItem.colorId,
                    colorName: selectedItem.colorName,
                    colorHex: selectedItem.colorHex,
                  });
                }}
                activeOpacity={0.85}
              >
                <Ionicons name="information-circle-outline" size={16} color={C.red} style={{ marginRight: 6 }} />
                <Text style={styles.qtyDetailText}>View Part Details</Text>
                <Ionicons name="chevron-forward" size={14} color={C.red} style={{ marginLeft: 4 }} />
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.qtyCancelBtn}
                onPress={() => setSelectedItem(null)}
              >
                <Text style={styles.qtyCancelText}>Cancel</Text>
              </TouchableOpacity>
            </View>
          </TouchableOpacity>
        </TouchableOpacity>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: C.bg },

  header: {
    backgroundColor: C.white,
    paddingTop: Platform.OS === 'ios' ? 56 : 24,
    paddingHorizontal: S.md,
    paddingBottom: S.md,
    borderBottomWidth: 1, borderBottomColor: C.border,
  },
  headerTitle: { fontSize: 28, fontWeight: '800', color: C.text, letterSpacing: -0.5 },
  headerSub: { fontSize: 13, color: C.textMuted, marginTop: 2 },

  searchRow: {
    flexDirection: 'row', alignItems: 'center', gap: 8,
    paddingHorizontal: S.md, paddingVertical: 10,
    backgroundColor: C.white, borderBottomWidth: 1, borderBottomColor: C.border,
  },
  searchInputWrap: {
    flex: 1, flexDirection: 'row', alignItems: 'center',
    backgroundColor: C.bg, borderRadius: R.full,
    paddingHorizontal: 12, paddingVertical: 8,
  },
  searchIcon: { marginRight: 8 },
  searchInput: { flex: 1, fontSize: 14, color: C.text },
  sortBtn: {
    paddingHorizontal: 12, height: 40, borderRadius: R.full,
    backgroundColor: C.bg, alignItems: 'center', justifyContent: 'center',
    borderWidth: 1, borderColor: C.border,
  },
  sortBtnText: { fontSize: 12, fontWeight: '700', color: C.textSub },
  filterBtn: {
    width: 40, height: 40, borderRadius: R.full,
    backgroundColor: '#FFF0F0', alignItems: 'center', justifyContent: 'center',
  },
  filterBtnActive: { backgroundColor: C.red },
  hintRow: {
    flexDirection: 'row', alignItems: 'center', gap: 4,
    paddingHorizontal: S.md, paddingVertical: 6,
  },
  hintText: { fontSize: 11, color: C.textMuted },

  activeFilterRow: { flexDirection: 'row', paddingHorizontal: S.md, paddingVertical: 8, gap: 8 },
  filterChip: {
    flexDirection: 'row', alignItems: 'center', gap: 6,
    backgroundColor: '#FFF0F0', paddingHorizontal: 12, paddingVertical: 5,
    borderRadius: R.full, borderWidth: 1, borderColor: '#FFCCCC',
  },
  colorDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: C.red },
  filterChipText: { fontSize: 13, fontWeight: '600', color: C.red },

  columnWrap: { gap: 8 },
  listContent: { padding: 12, gap: 8, paddingBottom: 32 },
  cardWrap: { flex: 1 },

  emptyState: { alignItems: 'center', paddingTop: 80, paddingHorizontal: S.xl },
  emptyIcon: {
    width: 80, height: 80, borderRadius: 20, backgroundColor: C.cardAlt,
    alignItems: 'center', justifyContent: 'center', marginBottom: S.md,
  },
  emptyTitle: { fontSize: 17, fontWeight: '700', color: C.text, marginBottom: 6 },
  emptySub: { fontSize: 14, color: C.textSub, textAlign: 'center', lineHeight: 20 },

  // Filter modal
  modalBg: { flex: 1, backgroundColor: C.overlay, justifyContent: 'flex-end' },
  bottomSheet: {
    backgroundColor: C.white, borderTopLeftRadius: 24, borderTopRightRadius: 24,
    padding: S.lg, paddingBottom: 40,
  },
  sheetHandle: {
    width: 40, height: 4, borderRadius: 2, backgroundColor: C.border,
    alignSelf: 'center', marginBottom: S.md,
  },
  sheetTitle: { fontSize: 18, fontWeight: '800', color: C.text, marginBottom: S.md },
  colorRow: {
    flexDirection: 'row', alignItems: 'center', padding: 14,
    borderRadius: R.md, marginBottom: 4, gap: 10,
  },
  colorRowActive: { backgroundColor: '#FFF0F0' },
  colorSwatch: { width: 16, height: 16, borderRadius: 4 },
  colorRowText: { flex: 1, fontSize: 15, color: C.text },
  colorRowTextActive: { fontWeight: '700', color: C.red },

  // Qty modal
  qtyModalBg: {
    flex: 1, backgroundColor: C.overlay,
    alignItems: 'center', justifyContent: 'flex-end', padding: S.lg,
  },
  qtyCard: {
    backgroundColor: C.white, borderRadius: 24, width: '100%',
    maxWidth: 380, overflow: 'hidden', marginBottom: S.lg, ...shadow(3),
  },
  detailImage: {
    width: '100%', height: 160, backgroundColor: C.bg,
    alignItems: 'center', justifyContent: 'center',
  },
  detailContent: {
    padding: S.lg, gap: 12,
  },
  qtyTitle: { fontSize: 18, fontWeight: '800', color: C.text, marginBottom: 0 },
  qtyPartNum: {
    fontSize: 12, color: C.textMuted, fontFamily: Platform.OS === 'ios' ? 'Courier New' : 'monospace',
    letterSpacing: 0.5,
  },
  colorBadge: {
    flexDirection: 'row', alignItems: 'center', gap: 8,
    backgroundColor: C.bg, paddingHorizontal: S.md, paddingVertical: 8,
    borderRadius: R.md,
  },
  colorBadgeDot: { width: 14, height: 14, borderRadius: 7, borderWidth: 1, borderColor: 'rgba(0,0,0,0.1)' },
  colorBadgeText: { fontSize: 13, fontWeight: '600', color: C.text },
  qtySection: {
    marginTop: S.sm,
  },
  qtySectionLabel: { fontSize: 11, fontWeight: '700', color: C.textMuted, marginBottom: 8, letterSpacing: 0.5 },
  qtyRow: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    backgroundColor: C.bg, borderRadius: R.md, padding: 8,
  },
  qtyBtn: {
    width: 48, height: 48, borderRadius: 12,
    backgroundColor: C.white, alignItems: 'center', justifyContent: 'center',
    ...shadow(1),
  },
  qtyDisplay: {
    flex: 1, alignItems: 'center',
  },
  qtyNumber: { fontSize: 36, fontWeight: '900', color: C.red },
  buttonRow: {
    flexDirection: 'row', gap: 8, marginTop: S.sm,
  },
  qtyConfirmBtn: {
    flex: 1, backgroundColor: C.red, borderRadius: R.md, paddingVertical: 12,
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
  },
  qtyConfirmText: { color: C.white, fontSize: 15, fontWeight: '700', marginLeft: 2 },
  qtyDeleteBtn: {
    width: 48, height: 48, borderRadius: R.md,
    backgroundColor: '#FFF0F0', borderWidth: 1.5, borderColor: '#FFCCCC',
    alignItems: 'center', justifyContent: 'center',
  },
  qtyCancelBtn: {
    backgroundColor: C.white, borderWidth: 1.5, borderColor: C.border,
    borderRadius: R.md, paddingVertical: 11, alignItems: 'center',
  },
  qtyCancelText: { color: C.text, fontSize: 15, fontWeight: '600' },
  qtyDetailBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    backgroundColor: '#FFF5F5', borderRadius: R.md, paddingVertical: 11,
    borderWidth: 1.5, borderColor: C.red + '30',
  },
  qtyDetailText: { color: C.red, fontSize: 14, fontWeight: '600' },
});
