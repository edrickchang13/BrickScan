import React, { useState } from 'react';
import {
  View, Text, Image, TouchableOpacity, ScrollView,
  Alert, StyleSheet, Platform, StatusBar,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useInventoryStore } from '@/store/inventoryStore';
import { LoadingOverlay } from '@/components/LoadingOverlay';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { ScanStackParamList, DetectedPiece } from '@/types';
import { C, R, S, shadow } from '@/constants/theme';

type Props = NativeStackScreenProps<ScanStackParamList, 'MultiResultScreen'>;

const PieceImage: React.FC<{ imageUrl?: string; size?: number }> = ({ imageUrl, size = 56 }) => {
  const [err, setErr] = useState(false);
  if (imageUrl && !err) {
    return (
      <Image
        source={{ uri: imageUrl }}
        style={{ width: size, height: size, borderRadius: 8 }}
        resizeMode="contain"
        onError={() => setErr(true)}
      />
    );
  }
  return (
    <View style={[{ width: size, height: size, borderRadius: 8 }, styles.imgPlaceholder]}>
      <Ionicons name="cube-outline" size={size * 0.4} color={C.textMuted} />
    </View>
  );
};

export const MultiResultScreen: React.FC<Props> = ({ route, navigation }) => {
  const { pieces } = route.params;
  const [selectedPieces, setSelectedPieces] = useState<Set<number>>(
    new Set(pieces.map((_, i) => i))
  );
  const [isAdding, setIsAdding] = useState(false);
  const addItem = useInventoryStore((state) => state.addItem);

  const togglePiece = (index: number) => {
    const next = new Set(selectedPieces);
    if (next.has(index)) {
      next.delete(index);
    } else {
      next.add(index);
    }
    setSelectedPieces(next);
  };

  const handleAddAll = async () => {
    const toAdd = pieces.filter((_, i) => selectedPieces.has(i));
    if (toAdd.length === 0) {
      Alert.alert('Nothing Selected', 'Select at least one piece to add.');
      return;
    }

    setIsAdding(true);
    try {
      let added = 0;
      for (const piece of toAdd) {
        const pred = piece.primaryPrediction;
        await addItem(pred.partNum, pred.colorId, 1, pred.colorName, pred.colorHex);
        added++;
      }
      Alert.alert(
        'Added!',
        `${added} piece${added === 1 ? '' : 's'} added to inventory`,
        [
          { text: 'Scan More', onPress: () => navigation.navigate('ScanScreen') },
          { text: 'Done', onPress: () => navigation.navigate('ScanScreen') },
        ],
      );
    } catch (error: any) {
      Alert.alert('Error', error?.message || 'Failed to add items');
    } finally {
      setIsAdding(false);
    }
  };

  return (
    <View style={styles.root}>
      <StatusBar barStyle="dark-content" />
      <LoadingOverlay visible={isAdding} message="Adding to inventory..." />

      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity style={styles.backBtn} onPress={() => navigation.navigate('ScanScreen')}>
          <Ionicons name="arrow-back" size={22} color={C.text} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Multi-Piece Scan</Text>
        <View style={styles.countBadge}>
          <Text style={styles.countText}>{pieces.length}</Text>
        </View>
      </View>

      <ScrollView style={styles.scroll} contentContainerStyle={styles.scrollContent}>
        {/* Summary banner */}
        <View style={styles.summaryBanner}>
          <Ionicons name="grid-outline" size={20} color="#6366F1" />
          <Text style={styles.summaryText}>
            Found {pieces.length} piece{pieces.length === 1 ? '' : 's'} — tap to select/deselect
          </Text>
        </View>

        {/* Piece cards */}
        {pieces.map((piece, index) => {
          const pred = piece.primaryPrediction;
          const confPct = Math.round(pred.confidence * 100);
          const confColor = confPct >= 80 ? C.green : confPct >= 50 ? '#F59E0B' : C.red;
          const isSelected = selectedPieces.has(index);

          return (
            <TouchableOpacity
              key={index}
              style={[styles.pieceCard, isSelected && styles.pieceCardSelected]}
              onPress={() => togglePiece(index)}
              activeOpacity={0.8}
            >
              <View style={styles.checkCol}>
                <View style={[styles.checkbox, isSelected && styles.checkboxChecked]}>
                  {isSelected && <Ionicons name="checkmark" size={14} color={C.white} />}
                </View>
              </View>

              <PieceImage imageUrl={pred.imageUrl} />

              <View style={styles.pieceInfo}>
                <Text style={styles.pieceName} numberOfLines={1}>{pred.partName}</Text>
                <Text style={styles.pieceNum}>#{pred.partNum}</Text>
                <View style={styles.pieceMetaRow}>
                  {pred.colorName ? (
                    <View style={styles.colorChip}>
                      <View style={[styles.colorDot, { backgroundColor: pred.colorHex || '#ccc' }]} />
                      <Text style={styles.colorLabel}>{pred.colorName}</Text>
                    </View>
                  ) : null}
                  <View style={[styles.confBadge, { backgroundColor: confColor + '18' }]}>
                    <Text style={[styles.confText, { color: confColor }]}>{confPct}%</Text>
                  </View>
                </View>
              </View>

              {/* Alt predictions count */}
              {piece.predictions.length > 1 && (
                <View style={styles.altCount}>
                  <Text style={styles.altCountText}>+{piece.predictions.length - 1}</Text>
                </View>
              )}
            </TouchableOpacity>
          );
        })}
      </ScrollView>

      {/* Bottom action bar */}
      <View style={styles.actionBar}>
        <TouchableOpacity
          style={styles.selectAllBtn}
          onPress={() => {
            if (selectedPieces.size === pieces.length) {
              setSelectedPieces(new Set());
            } else {
              setSelectedPieces(new Set(pieces.map((_, i) => i)));
            }
          }}
        >
          <Ionicons
            name={selectedPieces.size === pieces.length ? 'checkbox' : 'square-outline'}
            size={20}
            color={C.text}
          />
          <Text style={styles.selectAllText}>
            {selectedPieces.size === pieces.length ? 'Deselect All' : 'Select All'}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.addAllBtn, selectedPieces.size === 0 && styles.addAllBtnDisabled]}
          onPress={handleAddAll}
          disabled={selectedPieces.size === 0}
          activeOpacity={0.85}
        >
          <Ionicons name="add-circle-outline" size={18} color={C.white} style={{ marginRight: 6 }} />
          <Text style={styles.addAllText}>
            Add {selectedPieces.size} Piece{selectedPieces.size === 1 ? '' : 's'}
          </Text>
        </TouchableOpacity>
      </View>
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
  countBadge: {
    backgroundColor: C.red, borderRadius: R.full,
    paddingHorizontal: 10, paddingVertical: 4,
  },
  countText: { color: C.white, fontSize: 13, fontWeight: '700' },

  scroll: { flex: 1 },
  scrollContent: { padding: S.md, gap: 10, paddingBottom: 120 },

  summaryBanner: {
    flexDirection: 'row', alignItems: 'center', gap: 10,
    backgroundColor: '#EEF2FF', borderRadius: R.md,
    padding: 12, borderWidth: 1, borderColor: '#C7D2FE',
  },
  summaryText: { flex: 1, fontSize: 13, color: '#4338CA', lineHeight: 18 },

  // Piece cards
  pieceCard: {
    flexDirection: 'row', alignItems: 'center',
    backgroundColor: C.white, borderRadius: R.lg,
    padding: S.md, gap: S.md,
    borderWidth: 1.5, borderColor: C.border,
    ...shadow(1),
  },
  pieceCardSelected: {
    borderColor: C.red,
    backgroundColor: '#FFF5F5',
  },

  checkCol: { width: 28 },
  checkbox: {
    width: 22, height: 22, borderRadius: 6,
    borderWidth: 2, borderColor: C.border,
    alignItems: 'center', justifyContent: 'center',
  },
  checkboxChecked: {
    backgroundColor: C.red, borderColor: C.red,
  },

  imgPlaceholder: {
    backgroundColor: C.bg, alignItems: 'center', justifyContent: 'center',
  },

  pieceInfo: { flex: 1, gap: 2 },
  pieceName: { fontSize: 14, fontWeight: '600', color: C.text },
  pieceNum: { fontSize: 11, color: C.textMuted, fontFamily: Platform.OS === 'ios' ? 'Courier New' : 'monospace' },
  pieceMetaRow: { flexDirection: 'row', gap: 6, marginTop: 4, alignItems: 'center' },
  colorChip: {
    flexDirection: 'row', alignItems: 'center', gap: 4,
    backgroundColor: C.bg, paddingHorizontal: 8, paddingVertical: 3,
    borderRadius: R.full,
  },
  colorDot: { width: 10, height: 10, borderRadius: 5, borderWidth: 1, borderColor: 'rgba(0,0,0,0.1)' },
  colorLabel: { fontSize: 11, fontWeight: '500', color: C.textSub },
  confBadge: {
    paddingHorizontal: 8, paddingVertical: 3, borderRadius: R.full,
  },
  confText: { fontSize: 11, fontWeight: '700' },

  altCount: {
    backgroundColor: C.bg, borderRadius: R.full,
    paddingHorizontal: 8, paddingVertical: 3,
  },
  altCountText: { fontSize: 11, fontWeight: '600', color: C.textMuted },

  // Action bar
  actionBar: {
    position: 'absolute', bottom: 0, left: 0, right: 0,
    backgroundColor: C.white, borderTopWidth: 1, borderTopColor: C.border,
    paddingHorizontal: S.md,
    paddingTop: S.md,
    paddingBottom: Platform.OS === 'ios' ? 40 : S.lg,
    flexDirection: 'row', alignItems: 'center', gap: S.md,
    ...shadow(2),
  },
  selectAllBtn: {
    flexDirection: 'row', alignItems: 'center', gap: 6,
    paddingVertical: 12, paddingHorizontal: 12,
  },
  selectAllText: { fontSize: 13, fontWeight: '600', color: C.text },
  addAllBtn: {
    flex: 1, flexDirection: 'row',
    alignItems: 'center', justifyContent: 'center',
    backgroundColor: C.red, borderRadius: R.md,
    paddingVertical: 14,
  },
  addAllBtnDisabled: { opacity: 0.4 },
  addAllText: { color: C.white, fontSize: 15, fontWeight: '700' },
});
