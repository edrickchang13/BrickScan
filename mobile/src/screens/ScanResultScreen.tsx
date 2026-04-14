import React, { useState, useCallback, useEffect } from 'react';
import {
  View, Text, Image, TouchableOpacity, ScrollView,
  Alert, Modal, StyleSheet, Platform, StatusBar,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useInventoryStore } from '@/store/inventoryStore';
import { LoadingOverlay } from '@/components/LoadingOverlay';
import { saveRecentScan } from '@/utils/storageUtils';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { ScanStackParamList } from '@/types';
import { C, R, S, shadow } from '@/constants/theme';
import { FeedbackRow } from '@/components/FeedbackRow';
import { SubstituteSuggestions } from '@/components/SubstituteSuggestions';

type Props = NativeStackScreenProps<ScanStackParamList, 'ScanResultScreen'>;

// Small helper so each alt row has its own error state
const AltImage: React.FC<{ imageUrl?: string }> = ({ imageUrl }) => {
  const [err, setErr] = useState(false);
  if (imageUrl && !err) {
    return <Image source={{ uri: imageUrl }} style={styles.altImg} resizeMode="contain" onError={() => setErr(true)} />;
  }
  return <View style={[styles.altImg, styles.altImgPlaceholder]}><Ionicons name="cube-outline" size={20} color={C.textMuted} /></View>;
};

export const ScanResultScreen: React.FC<Props> = ({ route, navigation }) => {
  const { predictions, scanMode, framesAnalyzed, agreementScore } = route.params;
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [quantity, setQuantity] = useState(1);
  const [showQuantityModal, setShowQuantityModal] = useState(false);
  const [isAdding, setIsAdding] = useState(false);
  const [imageError, setImageError] = useState(false);
  const scanId = React.useRef(
    `scan_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  ).current;

  const addItem = useInventoryStore((state) => state.addItem);
  const selected = predictions[selectedIndex];
  const isLowConfidence = selected.confidence < 0.25;

  // Save the top prediction to scan history as soon as this screen loads
  useEffect(() => {
    const top = predictions[0];
    if (top?.partNum) {
      saveRecentScan({
        id: `${top.partNum}-${Date.now()}`,
        partNum: top.partNum,
        partName: top.partName,
        colorId: top.colorId,
        colorName: top.colorName,
        colorHex: top.colorHex,
        confidence: top.confidence,
        imageUrl: top.imageUrl,
        timestamp: Date.now(),
        scanMode,
        source: (top as any).source,
        framesAnalyzed,
        agreementScore,
        // Preserve all predictions so history re-opens with the full result
        allPredictions: predictions.map((p) => ({
          partNum: p.partNum,
          partName: p.partName,
          colorId: p.colorId,
          colorName: p.colorName,
          colorHex: p.colorHex,
          confidence: p.confidence,
          imageUrl: p.imageUrl,
          source: (p as any).source,
        })),
      }).catch(() => {});
    }
  }, []);

  const handleAddToInventory = async () => {
    setShowQuantityModal(false);
    setIsAdding(true);
    try {
      await addItem(selected.partNum, selected.colorId, quantity, selected.colorName, selected.colorHex);
      Alert.alert(
        'Added!',
        `${quantity}× ${selected.partName} saved to inventory`,
        [
          { text: 'Scan Another', onPress: () => { setQuantity(1); setSelectedIndex(0); navigation.navigate('ScanScreen'); } },
          { text: 'Done', onPress: () => navigation.navigate('ScanScreen') },
        ]
      );
    } catch (error: any) {
      Alert.alert('Error', error?.message || 'Failed to add item to inventory');
    } finally {
      setIsAdding(false);
    }
  };

  const confidencePct = Math.round(selected.confidence * 100);
  const confidenceColor = confidencePct >= 80 ? C.green : confidencePct >= 50 ? '#F59E0B' : C.red;

  return (
    <View style={styles.root}>
      <StatusBar barStyle="dark-content" />
      <LoadingOverlay visible={isAdding} message="Adding to inventory…" />

      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity style={styles.backBtn} onPress={() => navigation.navigate('ScanScreen')}>
          <Ionicons name="arrow-back" size={22} color={C.text} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Scan Result</Text>
        <View style={{ width: 38 }} />
      </View>

      <ScrollView style={styles.scroll} contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>

        {/* Low-confidence warning banner */}
        {isLowConfidence && (
          <View style={styles.uncertainBanner}>
            <Ionicons name="warning-outline" size={18} color="#92400E" />
            <Text style={styles.uncertainText}>
              Low confidence — the model isn't sure. Try better lighting or a cleaner background.
            </Text>
          </View>
        )}

        {/* Main result card */}
        <View style={styles.resultCard}>
          {/* Image */}
          <View style={styles.imageWrap}>
            {selected.imageUrl && !imageError
              ? (
                <Image
                  source={{ uri: selected.imageUrl }}
                  style={styles.partImage}
                  resizeMode="contain"
                  onError={() => setImageError(true)}
                />
              )
              : (
                <View style={styles.imagePlaceholder}>
                  <Ionicons name="cube-outline" size={48} color={C.textMuted} />
                </View>
              )
            }
          </View>

          {/* Part info */}
          <Text style={styles.partName}>{selected.partName}</Text>
          <Text style={styles.partNum}>Part #{selected.partNum}</Text>

          {/* Color + confidence row */}
          <View style={styles.metaRow}>
            <View style={styles.colorChip}>
              <View style={[styles.colorDot, { backgroundColor: selected.colorHex || '#ccc' }]} />
              <Text style={styles.colorLabel}>{selected.colorName}</Text>
            </View>

            <View style={[styles.confidenceBadge, { backgroundColor: confidenceColor + '18' }]}>
              <Ionicons name={isLowConfidence ? 'help-circle' : 'checkmark-circle'} size={14} color={confidenceColor} />
              <Text style={[styles.confidenceText, { color: confidenceColor }]}>
                {confidencePct}% match
              </Text>
            </View>
          </View>

          {/* Recognition source badge */}
          {(selected as any).source && (
            <View style={styles.sourceBadge}>
              <Ionicons
                name={
                  (selected as any).source?.includes('video') ? 'videocam' :
                  (selected as any).source?.includes('brickognize') ? 'flash' :
                  (selected as any).source?.includes('gemini') ? 'sparkles' : 'hardware-chip'
                }
                size={12}
                color="#6366F1"
              />
              <Text style={styles.sourceText}>
                {(selected as any).source?.includes('video') ? 'Video Scan' :
                 (selected as any).source === 'brickognize+gemini' ? 'Brickognize + AI Verified' :
                 (selected as any).source === 'brickognize' ? 'Brickognize' :
                 (selected as any).source === 'gemini' ? 'AI Vision' : 'Local Model'}
              </Text>
            </View>
          )}

          {/* Video scan stats */}
          {scanMode === 'video' && framesAnalyzed != null && (
            <View style={styles.videoStatsBadge}>
              <Ionicons name="analytics-outline" size={12} color="#059669" />
              <Text style={styles.videoStatsText}>
                {framesAnalyzed} frames · {Math.round((agreementScore || 0) * 100)}% agreement
              </Text>
            </View>
          )}
        </View>

        {/* Substitute suggestions */}
        <SubstituteSuggestions partNum={selected.partNum} />

        {/* Alternative matches */}
        {/* Feedback row */}
        <FeedbackRow
          scanId={scanId}
          predictedPartNum={selected.partNum}
          confidence={selected.confidence}
          source={(selected as any).source}
          colorId={selected.colorId}
        />
        {predictions.length > 1 && (
          <>
            <Text style={styles.sectionLabel}>OTHER POSSIBILITIES</Text>
            <View style={styles.altCard}>
              {predictions.slice(1, 4).map((item, i) => {
                const idx = i + 1;
                const pct = Math.round(item.confidence * 100);
                return (
                  <TouchableOpacity
                    key={idx}
                    style={[
                      styles.altRow,
                      i < predictions.slice(1, 4).length - 1 && styles.altRowBorder,
                      selectedIndex === idx && styles.altRowSelected,
                    ]}
                    onPress={() => { setSelectedIndex(idx); setImageError(false); }}
                    activeOpacity={0.75}
                  >
                    <AltImage imageUrl={item.imageUrl} />
                    <View style={styles.altInfo}>
                      <Text style={styles.altName} numberOfLines={1}>{item.partName}</Text>
                      <Text style={styles.altPartNum}>#{item.partNum}</Text>
                    </View>
                    <View style={styles.altRight}>
                      <Text style={styles.altPct}>{pct}%</Text>
                      {selectedIndex === idx && <Ionicons name="checkmark-circle" size={16} color={C.red} style={{ marginTop: 2 }} />}
                    </View>
                  </TouchableOpacity>
                );
              })}
            </View>
          </>
        )}

        {/* CTA buttons */}
        <TouchableOpacity style={styles.primaryBtn} onPress={() => setShowQuantityModal(true)} activeOpacity={0.85}>
          <Ionicons name="add-circle-outline" size={20} color={C.white} style={{ marginRight: 8 }} />
          <Text style={styles.primaryBtnText}>Add to Inventory</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.detailBtn}
          onPress={() => navigation.navigate('PartDetailScreen', {
            partNum: selected.partNum,
            partName: selected.partName,
            imageUrl: selected.imageUrl,
            colorId: selected.colorId,
            colorName: selected.colorName,
            colorHex: selected.colorHex,
            confidence: selected.confidence,
          })}
          activeOpacity={0.85}
        >
          <Ionicons name="information-circle-outline" size={20} color={C.red} style={{ marginRight: 8 }} />
          <Text style={styles.detailBtnText}>View Part Details</Text>
          <Ionicons name="chevron-forward" size={16} color={C.red} style={{ marginLeft: 4 }} />
        </TouchableOpacity>

        <TouchableOpacity style={styles.secondaryBtn} onPress={() => navigation.navigate('ScanScreen')} activeOpacity={0.85}>
          <Ionicons name="camera-outline" size={20} color={C.text} style={{ marginRight: 8 }} />
          <Text style={styles.secondaryBtnText}>That's Not Right — Rescan</Text>
        </TouchableOpacity>

      </ScrollView>

      {/* Quantity Modal */}
      <Modal visible={showQuantityModal} transparent animationType="slide">
        <TouchableOpacity style={styles.modalBg} activeOpacity={1} onPress={() => setShowQuantityModal(false)}>
          <TouchableOpacity activeOpacity={1} style={styles.qtySheet} onPress={() => {}}>
            <View style={styles.sheetHandle} />

            <Text style={styles.qtyTitle}>How many pieces?</Text>
            <Text style={styles.qtyPartLabel}>{selected.partName}</Text>

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

            <TouchableOpacity style={[styles.secondaryBtn, { marginTop: 0 }]} onPress={() => setShowQuantityModal(false)}>
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
  scrollContent: { padding: S.md, gap: 12, paddingBottom: 40 },

  uncertainBanner: {
    flexDirection: 'row', alignItems: 'flex-start', gap: 10,
    backgroundColor: '#FEF3C7', borderRadius: R.md,
    padding: 12, borderWidth: 1, borderColor: '#FCD34D',
  },
  uncertainText: { flex: 1, fontSize: 13, color: '#92400E', lineHeight: 18 },

  // Main card
  resultCard: {
    backgroundColor: C.white, borderRadius: R.xl,
    padding: S.lg, alignItems: 'center',
    ...shadow(2),
  },
  imageWrap: {
    width: 160, height: 160, borderRadius: R.lg,
    backgroundColor: C.bg, alignItems: 'center', justifyContent: 'center',
    marginBottom: S.md,
  },
  partImage: { width: 140, height: 140 },
  imagePlaceholder: { alignItems: 'center', justifyContent: 'center' },
  partName: { fontSize: 20, fontWeight: '800', color: C.text, textAlign: 'center', marginBottom: 4 },
  partNum: { fontSize: 13, color: C.textMuted, marginBottom: S.md, fontFamily: Platform.OS === 'ios' ? 'Courier New' : 'monospace' },

  metaRow: { flexDirection: 'row', gap: 10, alignItems: 'center', flexWrap: 'wrap', justifyContent: 'center' },
  colorChip: {
    flexDirection: 'row', alignItems: 'center', gap: 6,
    backgroundColor: C.bg, paddingHorizontal: 12, paddingVertical: 6,
    borderRadius: R.full, borderWidth: 1, borderColor: C.border,
  },
  colorDot: { width: 12, height: 12, borderRadius: 6, borderWidth: 1, borderColor: 'rgba(0,0,0,0.1)' },
  colorLabel: { fontSize: 13, fontWeight: '600', color: C.textSub },
  confidenceBadge: {
    flexDirection: 'row', alignItems: 'center', gap: 5,
    paddingHorizontal: 12, paddingVertical: 6, borderRadius: R.full,
  },
  confidenceText: { fontSize: 13, fontWeight: '700' },
  sourceBadge: {
    flexDirection: 'row', alignItems: 'center', gap: 5,
    backgroundColor: '#EEF2FF', paddingHorizontal: 12, paddingVertical: 5,
    borderRadius: R.full, marginTop: 8,
  },
  sourceText: { fontSize: 11, fontWeight: '600', color: '#6366F1' },
  videoStatsBadge: {
    flexDirection: 'row', alignItems: 'center', gap: 5,
    backgroundColor: '#ECFDF5', paddingHorizontal: 12, paddingVertical: 5,
    borderRadius: R.full, marginTop: 4,
  },
  videoStatsText: { fontSize: 11, fontWeight: '600', color: '#059669' },

  // Alternatives
  sectionLabel: {
    fontSize: 11, fontWeight: '700', color: C.textMuted,
    letterSpacing: 0.8, paddingHorizontal: 2,
  },
  altCard: {
    backgroundColor: C.white, borderRadius: R.xl, overflow: 'hidden', ...shadow(1),
  },
  altRow: {
    flexDirection: 'row', alignItems: 'center', padding: S.md, gap: S.md,
  },
  altRowBorder: { borderBottomWidth: 1, borderBottomColor: C.border },
  altRowSelected: { backgroundColor: '#FFF5F5' },
  altImg: { width: 48, height: 48, borderRadius: 8 },
  altImgPlaceholder: { backgroundColor: C.bg, alignItems: 'center', justifyContent: 'center' },
  altInfo: { flex: 1 },
  altName: { fontSize: 14, fontWeight: '600', color: C.text },
  altPartNum: { fontSize: 11, color: C.textMuted, marginTop: 2 },
  altRight: { alignItems: 'flex-end' },
  altPct: { fontSize: 13, fontWeight: '700', color: C.textSub },

  // Buttons
  primaryBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    backgroundColor: C.red, borderRadius: R.md,
    paddingVertical: 15, marginTop: 4,
  },
  primaryBtnText: { color: C.white, fontSize: 16, fontWeight: '700' },
  detailBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    backgroundColor: '#FFF5F5', borderRadius: R.md,
    paddingVertical: 14, marginTop: 4,
    borderWidth: 1.5, borderColor: C.red + '30',
  },
  detailBtnText: { color: C.red, fontSize: 15, fontWeight: '600' },
  secondaryBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    backgroundColor: C.white, borderRadius: R.md,
    paddingVertical: 14, marginTop: 4,
    borderWidth: 1.5, borderColor: C.border,
    ...shadow(1),
  },
  secondaryBtnText: { color: C.text, fontSize: 15, fontWeight: '600' },

  // Qty modal
  modalBg: { flex: 1, backgroundColor: C.overlay, justifyContent: 'flex-end' },
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
    backgroundColor: C.white, alignItems: 'center', justifyContent: 'center',
    ...shadow(1),
  },
  qtyDisplay: { flex: 1, alignItems: 'center' },
  qtyNumber: { fontSize: 40, fontWeight: '900', color: C.red },
});
