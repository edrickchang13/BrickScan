/**
 * FeedbackRow — "Was this correct?" UI shown below the primary prediction card.
 *
 * HOW TO ADD TO ScanResultScreen.tsx:
 * ------------------------------------
 * 1. Import at the top:
 *      import { FeedbackRow } from '@/components/FeedbackRow';
 *
 * 2. Generate a scan ID once when the screen mounts (add near your other state):
 *      const scanId = React.useRef(
 *        `scan_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
 *      ).current;
 *
 * 3. Drop this component right after the closing </View> of resultCard,
 *    before the "OTHER POSSIBILITIES" section:
 *
 *      <FeedbackRow
 *        scanId={scanId}
 *        predictedPartNum={selected.partNum}
 *        confidence={selected.confidence}
 *        source={selected.source ?? 'unknown'}
 *        colorId={selected.colorId?.toString()}
 *      />
 *
 * That's it — no other changes needed.
 */

import React, { useCallback, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  FlatList,
  Modal,
  Platform,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { submitFeedback } from '@/services/feedbackApi';

// Minimal theme constants — update to match your project's theme file
const C = {
  text:     '#111827',
  textSub:  '#6B7280',
  textMuted:'#9CA3AF',
  bg:       '#F9FAFB',
  white:    '#FFFFFF',
  border:   '#E5E7EB',
  green:    '#10B981',
  red:      '#EF4444',
  overlay:  'rgba(0,0,0,0.4)',
};
const R = { md: 12, lg: 16, full: 999 };
const S = { sm: 8, md: 16, lg: 20 };

// ---------------------------------------------------------------------------

interface PartSearchResult {
  part_num: string;
  part_name: string;
  category_id?: string;
}

interface LegoColor {
  id: number;
  name: string;
  hex: string;
}

interface FeedbackRowProps {
  scanId: string;
  predictedPartNum: string;
  confidence: number;
  source: string;
  colorId?: string;
}

type FeedbackState = 'idle' | 'submitting' | 'done_correct' | 'done_fixed' | 'part_selected_ask_color';

const COMMON_LEGO_COLORS: LegoColor[] = [
  { id: 0, name: 'Black', hex: '#05131D' },
  { id: 1, name: 'Blue', hex: '#0055BF' },
  { id: 2, name: 'Green', hex: '#257A3E' },
  { id: 4, name: 'Red', hex: '#C91A09' },
  { id: 6, name: 'Brown', hex: '#583927' },
  { id: 7, name: 'Light Gray', hex: '#9BA19D' },
  { id: 14, name: 'Yellow', hex: '#F2CD37' },
  { id: 15, name: 'White', hex: '#FFFFFF' },
  { id: 25, name: 'Orange', hex: '#FE8A18' },
  { id: 70, name: 'Reddish Brown', hex: '#582A12' },
  { id: 71, name: 'Light Bluish Gray', hex: '#A0A5A9' },
  { id: 72, name: 'Dark Bluish Gray', hex: '#6C6E68' },
];

export const FeedbackRow: React.FC<FeedbackRowProps> = ({
  scanId,
  predictedPartNum,
  confidence,
  source,
  colorId,
}) => {
  const [feedbackState, setFeedbackState] = useState<FeedbackState>('idle');
  const [showFixModal, setShowFixModal] = useState(false);
  const [showColorPicker, setShowColorPicker] = useState(false);
  const [searchQuery, setSearchQuery]   = useState('');
  const [searchResults, setSearchResults] = useState<PartSearchResult[]>([]);
  const [searching, setSearching]       = useState(false);
  const [selectedCorrectPart, setSelectedCorrectPart] = useState<PartSearchResult | null>(null);
  const [selectedCorrectColorId, setSelectedCorrectColorId] = useState<number | null>(null);
  const searchTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ── "Yes, correct" ─────────────────────────────────────────────────────────
  const handleConfirm = useCallback(async () => {
    setFeedbackState('submitting');
    try {
      await submitFeedback({
        scanId,
        predictedPartNum,
        correctPartNum: predictedPartNum,   // same = confirmation
        confidence,
        source,
        correctColorId: colorId,
      });
      setFeedbackState('done_correct');
    } catch {
      setFeedbackState('idle');
    }
  }, [scanId, predictedPartNum, confidence, source, colorId]);

  // ── Part search (debounced) ─────────────────────────────────────────────────
  const handleSearchChange = useCallback((q: string) => {
    setSearchQuery(q);
    if (searchTimer.current) clearTimeout(searchTimer.current);
    if (!q.trim()) { setSearchResults([]); return; }
    searchTimer.current = setTimeout(async () => {
      setSearching(true);
      try {
        const base = __DEV__
          ? `http://${(require('react-native').NativeModules.SourceCode?.scriptURL
              ? new URL(require('react-native').NativeModules.SourceCode.scriptURL).hostname
              : 'localhost')}:8000`
          : ((globalThis as any).process?.env?.EXPO_PUBLIC_API_URL ?? 'http://localhost:8000');
        const res = await fetch(
          `${base}/api/local-inventory/parts/search?q=${encodeURIComponent(q)}&limit=10`,
        );
        const data: PartSearchResult[] = await res.json();
        setSearchResults(data);
      } catch {
        setSearchResults([]);
      } finally {
        setSearching(false);
      }
    }, 350);
  }, []);

  // ── After part selected, ask about color ────────────────────────────────────
  const handlePartSelected = useCallback((correctPart: PartSearchResult) => {
    setSelectedCorrectPart(correctPart);
    setShowFixModal(false);
    setFeedbackState('part_selected_ask_color');
  }, []);

  // ── Submit correction with optional color ────────────────────────────────────
  const handleSubmitCorrectionWithColor = useCallback(
    async (correctPart: PartSearchResult, colorId: number | null) => {
      setShowColorPicker(false);
      setFeedbackState('submitting');
      try {
        const selectedColor = colorId !== null
          ? COMMON_LEGO_COLORS.find(c => c.id === colorId)
          : null;

        await submitFeedback({
          scanId,
          predictedPartNum,
          correctPartNum: correctPart.part_num,
          confidence,
          source,
          correctColorId: colorId?.toString(),
          correctColorName: selectedColor?.name,
        });
        setFeedbackState('done_fixed');
        setSelectedCorrectPart(null);
        setSelectedCorrectColorId(null);
      } catch {
        Alert.alert('Error', 'Could not submit correction. Please try again.');
        setFeedbackState('idle');
        setSelectedCorrectPart(null);
        setSelectedCorrectColorId(null);
      }
    },
    [scanId, predictedPartNum, confidence, source],
  );

  // ── Handle color selection and proceed ─────────────────────────────────────
  const handleColorSelected = useCallback((colorId: number | null) => {
    if (selectedCorrectPart) {
      handleSubmitCorrectionWithColor(selectedCorrectPart, colorId);
    }
  }, [selectedCorrectPart, handleSubmitCorrectionWithColor]);

  const handleSkipColorAndSubmit = useCallback(() => {
    if (selectedCorrectPart) {
      handleSubmitCorrectionWithColor(selectedCorrectPart, null);
    }
  }, [selectedCorrectPart, handleSubmitCorrectionWithColor]);

  // ── Render ─────────────────────────────────────────────────────────────────
  if (feedbackState === 'done_correct') {
    return (
      <View style={styles.thanksBanner}>
        <Ionicons name="checkmark-circle" size={15} color={C.green} />
        <Text style={styles.thanksText}>Got it — confirmed correct!</Text>
      </View>
    );
  }

  if (feedbackState === 'done_fixed') {
    return (
      <View style={styles.thanksBanner}>
        <Ionicons name="heart" size={15} color="#6366F1" />
        <Text style={[styles.thanksText, { color: '#6366F1' }]}>
          Thanks! This helps BrickScan improve.
        </Text>
      </View>
    );
  }

  if (feedbackState === 'submitting') {
    return (
      <View style={styles.feedbackRow}>
        <ActivityIndicator size="small" color={C.textMuted} />
        <Text style={styles.feedbackLabel}>Saving…</Text>
      </View>
    );
  }

  return (
    <>
      {/* ── Was this correct? row ── */}
      <View style={styles.feedbackRow}>
        <Text style={styles.feedbackLabel}>Was this correct?</Text>
        <View style={styles.feedbackBtns}>
          <TouchableOpacity style={styles.yesBtn} onPress={handleConfirm} activeOpacity={0.8}>
            <Ionicons name="checkmark" size={14} color={C.green} />
            <Text style={[styles.feedbackBtnText, { color: C.green }]}>Yes</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.noBtn}
            onPress={() => { setShowFixModal(true); setSearchQuery(''); setSearchResults([]); }}
            activeOpacity={0.8}
          >
            <Ionicons name="close" size={14} color={C.red} />
            <Text style={[styles.feedbackBtnText, { color: C.red }]}>No, fix it</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* ── Fix modal ── */}
      <Modal visible={showFixModal} transparent animationType="slide">
        <TouchableOpacity
          style={styles.modalBg}
          activeOpacity={1}
          onPress={() => setShowFixModal(false)}
        >
          <TouchableOpacity activeOpacity={1} style={styles.sheet}>
            <View style={styles.sheetHandle} />
            <Text style={styles.sheetTitle}>What part is it actually?</Text>
            <Text style={styles.sheetSub}>
              Search by part number or name to find the correct part.
            </Text>

            {/* Search input */}
            <View style={styles.searchBox}>
              <Ionicons name="search" size={16} color={C.textMuted} style={{ marginRight: 8 }} />
              <TextInput
                style={styles.searchInput}
                value={searchQuery}
                onChangeText={handleSearchChange}
                placeholder="e.g. 3001 or Brick 2x4"
                placeholderTextColor={C.textMuted}
                autoFocus
                returnKeyType="search"
              />
              {searching && <ActivityIndicator size="small" color={C.textMuted} />}
            </View>

            {/* Results */}
            <FlatList
              data={searchResults}
              keyExtractor={(item) => item.part_num}
              style={styles.resultsList}
              keyboardShouldPersistTaps="handled"
              ListEmptyComponent={
                searchQuery.length > 1 && !searching ? (
                  <Text style={styles.noResults}>No parts found for "{searchQuery}"</Text>
                ) : null
              }
              renderItem={({ item }) => (
                <TouchableOpacity
                  style={styles.resultRow}
                  onPress={() => handlePartSelected(item)}
                  activeOpacity={0.75}
                >
                  <View style={styles.resultIcon}>
                    <Ionicons name="cube-outline" size={18} color={C.textMuted} />
                  </View>
                  <View style={styles.resultInfo}>
                    <Text style={styles.resultName} numberOfLines={1}>{item.part_name}</Text>
                    <Text style={styles.resultNum}>#{item.part_num}</Text>
                  </View>
                  <Ionicons name="chevron-forward" size={16} color={C.textMuted} />
                </TouchableOpacity>
              )}
            />

            <TouchableOpacity
              style={styles.cancelBtn}
              onPress={() => setShowFixModal(false)}
            >
              <Text style={styles.cancelBtnText}>Cancel</Text>
            </TouchableOpacity>
          </TouchableOpacity>
        </TouchableOpacity>
      </Modal>

      {/* ── Color picker modal ── */}
      <Modal visible={feedbackState === 'part_selected_ask_color'} transparent animationType="fade">
        <TouchableOpacity
          style={styles.modalBg}
          activeOpacity={1}
          onPress={() => {
            setFeedbackState('idle');
            setSelectedCorrectPart(null);
            setSelectedCorrectColorId(null);
          }}
        >
          <TouchableOpacity activeOpacity={1} style={styles.colorSheet}>
            <View style={styles.sheetHandle} />
            <Text style={styles.sheetTitle}>Was the color also wrong?</Text>
            <Text style={styles.sheetSub}>
              Select the correct color, or skip if only the part was wrong.
            </Text>

            {/* Color swatches */}
            <FlatList
              data={COMMON_LEGO_COLORS}
              keyExtractor={(item) => item.id.toString()}
              numColumns={4}
              columnWrapperStyle={styles.colorGridRow}
              scrollEnabled={false}
              renderItem={({ item }) => (
                <TouchableOpacity
                  style={[
                    styles.colorSwatch,
                    { borderColor: item.hex, backgroundColor: item.hex },
                    selectedCorrectColorId === item.id && styles.colorSwatchSelected,
                  ]}
                  onPress={() => setSelectedCorrectColorId(item.id)}
                  activeOpacity={0.75}
                >
                  {selectedCorrectColorId === item.id && (
                    <Ionicons name="checkmark" size={16} color={C.white} />
                  )}
                </TouchableOpacity>
              )}
            />

            {/* Color name display */}
            {selectedCorrectColorId !== null && (
              <View style={styles.colorNameBox}>
                <Text style={styles.colorNameText}>
                  {COMMON_LEGO_COLORS.find(c => c.id === selectedCorrectColorId)?.name}
                </Text>
              </View>
            )}

            {/* Action buttons */}
            <View style={styles.colorActions}>
              <TouchableOpacity
                style={styles.colorSubmitBtn}
                onPress={() => handleColorSelected(selectedCorrectColorId)}
                activeOpacity={0.8}
              >
                <Text style={styles.colorSubmitBtnText}>Confirm</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.colorSkipBtn}
                onPress={handleSkipColorAndSubmit}
                activeOpacity={0.8}
              >
                <Text style={styles.colorSkipBtnText}>Skip (color was correct)</Text>
              </TouchableOpacity>
            </View>
          </TouchableOpacity>
        </TouchableOpacity>
      </Modal>
    </>
  );
};

// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  feedbackRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: C.white,
    borderRadius: R.md,
    paddingHorizontal: S.md,
    paddingVertical: 12,
    borderWidth: 1,
    borderColor: C.border,
  },
  feedbackLabel: { fontSize: 13, fontWeight: '600', color: C.text },
  feedbackBtns:  { flexDirection: 'row', gap: 10 },
  yesBtn: {
    flexDirection: 'row', alignItems: 'center', gap: 5,
    backgroundColor: '#ECFDF5',
    paddingHorizontal: 14, paddingVertical: 7,
    borderRadius: R.full,
    borderWidth: 1, borderColor: '#A7F3D0',
  },
  noBtn: {
    flexDirection: 'row', alignItems: 'center', gap: 5,
    backgroundColor: '#FEF2F2',
    paddingHorizontal: 14, paddingVertical: 7,
    borderRadius: R.full,
    borderWidth: 1, borderColor: '#FECACA',
  },
  feedbackBtnText: { fontSize: 13, fontWeight: '700' },

  thanksBanner: {
    flexDirection: 'row', alignItems: 'center', gap: 8,
    backgroundColor: C.white,
    borderRadius: R.md, padding: 12,
    borderWidth: 1, borderColor: C.border,
    justifyContent: 'center',
  },
  thanksText: { fontSize: 13, fontWeight: '600', color: C.green },

  // Modal
  modalBg: { flex: 1, backgroundColor: C.overlay, justifyContent: 'flex-end' },
  sheet: {
    backgroundColor: C.white,
    borderTopLeftRadius: 24, borderTopRightRadius: 24,
    padding: S.lg,
    paddingBottom: Platform.OS === 'ios' ? 40 : S.lg,
    maxHeight: '80%',
  },
  sheetHandle: {
    width: 40, height: 4, borderRadius: 2, backgroundColor: C.border,
    alignSelf: 'center', marginBottom: S.md,
  },
  sheetTitle: { fontSize: 18, fontWeight: '800', color: C.text, marginBottom: 4 },
  sheetSub:   { fontSize: 13, color: C.textSub, marginBottom: S.md, lineHeight: 18 },

  searchBox: {
    flexDirection: 'row', alignItems: 'center',
    backgroundColor: C.bg, borderRadius: R.md,
    paddingHorizontal: S.md, paddingVertical: 10,
    borderWidth: 1, borderColor: C.border,
    marginBottom: S.sm,
  },
  searchInput: { flex: 1, fontSize: 15, color: C.text },

  resultsList: { maxHeight: 260 },
  resultRow: {
    flexDirection: 'row', alignItems: 'center', gap: S.sm,
    paddingVertical: 12,
    borderBottomWidth: 1, borderBottomColor: C.border,
  },
  resultIcon: {
    width: 36, height: 36, borderRadius: 8,
    backgroundColor: C.bg, alignItems: 'center', justifyContent: 'center',
  },
  resultInfo: { flex: 1 },
  resultName: { fontSize: 14, fontWeight: '600', color: C.text },
  resultNum:  { fontSize: 11, color: C.textMuted, marginTop: 1 },
  noResults:  { fontSize: 13, color: C.textMuted, textAlign: 'center', paddingVertical: 20 },

  cancelBtn: {
    marginTop: S.md, paddingVertical: 14,
    borderRadius: R.md, borderWidth: 1.5, borderColor: C.border,
    alignItems: 'center',
  },
  cancelBtnText: { fontSize: 15, fontWeight: '600', color: C.text },

  // Color picker
  colorSheet: {
    backgroundColor: C.white,
    borderTopLeftRadius: 24, borderTopRightRadius: 24,
    padding: S.lg,
    paddingBottom: Platform.OS === 'ios' ? 40 : S.lg,
    maxHeight: '85%',
  },
  colorGridRow: {
    justifyContent: 'space-around',
    marginBottom: S.md,
  },
  colorSwatch: {
    width: 60,
    height: 60,
    borderRadius: 30,
    borderWidth: 2,
    borderColor: '#ccc',
    alignItems: 'center',
    justifyContent: 'center',
  },
  colorSwatchSelected: {
    borderWidth: 3,
    borderColor: C.text,
  },
  colorNameBox: {
    backgroundColor: C.bg,
    borderRadius: R.md,
    paddingVertical: 10,
    paddingHorizontal: S.md,
    marginVertical: S.md,
    alignItems: 'center',
  },
  colorNameText: { fontSize: 14, fontWeight: '600', color: C.text },
  colorActions: {
    gap: S.sm,
    marginTop: S.md,
  },
  colorSubmitBtn: {
    backgroundColor: C.green,
    paddingVertical: 14,
    borderRadius: R.md,
    alignItems: 'center',
  },
  colorSubmitBtnText: { fontSize: 15, fontWeight: '600', color: C.white },
  colorSkipBtn: {
    borderWidth: 1.5, borderColor: C.border,
    paddingVertical: 14,
    borderRadius: R.md,
    alignItems: 'center',
  },
  colorSkipBtnText: { fontSize: 15, fontWeight: '600', color: C.text },
});
