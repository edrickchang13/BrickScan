/**
 * FeedbackRow — three-way "was this right?" UI shown below the primary
 * prediction card. Part of the active-learning flywheel.
 *
 * Three tap paths, each mapped to a backend feedback_type:
 *   [✓ Yes, it's this]       → top_correct         (rank 0)
 *   [Right brick, wrong colour] → partially_correct
 *   [None of these match →]  → none_correct        (rank -1, opens search)
 *
 * Alternative-card taps (2nd / 3rd predictions on ScanResultScreen) go through
 * `submitAlternativeFeedback()` directly from feedbackApi — not through this
 * component.
 *
 * Integration on ScanResultScreen.tsx:
 *   const scanId = useRef(`scan_${Date.now()}_${Math.random().toString(36).substr(2,9)}`).current;
 *   const scanStartMs = useRef(Date.now()).current;
 *   …
 *   <FeedbackRow
 *     scanId={scanId}
 *     predictedPartNum={top.partNum}
 *     confidence={top.confidence}
 *     source={top.source ?? 'unknown'}
 *     colorId={top.colorId}
 *     predictionsShown={top5}              // pass the full top-5
 *     scanStartMs={scanStartMs}
 *     imageBase64={capturedBase64}         // optional — enables labelled image saving
 *   />
 */

import React, { useCallback, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  FlatList,
  Image,
  Keyboard,
  KeyboardAvoidingView,
  Modal,
  Platform,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  TouchableWithoutFeedback,
  View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import {
  submitFeedback,
  type PredictionShown,
} from '@/services/feedbackApi';
import { getApiBaseUrl } from '@/constants/config';

// Minimal theme constants — self-contained so this component has no deps.
const C = {
  text:     '#111827',
  textSub:  '#6B7280',
  textMuted:'#9CA3AF',
  bg:       '#F9FAFB',
  white:    '#FFFFFF',
  border:   '#E5E7EB',
  green:    '#10B981',
  greenBg:  '#ECFDF5',
  amber:    '#D97706',
  amberBg:  '#FEF3C7',
  red:      '#EF4444',
  redBg:    '#FEF2F2',
  indigo:   '#6366F1',
  overlay:  'rgba(0,0,0,0.4)',
};
const R = { md: 12, lg: 16, full: 999 };
const S = { sm: 8, md: 16, lg: 20 };

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

export interface FeedbackRowProps {
  scanId: string;
  predictedPartNum: string;
  confidence: number;
  source: string;
  colorId?: string;
  /** Full top-5 shown on screen — powers confusion analysis on the backend. */
  predictionsShown?: PredictionShown[];
  /** Epoch ms when the scan result appeared, for time_to_confirm_ms. */
  scanStartMs?: number;
  /** Optional: base64 JPEG of the scan. Enables labelled-image saving for retraining. */
  imageBase64?: string;
}

type FeedbackState =
  | 'idle'
  | 'submitting'
  | 'done_top_correct'
  | 'done_partially_correct'
  | 'done_none_correct'
  | 'await_search'
  | 'await_color';

const COMMON_LEGO_COLORS: LegoColor[] = [
  { id: 0,  name: 'Black',             hex: '#05131D' },
  { id: 1,  name: 'Blue',              hex: '#0055BF' },
  { id: 2,  name: 'Green',             hex: '#257A3E' },
  { id: 4,  name: 'Red',               hex: '#C91A09' },
  { id: 6,  name: 'Brown',             hex: '#583927' },
  { id: 7,  name: 'Light Gray',        hex: '#9BA19D' },
  { id: 14, name: 'Yellow',            hex: '#F2CD37' },
  { id: 15, name: 'White',             hex: '#FFFFFF' },
  { id: 25, name: 'Orange',            hex: '#FE8A18' },
  { id: 70, name: 'Reddish Brown',     hex: '#582A12' },
  { id: 71, name: 'Light Bluish Gray', hex: '#A0A5A9' },
  { id: 72, name: 'Dark Bluish Gray',  hex: '#6C6E68' },
];

/**
 * Tiny part thumbnail with a 3-tier source fallback:
 *   1. BrickLink CDN (color 11 = black, neutral default that exists for most parts)
 *   2. Rebrickable CDN (different image set — covers some parts BrickLink doesn't)
 *   3. Cube icon placeholder
 *
 * The fallback chain is needed because both CDNs have gaps, especially for
 * obscure / Duplo / printed-variant parts.
 */
const PartThumbnail: React.FC<{ partNum: string }> = ({ partNum }) => {
  const [tier, setTier] = useState<0 | 1 | 2>(0);

  const advance = () => {
    if (__DEV__) console.log(`[PartThumbnail] ${partNum} tier ${tier} failed → advancing`);
    setTier(t => (t < 2 ? ((t + 1) as 0 | 1 | 2) : t));
  };

  if (tier === 2) {
    return (
      <View style={styles.resultIcon}>
        <Ionicons name="cube-outline" size={20} color={C.textMuted} />
      </View>
    );
  }

  const url = tier === 0
    ? `https://img.bricklink.com/ItemImage/PN/11/${encodeURIComponent(partNum)}.png`
    : `https://cdn.rebrickable.com/media/parts/elements/${encodeURIComponent(partNum)}.jpg`;

  return (
    <Image
      key={tier}
      source={{ uri: url }}
      style={styles.resultThumb}
      resizeMode="contain"
      onError={advance}
    />
  );
};


export const FeedbackRow: React.FC<FeedbackRowProps> = ({
  scanId,
  predictedPartNum,
  confidence,
  source,
  colorId,
  predictionsShown,
  scanStartMs,
  imageBase64,
}) => {
  const [state, setState] = useState<FeedbackState>('idle');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<PartSearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [selectedColorId, setSelectedColorId] = useState<number | null>(null);
  const searchTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Tracks the most-recent query so out-of-order responses get dropped.
  // Without this, fast typing causes the response from "30" to overwrite
  // the response from "3001" if the network reorders them.
  const latestQueryRef = useRef<string>('');

  const elapsedMs = useCallback(
    (): number | undefined => scanStartMs ? Math.max(0, Date.now() - scanStartMs) : undefined,
    [scanStartMs],
  );

  // ── ✓ Yes, top pick is right ───────────────────────────────────────────────
  const handleTopCorrect = useCallback(async () => {
    setState('submitting');
    try {
      await submitFeedback({
        scanId,
        predictedPartNum,
        correctPartNum: predictedPartNum,
        confidence,
        source,
        correctColorId: colorId,
        feedbackType: 'top_correct',
        correctRank: 0,
        predictionsShown,
        timeToConfirmMs: elapsedMs(),
      });
      setState('done_top_correct');
    } catch {
      setState('idle');
    }
  }, [scanId, predictedPartNum, confidence, source, colorId, predictionsShown, elapsedMs]);

  // ── Right brick, wrong colour ───────────────────────────────────────────────
  const handleWrongColor = useCallback(() => {
    setSelectedColorId(null);
    setState('await_color');
  }, []);

  const submitWrongColor = useCallback(async (chosenColorId: number | null) => {
    setState('submitting');
    try {
      const chosen = chosenColorId !== null
        ? COMMON_LEGO_COLORS.find(c => c.id === chosenColorId)
        : null;
      await submitFeedback({
        scanId,
        predictedPartNum,
        correctPartNum: predictedPartNum,
        confidence,
        source,
        correctColorId: chosenColorId !== null ? chosenColorId.toString() : undefined,
        correctColorName: chosen?.name,
        feedbackType: 'partially_correct',
        correctRank: 0,
        predictionsShown,
        timeToConfirmMs: elapsedMs(),
        imageBase64,
      });
      setState('done_partially_correct');
    } catch {
      Alert.alert('Error', 'Could not submit correction. Please try again.');
      setState('idle');
    }
  }, [scanId, predictedPartNum, confidence, source, predictionsShown, elapsedMs, imageBase64]);

  // ── None of these match (opens search) ──────────────────────────────────────
  const handleNoneMatch = useCallback(() => {
    setSearchQuery('');
    setSearchResults([]);
    setState('await_search');
  }, []);

  const handleSearchChange = useCallback((q: string) => {
    setSearchQuery(q);
    latestQueryRef.current = q;
    if (searchTimer.current) clearTimeout(searchTimer.current);
    const trimmed = q.trim();
    if (!trimmed) {
      setSearchResults([]);
      setSearching(false);
      return;
    }
    setSearching(true);
    searchTimer.current = setTimeout(async () => {
      const queryAtRequestTime = trimmed;
      try {
        // Use the same EXPO_PUBLIC_API_URL fallback the rest of the app uses
        // (NativeModules.SourceCode.scriptURL is empty under SDK 55 + RN 0.83
        // + expo-dev-client, so the old host-detection path silently falls
        // through to localhost — which on the phone means the phone itself.
        // See mobile/DEVELOPMENT.md for the full story.)
        const base = getApiBaseUrl();
        const url = `${base}/api/local-inventory/parts/search?q=${encodeURIComponent(queryAtRequestTime)}&limit=15`;
        if (__DEV__) console.log('[FeedbackRow] search →', url);
        const res = await fetch(url);
        if (!res.ok) {
          console.warn('[FeedbackRow] search HTTP', res.status, 'for', url);
        }
        // Drop the response if the user has typed more characters since this
        // request was fired — prevents older results from clobbering newer ones.
        if (latestQueryRef.current.trim() !== queryAtRequestTime) return;
        const data: PartSearchResult[] = await res.json();
        setSearchResults(Array.isArray(data) ? data : []);
        setSearching(false);
      } catch (e) {
        if (latestQueryRef.current.trim() === queryAtRequestTime) {
          setSearchResults([]);
          setSearching(false);
        }
      }
    }, 250);
  }, []);

  const handleSearchResultTap = useCallback(async (chosen: PartSearchResult) => {
    setState('submitting');
    try {
      await submitFeedback({
        scanId,
        predictedPartNum,
        correctPartNum: chosen.part_num,
        confidence,
        source,
        feedbackType: 'none_correct',
        correctRank: -1,
        predictionsShown,
        timeToConfirmMs: elapsedMs(),
        imageBase64,
      });
      setState('done_none_correct');
    } catch {
      Alert.alert('Error', 'Could not submit correction. Please try again.');
      setState('idle');
    }
  }, [scanId, predictedPartNum, confidence, source, predictionsShown, elapsedMs, imageBase64]);

  // ── Render: done / submitting states ───────────────────────────────────────
  if (state === 'done_top_correct') {
    return (
      <View style={[styles.thanksBanner, { backgroundColor: C.greenBg, borderColor: '#A7F3D0' }]}>
        <Ionicons name="checkmark-circle" size={16} color={C.green} />
        <Text style={[styles.thanksText, { color: C.green }]}>Confirmed — thanks!</Text>
      </View>
    );
  }
  if (state === 'done_partially_correct' || state === 'done_none_correct') {
    return (
      <View style={[styles.thanksBanner, { backgroundColor: '#EEF2FF', borderColor: '#C7D2FE' }]}>
        <Ionicons name="heart" size={16} color={C.indigo} />
        <Text style={[styles.thanksText, { color: C.indigo }]}>
          Thanks — BrickScan just got smarter.
        </Text>
      </View>
    );
  }
  if (state === 'submitting') {
    return (
      <View style={styles.row}>
        <ActivityIndicator size="small" color={C.textMuted} />
        <Text style={[styles.label, { marginLeft: 10 }]}>Saving…</Text>
      </View>
    );
  }

  // ── Render: idle three-way UI ──────────────────────────────────────────────
  return (
    <>
      <View style={styles.wrap}>
        <Text style={styles.label}>Is this right?</Text>

        <TouchableOpacity
          style={styles.yesBtn}
          onPress={handleTopCorrect}
          activeOpacity={0.8}
        >
          <Ionicons name="checkmark" size={18} color={C.green} />
          <Text style={[styles.yesText]}>Yes, top pick is correct</Text>
        </TouchableOpacity>

        <View style={styles.secondaryBtnRow}>
          <TouchableOpacity style={styles.amberBtn} onPress={handleWrongColor} activeOpacity={0.8}>
            <Ionicons name="color-palette-outline" size={15} color={C.amber} />
            <Text style={[styles.secondaryText, { color: C.amber }]}>Wrong colour</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.redBtn} onPress={handleNoneMatch} activeOpacity={0.8}>
            <Ionicons name="search" size={15} color={C.red} />
            <Text style={[styles.secondaryText, { color: C.red }]}>None of these</Text>
          </TouchableOpacity>
        </View>

        <Text style={styles.hint}>
          Tip: if a different option below is correct, just tap that one.
        </Text>
      </View>

      {/* Search modal for "none of these" — KeyboardAvoidingView lifts the sheet
          above the keyboard so the search input + results stay visible. */}
      <Modal
        visible={state === 'await_search'}
        transparent
        animationType="slide"
        onRequestClose={() => { Keyboard.dismiss(); setState('idle'); }}
      >
        <KeyboardAvoidingView
          style={styles.kbAvoidWrap}
          behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        >
          <TouchableWithoutFeedback onPress={() => { Keyboard.dismiss(); setState('idle'); }}>
            <View style={styles.modalBg}>
              <TouchableWithoutFeedback>
                <View style={styles.sheet}>
                  <View style={styles.sheetHandle} />
                  <Text style={styles.sheetTitle}>What part is it actually?</Text>
                  <Text style={styles.sheetSub}>
                    Search by part number or name to find the correct part.
                  </Text>

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

                  <FlatList
                    data={searchResults}
                    keyExtractor={(item) => item.part_num}
                    style={styles.resultsList}
                    keyboardShouldPersistTaps="handled"
                    keyboardDismissMode="on-drag"
                    ListEmptyComponent={
                      searchQuery.length > 1 && !searching ? (
                        <Text style={styles.noResults}>No parts found for "{searchQuery}"</Text>
                      ) : null
                    }
                    renderItem={({ item }) => (
                      <TouchableOpacity
                        style={styles.resultRow}
                        onPress={() => handleSearchResultTap(item)}
                        activeOpacity={0.75}
                      >
                        <PartThumbnail partNum={item.part_num} />
                        <View style={styles.resultInfo}>
                          <Text style={styles.resultName} numberOfLines={2}>{item.part_name}</Text>
                          <Text style={styles.resultNum}>#{item.part_num}</Text>
                        </View>
                        <Ionicons name="chevron-forward" size={16} color={C.textMuted} />
                      </TouchableOpacity>
                    )}
                  />

                  <TouchableOpacity
                    style={styles.cancelBtn}
                    onPress={() => { Keyboard.dismiss(); setState('idle'); }}
                  >
                    <Text style={styles.cancelBtnText}>Cancel</Text>
                  </TouchableOpacity>
                </View>
              </TouchableWithoutFeedback>
            </View>
          </TouchableWithoutFeedback>
        </KeyboardAvoidingView>
      </Modal>

      {/* Colour picker for "wrong colour" */}
      <Modal visible={state === 'await_color'} transparent animationType="fade">
        <TouchableOpacity style={styles.modalBg} activeOpacity={1} onPress={() => setState('idle')}>
          <TouchableOpacity activeOpacity={1} style={styles.colorSheet}>
            <View style={styles.sheetHandle} />
            <Text style={styles.sheetTitle}>Which colour is it actually?</Text>
            <Text style={styles.sheetSub}>
              Tap the correct colour. The brick identity stays the same.
            </Text>

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
                    selectedColorId === item.id && styles.colorSwatchSelected,
                  ]}
                  onPress={() => setSelectedColorId(item.id)}
                  activeOpacity={0.75}
                >
                  {selectedColorId === item.id && (
                    <Ionicons name="checkmark" size={18} color={C.white} />
                  )}
                </TouchableOpacity>
              )}
            />

            {selectedColorId !== null && (
              <View style={styles.colorNameBox}>
                <Text style={styles.colorNameText}>
                  {COMMON_LEGO_COLORS.find(c => c.id === selectedColorId)?.name}
                </Text>
              </View>
            )}

            <View style={styles.colorActions}>
              <TouchableOpacity
                style={[
                  styles.colorSubmitBtn,
                  selectedColorId === null && styles.colorSubmitBtnDisabled,
                ]}
                onPress={() => selectedColorId !== null && submitWrongColor(selectedColorId)}
                disabled={selectedColorId === null}
                activeOpacity={0.8}
              >
                <Text style={styles.colorSubmitBtnText}>Confirm colour</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.colorSkipBtn}
                onPress={() => setState('idle')}
                activeOpacity={0.8}
              >
                <Text style={styles.colorSkipBtnText}>Cancel</Text>
              </TouchableOpacity>
            </View>
          </TouchableOpacity>
        </TouchableOpacity>
      </Modal>
    </>
  );
};

const styles = StyleSheet.create({
  wrap: {
    backgroundColor: C.white,
    borderRadius: R.md,
    padding: S.md,
    borderWidth: 1,
    borderColor: C.border,
    gap: S.sm,
  },
  label: { fontSize: 13, fontWeight: '700', color: C.text },

  row: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: S.md,
    backgroundColor: C.white,
    borderRadius: R.md,
    borderWidth: 1,
    borderColor: C.border,
  },

  yesBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8,
    backgroundColor: C.greenBg,
    paddingVertical: 14,
    borderRadius: R.md,
    borderWidth: 1, borderColor: '#A7F3D0',
  },
  yesText: { fontSize: 15, fontWeight: '700', color: C.green },

  secondaryBtnRow: { flexDirection: 'row', gap: S.sm },
  amberBtn: {
    flex: 1,
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6,
    backgroundColor: C.amberBg,
    paddingVertical: 11,
    borderRadius: R.md,
    borderWidth: 1, borderColor: '#FCD34D',
  },
  redBtn: {
    flex: 1,
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6,
    backgroundColor: C.redBg,
    paddingVertical: 11,
    borderRadius: R.md,
    borderWidth: 1, borderColor: '#FECACA',
  },
  secondaryText: { fontSize: 13, fontWeight: '700' },

  hint: { fontSize: 11, color: C.textMuted, textAlign: 'center', marginTop: 2 },

  thanksBanner: {
    flexDirection: 'row', alignItems: 'center', gap: 8,
    borderRadius: R.md, paddingVertical: 14, paddingHorizontal: S.md,
    borderWidth: 1,
    justifyContent: 'center',
  },
  thanksText: { fontSize: 13, fontWeight: '700' },

  // Modals
  kbAvoidWrap: { flex: 1 },
  modalBg: { flex: 1, backgroundColor: C.overlay, justifyContent: 'flex-end' },
  sheet: {
    backgroundColor: C.white,
    borderTopLeftRadius: 24, borderTopRightRadius: 24,
    padding: S.lg,
    paddingBottom: Platform.OS === 'ios' ? 24 : S.lg,
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

  resultsList: { maxHeight: 320 },
  resultRow: {
    flexDirection: 'row', alignItems: 'center', gap: S.sm,
    paddingVertical: 10,
    borderBottomWidth: 1, borderBottomColor: C.border,
  },
  resultIcon: {
    width: 48, height: 48, borderRadius: 8,
    backgroundColor: C.bg, alignItems: 'center', justifyContent: 'center',
  },
  resultThumb: {
    width: 48, height: 48, borderRadius: 8,
    backgroundColor: C.bg,
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

  colorSheet: {
    backgroundColor: C.white,
    borderTopLeftRadius: 24, borderTopRightRadius: 24,
    padding: S.lg,
    paddingBottom: Platform.OS === 'ios' ? 40 : S.lg,
    maxHeight: '85%',
  },
  colorGridRow: { justifyContent: 'space-around', marginBottom: S.md },
  colorSwatch: {
    width: 60, height: 60, borderRadius: 30,
    borderWidth: 2, borderColor: '#ccc',
    alignItems: 'center', justifyContent: 'center',
  },
  colorSwatchSelected: { borderWidth: 3, borderColor: C.text },
  colorNameBox: {
    backgroundColor: C.bg,
    borderRadius: R.md, paddingVertical: 10, paddingHorizontal: S.md,
    marginVertical: S.md, alignItems: 'center',
  },
  colorNameText: { fontSize: 14, fontWeight: '600', color: C.text },
  colorActions: { gap: S.sm, marginTop: S.md },
  colorSubmitBtn: {
    backgroundColor: C.green, paddingVertical: 14,
    borderRadius: R.md, alignItems: 'center',
  },
  colorSubmitBtnDisabled: { backgroundColor: '#A7F3D0' },
  colorSubmitBtnText: { fontSize: 15, fontWeight: '700', color: C.white },
  colorSkipBtn: {
    borderWidth: 1.5, borderColor: C.border,
    paddingVertical: 14, borderRadius: R.md, alignItems: 'center',
  },
  colorSkipBtnText: { fontSize: 15, fontWeight: '600', color: C.text },
});
