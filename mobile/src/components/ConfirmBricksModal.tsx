/**
 * ConfirmBricksModal — shown when the user taps "Done" at the end of a
 * continuous-scan session. Lets them review each detected brick, edit
 * quantities per part, and commit them all to inventory in one shot.
 *
 * Input is the locked ContinuousBrickTrack[]; output is a list of
 * {partNum, quantity} entries.
 */
import React, { useMemo, useState } from 'react';
import {
  Modal, View, Text, TouchableOpacity, ScrollView, StyleSheet,
  TextInput, Platform,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { C, R, S, shadow } from '@/constants/theme';
import type { ContinuousBrickTrack } from './DetectedBricksDrawer';

export type ConfirmedBrickEntry = {
  partNum: string;
  partName: string;
  colorName?: string;
  quantity: number;
  confidence: number;
};

interface Props {
  visible: boolean;
  tracks: ContinuousBrickTrack[];
  onCancel: () => void;
  onConfirm: (entries: ConfirmedBrickEntry[]) => void | Promise<void>;
}

export const ConfirmBricksModal: React.FC<Props> = ({
  visible, tracks, onCancel, onConfirm,
}) => {
  // Group tracks by (partNum, colorName) so two physical 3001 reds appear
  // as one row with quantity 2 rather than two separate rows.
  const aggregated = useMemo<ConfirmedBrickEntry[]>(() => {
    const byKey = new Map<string, ConfirmedBrickEntry>();
    for (const t of tracks) {
      const key = `${t.partNum}|${t.colorName ?? ''}`;
      const existing = byKey.get(key);
      if (existing) {
        existing.quantity += 1;
        existing.confidence = Math.max(existing.confidence, t.fusedConfidence);
      } else {
        byKey.set(key, {
          partNum: t.partNum,
          partName: t.partName,
          colorName: t.colorName,
          quantity: 1,
          confidence: t.fusedConfidence,
        });
      }
    }
    return Array.from(byKey.values());
  }, [tracks]);

  const [edited, setEdited] = useState<ConfirmedBrickEntry[]>(aggregated);
  const [submitting, setSubmitting] = useState(false);

  // Reset local state whenever the input changes (e.g. new modal open)
  React.useEffect(() => { setEdited(aggregated); }, [aggregated]);

  const totalBricks = edited.reduce((s, e) => s + e.quantity, 0);

  const updateQty = (idx: number, delta: number) => {
    setEdited(prev => prev.map((e, i) =>
      i === idx ? { ...e, quantity: Math.max(0, e.quantity + delta) } : e
    ));
  };

  const setQtyInput = (idx: number, text: string) => {
    const n = parseInt(text, 10);
    if (!Number.isNaN(n)) updateQty(idx, n - edited[idx].quantity);
  };

  const handleConfirm = async () => {
    setSubmitting(true);
    try {
      const nonZero = edited.filter(e => e.quantity > 0);
      await onConfirm(nonZero);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Modal
      visible={visible}
      transparent
      animationType="slide"
      onRequestClose={onCancel}
    >
      <View style={styles.backdrop}>
        <View style={[styles.sheet, shadow(3)]}>
          <View style={styles.handle} />
          <View style={styles.header}>
            <View style={{ flex: 1 }}>
              <Text style={styles.title}>Confirm bricks</Text>
              <Text style={styles.subtitle}>
                {edited.length} part{edited.length === 1 ? '' : 's'} · {totalBricks} brick{totalBricks === 1 ? '' : 's'} total
              </Text>
            </View>
            <TouchableOpacity onPress={onCancel} hitSlop={10}>
              <Ionicons name="close" size={26} color={C.textSub} />
            </TouchableOpacity>
          </View>

          <ScrollView style={styles.list} showsVerticalScrollIndicator={false}>
            {edited.map((entry, i) => (
              <View key={`${entry.partNum}_${entry.colorName ?? ''}`} style={styles.row}>
                <View style={styles.rowInfo}>
                  <Text style={styles.partNum}>#{entry.partNum}</Text>
                  <Text style={styles.partName} numberOfLines={1}>{entry.partName}</Text>
                  <View style={styles.rowMetaRow}>
                    {entry.colorName ? (
                      <Text style={styles.metaText}>{entry.colorName}</Text>
                    ) : null}
                    <Text style={styles.metaText}>
                      {Math.round(entry.confidence * 100)}% conf
                    </Text>
                  </View>
                </View>

                <View style={styles.qtyGroup}>
                  <TouchableOpacity
                    style={styles.qtyBtn}
                    onPress={() => updateQty(i, -1)}
                    disabled={entry.quantity <= 0}
                  >
                    <Ionicons name="remove" size={18} color={C.text} />
                  </TouchableOpacity>
                  <TextInput
                    style={styles.qtyInput}
                    value={String(entry.quantity)}
                    onChangeText={t => setQtyInput(i, t)}
                    keyboardType="number-pad"
                    returnKeyType="done"
                    selectTextOnFocus
                  />
                  <TouchableOpacity
                    style={styles.qtyBtn}
                    onPress={() => updateQty(i, 1)}
                  >
                    <Ionicons name="add" size={18} color={C.text} />
                  </TouchableOpacity>
                </View>
              </View>
            ))}
            {edited.length === 0 && (
              <Text style={styles.empty}>
                No locked bricks yet — close this modal and keep scanning.
              </Text>
            )}
          </ScrollView>

          <View style={styles.footer}>
            <TouchableOpacity style={styles.cancelBtn} onPress={onCancel}>
              <Text style={styles.cancelText}>Cancel</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[
                styles.confirmBtn,
                (totalBricks === 0 || submitting) && styles.confirmBtnDisabled,
              ]}
              disabled={totalBricks === 0 || submitting}
              onPress={handleConfirm}
            >
              <Ionicons name="cube" size={18} color={C.white} style={{ marginRight: 6 }} />
              <Text style={styles.confirmText}>
                {submitting ? 'Adding…' : `Add ${totalBricks} brick${totalBricks === 1 ? '' : 's'}`}
              </Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  backdrop: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.45)',
    justifyContent: 'flex-end',
  },
  sheet: {
    backgroundColor: C.white,
    borderTopLeftRadius: 18,
    borderTopRightRadius: 18,
    maxHeight: '85%',
    paddingBottom: Platform.OS === 'ios' ? 32 : 18,
  },
  handle: {
    width: 42, height: 4,
    borderRadius: 2,
    backgroundColor: C.border,
    alignSelf: 'center',
    marginTop: 8,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: S.lg,
    paddingTop: S.md,
    paddingBottom: S.sm,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: C.border,
  },
  title: { fontSize: 20, fontWeight: '700', color: C.text },
  subtitle: { fontSize: 13, color: C.textSub, marginTop: 2 },

  list: { paddingHorizontal: S.md, paddingVertical: S.sm },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: C.border,
  },
  rowInfo: { flex: 1, marginRight: S.sm },
  partNum: { fontSize: 14, fontWeight: '700', color: C.text },
  partName: { fontSize: 12, color: C.textSub, marginTop: 1 },
  rowMetaRow: { flexDirection: 'row', marginTop: 3, flexWrap: 'wrap' },
  metaText: { fontSize: 10, color: C.textMuted, marginRight: 8 },

  qtyGroup: {
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: R.sm,
    overflow: 'hidden',
  },
  qtyBtn: {
    width: 32, height: 32,
    alignItems: 'center', justifyContent: 'center',
    backgroundColor: C.cardAlt,
  },
  qtyInput: {
    width: 40,
    textAlign: 'center',
    fontSize: 14,
    fontWeight: '600',
    color: C.text,
    paddingVertical: 0,
    paddingHorizontal: 0,
  },

  empty: { textAlign: 'center', color: C.textMuted, padding: S.lg },

  footer: {
    flexDirection: 'row',
    paddingHorizontal: S.lg,
    paddingTop: S.md,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: C.border,
  },
  cancelBtn: {
    paddingVertical: 12,
    paddingHorizontal: 18,
    borderRadius: R.sm,
    marginRight: S.sm,
  },
  cancelText: { color: C.textSub, fontWeight: '600', fontSize: 14 },
  confirmBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: C.red,
    paddingVertical: 12,
    borderRadius: R.sm,
  },
  confirmBtnDisabled: { backgroundColor: 'rgba(227, 0, 11, 0.4)' },
  confirmText: { color: C.white, fontWeight: '700', fontSize: 14 },
});
