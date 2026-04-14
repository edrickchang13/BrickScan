import React, { useState } from 'react';
import {
  View, Text, TouchableOpacity, ScrollView, Alert,
  Modal, ActivityIndicator, StyleSheet, Platform, Share, Linking,
} from 'react-native';
import { useQuery } from '@tanstack/react-query';
import { Ionicons } from '@expo/vector-icons';
import { apiClient } from '@/services/api';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { SetsStackParamList, MissingPart } from '@/types';
import { C, R, S, shadow } from '@/constants/theme';

// ─── BrickLink color mapping: Rebrickable color ID → BrickLink color ID ─────
const REBRICKABLE_TO_BRICKLINK_COLOR: Record<number, number> = {
  0: 11,   // Black
  1: 1,    // Blue
  2: 5,    // Green
  3: 6,    // Dark Turquoise
  4: 4,    // Red
  5: 40,   // Dark Pink
  6: 25,   // Brown
  7: 10,   // Light Gray
  8: 28,   // Dark Gray
  9: 2,    // Light Bluish Gray (approximate)
  10: 7,   // White (approximate)
  11: 8,   // Light Yellow
  12: 9,   // Tan
  13: 12,  // Orange
  14: 3,   // Yellow
  15: 4,   // White
  17: 39,  // Light Blue
  18: 38,  // Salmon
  19: 29,  // Medium Lavender
  21: 26,  // Bright Green
  22: 30,  // Lavender
  23: 42,  // Medium Blue
  24: 24,  // Sand Yellow
  25: 22,  // Dark Purple
  26: 120, // Lime
  27: 34,  // Maersk Blue (approximate)
  28: 69,  // Dark Tan
  29: 36,  // Medium Lime
  36: 41,  // Bright Light Orange
  37: 119, // Lime Green
  38: 110, // Blue Violet
  46: 226, // Bright Light Yellow
  47: 20,  // Transparent
  70: 120, // Dark Brown
  71: 86,  // Light Bluish Gray
  72: 85,  // Dark Bluish Gray
  73: 67,  // Medium Blue
  84: 150, // Flesh
  85: 89,  // Dark Purple
  86: 91,  // Dark Flesh
  92: 150, // Flesh (alternate)
};

// Generate BrickLink Wanted List XML entirely client-side
function generateBrickLinkXml(missingParts: MissingPart[], condition: string = 'N'): string {
  const items = missingParts
    .filter(p => (p.quantityNeeded - (p.quantityHave || 0)) > 0)
    .map(p => {
      const qty = p.quantityNeeded - (p.quantityHave || 0);
      const blColor = p.colorId
        ? (REBRICKABLE_TO_BRICKLINK_COLOR[parseInt(p.colorId)] ?? 0)
        : 0;
      return `  <ITEM>
    <ITEMTYPE>P</ITEMTYPE>
    <ITEMID>${p.partNum}</ITEMID>
    <COLOR>${blColor}</COLOR>
    <MAXPRICE>-1</MAXPRICE>
    <MINQTY>${qty}</MINQTY>
    <QTYFILLED>0</QTYFILLED>
    <CONDITION>${condition}</CONDITION>
    <NOTIFY>N</NOTIFY>
  </ITEM>`;
    });

  return `<?xml version="1.0" encoding="UTF-8"?>\n<INVENTORY>\n${items.join('\n')}\n</INVENTORY>`;
}

type Props = NativeStackScreenProps<SetsStackParamList, 'BuildCheckScreen'>;

export const BuildCheckScreen: React.FC<Props> = ({ route }) => {
  const { setNum } = route.params;
  const [showBrickLinkModal, setShowBrickLinkModal] = useState(false);
  const [brickLinkXml, setBrickLinkXml] = useState('');
  const [expanded, setExpanded] = useState<{ have: boolean; missing: boolean }>({
    have: true, missing: true,
  });

  const { data: buildCheck, isLoading } = useQuery({
    queryKey: ['buildCheck', setNum],
    queryFn: () => apiClient.compareToSet(setNum),
    staleTime: 5 * 60 * 1000,
  });

  const [isBrickLinkLoading, setIsBrickLinkLoading] = useState(false);

  const handleGenerateBrickLink = () => {
    if (!buildCheck?.missingParts?.length) return;
    const xml = generateBrickLinkXml(buildCheck.missingParts, 'N');
    setBrickLinkXml(xml);
    setShowBrickLinkModal(true);
  };

  const handleShareXml = async () => {
    try {
      await Share.share({
        message: brickLinkXml,
        title: 'BrickLink Wanted List XML',
      });
    } catch (e: any) {
      // User dismissed — no-op
    }
  };

  const handleOpenBrickLink = async () => {
    // BrickLink wanted list upload page
    const url = 'https://www.bricklink.com/wantedXML.asp';
    try {
      const supported = await Linking.canOpenURL(url);
      if (supported) {
        await Linking.openURL(url);
      } else {
        Alert.alert('Cannot Open', 'Could not open BrickLink in browser.');
      }
    } catch (e) {
      Alert.alert('Error', 'Failed to open BrickLink.');
    }
  };

  if (isLoading) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color={C.red} />
        <Text style={styles.loadingText}>Checking your inventory…</Text>
      </View>
    );
  }

  if (!buildCheck) {
    return (
      <View style={styles.centered}>
        <Ionicons name="alert-circle-outline" size={48} color={C.textMuted} />
        <Text style={styles.emptyTitle}>Build check unavailable</Text>
      </View>
    );
  }

  const pct = Math.round(buildCheck.percentComplete || 0);
  const canBuild = pct >= 100;

  return (
    <View style={styles.root}>
      <ScrollView style={{ flex: 1 }} showsVerticalScrollIndicator={false} contentContainerStyle={styles.content}>

        {/* Progress hero card */}
        <View style={styles.heroCard}>
          {/* Circular progress (visual bar) */}
          <View style={styles.progressWrap}>
            <View style={styles.progressBg}>
              <View style={[styles.progressFill, { width: `${Math.min(pct, 100)}%` as any }]} />
            </View>
            <View style={styles.progressLabelRow}>
              <Text style={styles.progressPct}>{pct}%</Text>
              <Text style={styles.progressLabel}>complete</Text>
            </View>
          </View>

          <Text style={styles.heroSetName} numberOfLines={2}>{buildCheck.setName}</Text>

          {canBuild
            ? (
              <View style={styles.canBuildBadge}>
                <Ionicons name="checkmark-circle" size={16} color={C.green} />
                <Text style={styles.canBuildText}>You can build this set!</Text>
              </View>
            )
            : (
              <View style={styles.missingBadge}>
                <Ionicons name="construct-outline" size={14} color='#92610A' />
                <Text style={styles.missingBadgeText}>{buildCheck.missing} pieces still needed</Text>
              </View>
            )
          }
        </View>

        {/* Stats row */}
        <View style={styles.statsRow}>
          <View style={[styles.statCard, styles.statHave]}>
            <Ionicons name="checkmark-circle" size={22} color={C.green} style={{ marginBottom: 4 }} />
            <Text style={[styles.statNum, { color: C.green }]}>{buildCheck.have}</Text>
            <Text style={styles.statLabel}>You have</Text>
          </View>
          <View style={[styles.statCard, styles.statTotal]}>
            <Ionicons name="layers-outline" size={22} color={C.red} style={{ marginBottom: 4 }} />
            <Text style={[styles.statNum, { color: C.red }]}>{buildCheck.total}</Text>
            <Text style={styles.statLabel}>Total needed</Text>
          </View>
          <View style={[styles.statCard, styles.statMissing]}>
            <Ionicons name="close-circle" size={22} color='#DC2626' style={{ marginBottom: 4 }} />
            <Text style={[styles.statNum, { color: '#DC2626' }]}>{buildCheck.missing}</Text>
            <Text style={styles.statLabel}>Missing</Text>
          </View>
        </View>

        {/* BrickLink button */}
        {buildCheck.missing > 0 && (
          <TouchableOpacity style={styles.brickLinkBtn} onPress={handleGenerateBrickLink} activeOpacity={0.85}>
            <View style={styles.brickLinkIcon}>
              <Ionicons name="cart-outline" size={20} color={C.white} />
            </View>
            <View style={styles.brickLinkText}>
              <Text style={styles.brickLinkTitle}>Generate BrickLink List</Text>
              <Text style={styles.brickLinkSub}>Export missing pieces as XML</Text>
            </View>
            <Ionicons name="chevron-forward" size={18} color='rgba(255,255,255,0.6)' />
          </TouchableOpacity>
        )}

        {/* Parts you have */}
        <TouchableOpacity
          style={styles.sectionToggle}
          onPress={() => setExpanded(e => ({ ...e, have: !e.have }))}
          activeOpacity={0.8}
        >
          <View style={styles.sectionToggleLeft}>
            <View style={[styles.sectionDot, { backgroundColor: C.green }]} />
            <Text style={styles.sectionToggleTitle}>Parts You Have</Text>
            <View style={[styles.countBadge, { backgroundColor: '#DCFCE7' }]}>
              <Text style={[styles.countBadgeText, { color: C.green }]}>{buildCheck.haveParts?.length ?? 0}</Text>
            </View>
          </View>
          <Ionicons name={expanded.have ? 'chevron-up' : 'chevron-down'} size={18} color={C.textMuted} />
        </TouchableOpacity>

        {expanded.have && (
          <View style={styles.partsList}>
            {buildCheck.haveParts?.length > 0
              ? buildCheck.haveParts.map((part: any, i: number) => (
                <View
                  key={i}
                  style={[styles.partRow, i < buildCheck.haveParts.length - 1 && styles.partRowBorder]}
                >
                  <View style={styles.partInfoWrap}>
                    <Text style={styles.partName} numberOfLines={1}>{part.partName}</Text>
                    <Text style={styles.partNum}>#{part.partNum}</Text>
                  </View>
                  <View style={styles.partRight}>
                    <View style={[styles.colorSwatch, { backgroundColor: part.colorHex || '#ccc' }]} />
                    <View style={[styles.qtyBadge, { borderColor: '#BBF7D0', backgroundColor: '#F0FDF4' }]}>
                      <Text style={[styles.qtyText, { color: C.green }]}>×{part.quantity}</Text>
                    </View>
                  </View>
                </View>
              ))
              : <Text style={styles.emptyPartText}>Scan parts to track what you have</Text>
            }
          </View>
        )}

        {/* Missing parts */}
        <TouchableOpacity
          style={[styles.sectionToggle, { marginTop: 4 }]}
          onPress={() => setExpanded(e => ({ ...e, missing: !e.missing }))}
          activeOpacity={0.8}
        >
          <View style={styles.sectionToggleLeft}>
            <View style={[styles.sectionDot, { backgroundColor: '#DC2626' }]} />
            <Text style={styles.sectionToggleTitle}>Missing Parts</Text>
            <View style={[styles.countBadge, { backgroundColor: '#FEE2E2' }]}>
              <Text style={[styles.countBadgeText, { color: '#DC2626' }]}>{buildCheck.missingParts?.length ?? 0}</Text>
            </View>
          </View>
          <Ionicons name={expanded.missing ? 'chevron-up' : 'chevron-down'} size={18} color={C.textMuted} />
        </TouchableOpacity>

        {expanded.missing && (
          <View style={[styles.partsList, { marginBottom: S.xl }]}>
            {buildCheck.missingParts?.length > 0
              ? buildCheck.missingParts.map((part: any, i: number) => (
                <View
                  key={i}
                  style={[styles.partRow, i < buildCheck.missingParts.length - 1 && styles.partRowBorder]}
                >
                  <View style={styles.partInfoWrap}>
                    <Text style={styles.partName} numberOfLines={1}>{part.partName}</Text>
                    <Text style={styles.partNum}>#{part.partNum}</Text>
                  </View>
                  <View style={styles.partRight}>
                    <View style={[styles.colorSwatch, { backgroundColor: part.colorHex || '#ccc' }]} />
                    <View style={[styles.qtyBadge, { borderColor: '#FECACA', backgroundColor: '#FEF2F2' }]}>
                      <Text style={[styles.qtyText, { color: '#DC2626' }]}>need {part.quantityNeeded}</Text>
                    </View>
                  </View>
                </View>
              ))
              : (
                <View style={styles.allPartsWrap}>
                  <Ionicons name="checkmark-circle" size={32} color={C.green} />
                  <Text style={styles.allPartsText}>You have all the parts!</Text>
                </View>
              )
            }
          </View>
        )}

      </ScrollView>

      {/* BrickLink Modal */}
      <Modal visible={showBrickLinkModal} transparent animationType="slide">
        <TouchableOpacity style={styles.modalBg} activeOpacity={1} onPress={() => setShowBrickLinkModal(false)}>
          <TouchableOpacity activeOpacity={1} style={styles.brickLinkSheet} onPress={() => {}}>
            <View style={styles.sheetHandle} />

            <View style={styles.sheetHeaderRow}>
              <Text style={styles.sheetTitle}>BrickLink Wanted List</Text>
              <View style={styles.partCountBadge}>
                <Text style={styles.partCountText}>{buildCheck?.missingParts?.length ?? 0} parts</Text>
              </View>
            </View>

            <View style={styles.stepsCard}>
              <View style={styles.stepRow}>
                <View style={styles.stepNum}><Text style={styles.stepNumText}>1</Text></View>
                <Text style={styles.stepText}>Share the XML file below to your device</Text>
              </View>
              <View style={styles.stepRow}>
                <View style={styles.stepNum}><Text style={styles.stepNumText}>2</Text></View>
                <Text style={styles.stepText}>Tap "Open BrickLink" and go to My Wanted List → Upload</Text>
              </View>
              <View style={styles.stepRow}>
                <View style={styles.stepNum}><Text style={styles.stepNumText}>3</Text></View>
                <Text style={styles.stepText}>Paste the XML and click Upload to add all missing parts</Text>
              </View>
            </View>

            <View style={styles.xmlPreview}>
              <Text style={styles.xmlText} numberOfLines={6}>{brickLinkXml}</Text>
            </View>

            <TouchableOpacity style={styles.copyBtn} onPress={handleShareXml} activeOpacity={0.85}>
              <Ionicons name="share-outline" size={18} color={C.white} style={{ marginRight: 8 }} />
              <Text style={styles.copyBtnText}>Share / Copy XML</Text>
            </TouchableOpacity>

            <TouchableOpacity style={styles.brickLinkOpenBtn} onPress={handleOpenBrickLink} activeOpacity={0.85}>
              <Ionicons name="open-outline" size={18} color="#1D4ED8" style={{ marginRight: 8 }} />
              <Text style={styles.brickLinkOpenText}>Open BrickLink in Browser</Text>
            </TouchableOpacity>

            <TouchableOpacity style={styles.cancelBtn} onPress={() => setShowBrickLinkModal(false)}>
              <Text style={styles.cancelBtnText}>Close</Text>
            </TouchableOpacity>
          </TouchableOpacity>
        </TouchableOpacity>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: C.bg },
  content: { padding: S.md, gap: 12, paddingBottom: 40 },

  centered: {
    flex: 1, alignItems: 'center', justifyContent: 'center',
    backgroundColor: C.bg, gap: 12,
  },
  loadingText: { fontSize: 14, color: C.textMuted, marginTop: 8 },
  emptyTitle: { fontSize: 16, fontWeight: '600', color: C.textSub },

  // Hero
  heroCard: {
    backgroundColor: C.white, borderRadius: R.xl,
    padding: S.lg, gap: 12, alignItems: 'flex-start',
    ...shadow(2),
  },
  progressWrap: { width: '100%', gap: 6 },
  progressBg: {
    height: 12, borderRadius: 6, backgroundColor: C.bg,
    overflow: 'hidden', borderWidth: 1, borderColor: C.border,
  },
  progressFill: {
    height: '100%', borderRadius: 6,
    backgroundColor: C.red,
  },
  progressLabelRow: { flexDirection: 'row', alignItems: 'baseline', gap: 6 },
  progressPct: { fontSize: 40, fontWeight: '900', color: C.red, letterSpacing: -1 },
  progressLabel: { fontSize: 16, fontWeight: '600', color: C.textMuted },

  heroSetName: { fontSize: 18, fontWeight: '800', color: C.text, letterSpacing: -0.2 },

  canBuildBadge: {
    flexDirection: 'row', alignItems: 'center', gap: 6,
    backgroundColor: '#DCFCE7', paddingHorizontal: 12, paddingVertical: 6,
    borderRadius: R.full,
  },
  canBuildText: { fontSize: 13, fontWeight: '700', color: C.green },

  missingBadge: {
    flexDirection: 'row', alignItems: 'center', gap: 6,
    backgroundColor: '#FEF3C7', paddingHorizontal: 12, paddingVertical: 6,
    borderRadius: R.full,
  },
  missingBadgeText: { fontSize: 13, fontWeight: '700', color: '#92610A' },

  // Stats
  statsRow: { flexDirection: 'row', gap: 8 },
  statCard: {
    flex: 1, backgroundColor: C.white, borderRadius: R.lg,
    padding: 12, alignItems: 'center', ...shadow(1),
  },
  statHave: {},
  statTotal: {},
  statMissing: {},
  statNum: { fontSize: 24, fontWeight: '900', letterSpacing: -0.5 },
  statLabel: { fontSize: 11, fontWeight: '600', color: C.textMuted, marginTop: 2, textAlign: 'center' },

  // BrickLink btn
  brickLinkBtn: {
    flexDirection: 'row', alignItems: 'center', gap: S.md,
    backgroundColor: '#1D4ED8', borderRadius: R.lg,
    padding: S.md, ...shadow(2),
  },
  brickLinkIcon: {
    width: 40, height: 40, borderRadius: 10,
    backgroundColor: 'rgba(255,255,255,0.15)',
    alignItems: 'center', justifyContent: 'center',
  },
  brickLinkText: { flex: 1 },
  brickLinkTitle: { fontSize: 15, fontWeight: '700', color: C.white },
  brickLinkSub: { fontSize: 11, color: 'rgba(255,255,255,0.7)', marginTop: 1 },

  // Section toggles
  sectionToggle: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    backgroundColor: C.white, borderRadius: R.lg,
    paddingVertical: 14, paddingHorizontal: S.md,
    ...shadow(1),
  },
  sectionToggleLeft: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  sectionDot: { width: 10, height: 10, borderRadius: 5 },
  sectionToggleTitle: { fontSize: 15, fontWeight: '700', color: C.text },
  countBadge: {
    paddingHorizontal: 8, paddingVertical: 2,
    borderRadius: R.full,
  },
  countBadgeText: { fontSize: 12, fontWeight: '700' },

  // Parts list
  partsList: {
    backgroundColor: C.white, borderRadius: R.lg,
    overflow: 'hidden', ...shadow(1),
    marginTop: -4,
  },
  partRow: {
    flexDirection: 'row', alignItems: 'center',
    padding: S.md, gap: S.sm,
  },
  partRowBorder: { borderBottomWidth: 1, borderBottomColor: C.border },
  partInfoWrap: { flex: 1 },
  partName: { fontSize: 13, fontWeight: '600', color: C.text },
  partNum: { fontSize: 11, color: C.textMuted, marginTop: 2, fontFamily: Platform.OS === 'ios' ? 'Courier New' : 'monospace' },
  partRight: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  colorSwatch: {
    width: 16, height: 16, borderRadius: 4,
    borderWidth: 1, borderColor: 'rgba(0,0,0,0.12)',
  },
  qtyBadge: {
    paddingHorizontal: 8, paddingVertical: 3,
    borderRadius: R.full, borderWidth: 1,
  },
  qtyText: { fontSize: 12, fontWeight: '700' },

  emptyPartText: { fontSize: 13, color: C.textMuted, textAlign: 'center', padding: S.lg },
  allPartsWrap: { alignItems: 'center', padding: S.xl, gap: 8 },
  allPartsText: { fontSize: 15, fontWeight: '700', color: C.green },

  // BrickLink modal
  modalBg: { flex: 1, backgroundColor: C.overlay, justifyContent: 'flex-end' },
  brickLinkSheet: {
    backgroundColor: C.white,
    borderTopLeftRadius: 24, borderTopRightRadius: 24,
    padding: S.lg,
    paddingBottom: Platform.OS === 'ios' ? 40 : S.lg,
    gap: 12,
  },
  sheetHandle: {
    width: 40, height: 4, borderRadius: 2, backgroundColor: C.border,
    alignSelf: 'center', marginBottom: 4,
  },
  sheetHeaderRow: {
    flexDirection: 'row', alignItems: 'center', gap: 10,
  },
  sheetTitle: { fontSize: 20, fontWeight: '800', color: C.text, flex: 1 },
  partCountBadge: {
    backgroundColor: '#EFF6FF', paddingHorizontal: 10, paddingVertical: 4,
    borderRadius: R.full,
  },
  partCountText: { fontSize: 12, fontWeight: '700', color: '#1D4ED8' },
  sheetSub: { fontSize: 13, color: C.textMuted, lineHeight: 18 },

  stepsCard: {
    backgroundColor: C.bg, borderRadius: R.lg,
    padding: S.md, gap: 10,
  },
  stepRow: { flexDirection: 'row', alignItems: 'flex-start', gap: 10 },
  stepNum: {
    width: 22, height: 22, borderRadius: 11,
    backgroundColor: '#1D4ED8', alignItems: 'center', justifyContent: 'center',
    marginTop: 1, flexShrink: 0,
  },
  stepNumText: { fontSize: 12, fontWeight: '800', color: C.white },
  stepText: { flex: 1, fontSize: 13, color: C.textSub, lineHeight: 18 },
  xmlPreview: {
    backgroundColor: C.bg, borderRadius: R.md,
    padding: S.md, borderWidth: 1, borderColor: C.border,
    maxHeight: 180,
  },
  xmlText: {
    fontSize: 11, color: C.textSub, lineHeight: 16,
    fontFamily: Platform.OS === 'ios' ? 'Courier New' : 'monospace',
  },
  copyBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    backgroundColor: C.red, borderRadius: R.md, paddingVertical: 15,
  },
  copyBtnText: { color: C.white, fontSize: 15, fontWeight: '700' },
  brickLinkOpenBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    backgroundColor: '#EFF6FF', borderRadius: R.md, paddingVertical: 14,
    borderWidth: 1.5, borderColor: '#BFDBFE',
  },
  brickLinkOpenText: { color: '#1D4ED8', fontSize: 15, fontWeight: '700' },
  cancelBtn: {
    alignItems: 'center', paddingVertical: 14,
    borderRadius: R.md, borderWidth: 1.5, borderColor: C.border,
    backgroundColor: C.white, ...shadow(1),
  },
  cancelBtnText: { fontSize: 15, fontWeight: '600', color: C.text },
});
