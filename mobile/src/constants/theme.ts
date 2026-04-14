import { StyleSheet, Platform } from 'react-native';

// ─── Brand Colors ─────────────────────────────────────────────────────────────
export const C = {
  // LEGO red / primary
  red: '#E3000B',
  redDark: '#B5000A',
  redLight: '#FF2D1A',

  // LEGO yellow / accent
  yellow: '#FFD700',
  yellowDark: '#E6BF00',

  // Backgrounds
  bg: '#F5F5F7',
  bgDark: '#EBEBED',
  card: '#FFFFFF',
  cardAlt: '#F0F0F2',

  // Surface / overlays
  sheet: '#FFFFFF',

  // Text
  text: '#111111',
  textSub: '#555555',
  textMuted: '#999999',
  textOnRed: '#FFFFFF',

  // Borders
  border: '#E0E0E0',
  borderFocus: '#E3000B',

  // Status
  green: '#16A34A',
  greenLight: '#DCFCE7',
  orange: '#EA580C',
  orangeLight: '#FFF0E6',
  blue: '#2563EB',
  blueLight: '#EFF6FF',

  // Misc
  white: '#FFFFFF',
  black: '#000000',
  overlay: 'rgba(0,0,0,0.55)',
  transparent: 'transparent',
};

// ─── Typography ───────────────────────────────────────────────────────────────
export const T = {
  h1: { fontSize: 32, fontWeight: '800' as const, color: C.text, letterSpacing: -0.5 },
  h2: { fontSize: 24, fontWeight: '700' as const, color: C.text, letterSpacing: -0.3 },
  h3: { fontSize: 18, fontWeight: '700' as const, color: C.text },
  h4: { fontSize: 15, fontWeight: '600' as const, color: C.text },
  body: { fontSize: 15, fontWeight: '400' as const, color: C.text },
  bodySmall: { fontSize: 13, fontWeight: '400' as const, color: C.textSub },
  caption: { fontSize: 11, fontWeight: '500' as const, color: C.textMuted, letterSpacing: 0.3 },
  label: { fontSize: 13, fontWeight: '600' as const, color: C.textSub, letterSpacing: 0.2 },
  mono: { fontSize: 13, fontFamily: Platform.OS === 'ios' ? 'Courier New' : 'monospace', color: C.textSub },
};

// ─── Spacing ──────────────────────────────────────────────────────────────────
export const S = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
};

// ─── Radius ───────────────────────────────────────────────────────────────────
export const R = {
  sm: 8,
  md: 12,
  lg: 16,
  xl: 24,
  full: 999,
};

// ─── Shadows ──────────────────────────────────────────────────────────────────
export const shadow = (level: 1 | 2 | 3 = 1) => ({
  shadowColor: '#000',
  shadowOffset: { width: 0, height: level },
  shadowOpacity: level * 0.07,
  shadowRadius: level * 3,
  elevation: level * 2,
});

// ─── Common shared styles ─────────────────────────────────────────────────────
export const gs = StyleSheet.create({
  flex1: { flex: 1 },
  row: { flexDirection: 'row', alignItems: 'center' },
  center: { alignItems: 'center', justifyContent: 'center' },
  screenBg: { flex: 1, backgroundColor: C.bg },
  card: {
    backgroundColor: C.card,
    borderRadius: R.lg,
    ...shadow(1),
  },
  input: {
    backgroundColor: C.card,
    borderWidth: 1.5,
    borderColor: C.border,
    borderRadius: R.md,
    paddingHorizontal: S.md,
    paddingVertical: 13,
    fontSize: 15,
    color: C.text,
  },
  inputFocused: {
    borderColor: C.red,
  },
  btnPrimary: {
    backgroundColor: C.red,
    borderRadius: R.md,
    paddingVertical: 15,
    alignItems: 'center' as const,
    justifyContent: 'center' as const,
  },
  btnPrimaryText: {
    color: C.white,
    fontSize: 16,
    fontWeight: '700' as const,
    letterSpacing: 0.3,
  },
  btnSecondary: {
    backgroundColor: C.transparent,
    borderWidth: 1.5,
    borderColor: C.border,
    borderRadius: R.md,
    paddingVertical: 14,
    alignItems: 'center' as const,
    justifyContent: 'center' as const,
  },
  btnSecondaryText: {
    color: C.text,
    fontSize: 15,
    fontWeight: '600' as const,
  },
  pill: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: R.full,
    borderWidth: 1.5,
    borderColor: C.border,
  },
  pillActive: {
    backgroundColor: C.red,
    borderColor: C.red,
  },
  pillText: {
    fontSize: 13,
    fontWeight: '600',
    color: C.textSub,
  },
  pillTextActive: {
    color: C.white,
  },
  sectionTitle: {
    ...T.h4,
    marginBottom: S.sm,
    paddingHorizontal: S.md,
  },
  divider: {
    height: 1,
    backgroundColor: C.border,
    marginVertical: S.sm,
  },
  emptyState: {
    flex: 1,
    alignItems: 'center' as const,
    justifyContent: 'center' as const,
    paddingVertical: 64,
    paddingHorizontal: S.xl,
  },
  emptyTitle: {
    ...T.h3,
    marginTop: S.md,
    marginBottom: S.sm,
    textAlign: 'center' as const,
  },
  emptySubtitle: {
    ...T.bodySmall,
    textAlign: 'center' as const,
    lineHeight: 20,
  },
});
