/**
 * ColorSwatch — small inline component that renders a coloured circle plus
 * optional colour name. Used in scan results, the continuous-scan drawer,
 * and the brick confirmation modal.
 *
 * Notes:
 *  - Transparent / translucent LEGO colours render as a coloured ring around
 *    a translucent inner so users can see the "see-through" property at a
 *    glance. RN doesn't support CSS background-image patterns; we use a
 *    nested View instead.
 *  - When `colorHex` is missing or invalid we fall back to a neutral gray
 *    pill instead of an empty circle (looks intentional, not broken).
 *  - The optional `confidence` prop dims the swatch slightly for low-conf
 *    results — visual signal that the colour reading is uncertain.
 *  - Two layouts: inline (dot+name in one row) and stacked (dot above name).
 */
import React from 'react';
import { View, Text, StyleSheet, ViewStyle } from 'react-native';
import { C } from '@/constants/theme';

interface Props {
  /** Color name, e.g. "Bright Red". When omitted only the dot renders. */
  colorName?: string;
  /** Hex color. Accepts "#RRGGBB" or "RRGGBB"; falls back to gray when missing. */
  colorHex?: string;
  /** True for translucent LEGO colours. Renders an inner pale ring. */
  isTransparent?: boolean;
  /** 0-1 model confidence. <0.5 dims the swatch. Optional. */
  confidence?: number;
  /** Pixel size of the swatch dot. Default 14. */
  size?: number;
  /** Hide the textual name even when provided (dot-only). */
  hideName?: boolean;
  /** "inline" (default) = dot then name; "stacked" = dot above name. */
  layout?: 'inline' | 'stacked';
  style?: ViewStyle;
}

const NEUTRAL_GRAY = '#C0C0C0';

function normalizeHex(hex?: string): string {
  if (!hex) return NEUTRAL_GRAY;
  const trimmed = hex.trim();
  if (!trimmed) return NEUTRAL_GRAY;
  if (trimmed.toLowerCase() === 'transparent' || trimmed === '#00000000') {
    return NEUTRAL_GRAY;
  }
  return trimmed.startsWith('#') ? trimmed : `#${trimmed}`;
}

export const ColorSwatch: React.FC<Props> = ({
  colorName, colorHex, isTransparent, confidence,
  size = 14, hideName, layout = 'inline', style,
}) => {
  const fill = normalizeHex(colorHex);
  const opacity = confidence !== undefined && confidence < 0.5 ? 0.55 : 1;
  const showName = !hideName && !!colorName;

  const dot = (
    <View
      style={[
        styles.dot,
        {
          width: size,
          height: size,
          borderRadius: size / 2,
          backgroundColor: fill,
          opacity,
          borderWidth: isTransparent ? 1 : StyleSheet.hairlineWidth,
          borderColor: isTransparent ? 'rgba(0,0,0,0.30)' : 'rgba(0,0,0,0.15)',
        },
      ]}
    >
      {isTransparent && (
        <View
          style={{
            width: Math.max(0, size - 6),
            height: Math.max(0, size - 6),
            borderRadius: Math.max(0, (size - 6) / 2),
            backgroundColor: 'rgba(255,255,255,0.45)',
          }}
        />
      )}
    </View>
  );

  if (layout === 'stacked') {
    return (
      <View style={[styles.stackedRoot, style]}>
        {dot}
        {showName && (
          <Text
            style={[styles.stackedName, { opacity, maxWidth: size + 20 }]}
            numberOfLines={1}
          >
            {colorName}
          </Text>
        )}
      </View>
    );
  }

  return (
    <View style={[styles.row, style]}>
      {dot}
      {showName && (
        <Text style={[styles.inlineName, { opacity }]} numberOfLines={1}>
          {colorName}
        </Text>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  row: { flexDirection: 'row', alignItems: 'center' },
  dot: {
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  inlineName: {
    fontSize: 11,
    color: C.textSub,
    marginLeft: 5,
    fontWeight: '500',
  },
  stackedRoot: {
    alignItems: 'center',
  },
  stackedName: {
    fontSize: 12,
    color: C.textMuted,
    fontWeight: '500',
    textAlign: 'center',
    marginTop: 4,
  },
});
