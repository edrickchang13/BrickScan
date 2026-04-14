import React, { useState } from 'react';
import { View, Text, Image, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { C, R, S, shadow } from '@/constants/theme';

interface PartCardProps {
  partNum: string;
  name: string;
  colorName: string;
  colorHex: string;
  quantity?: number;
  imageUrl?: string;
  size?: 'small' | 'medium' | 'large';
}

export const PartCard: React.FC<PartCardProps> = ({
  partNum,
  name,
  colorName,
  colorHex,
  quantity,
  imageUrl,
  size = 'medium',
}) => {
  const imageHeight = size === 'small' ? 64 : size === 'large' ? 160 : 96;
  const [imgError, setImgError] = useState(false);

  return (
    <View style={styles.card}>
      {/* Image */}
      <View style={[styles.imageWrap, { height: imageHeight }]}>
        {imageUrl && !imgError
          ? (
            <Image
              source={{ uri: imageUrl }}
              style={styles.image}
              resizeMode="contain"
              onError={() => setImgError(true)}
            />
          )
          : (
            <View style={styles.imagePlaceholder}>
              <Ionicons name="cube-outline" size={size === 'small' ? 20 : 28} color={C.textMuted} />
            </View>
          )
        }
        {quantity !== undefined && (
          <View style={styles.qtyBadge}>
            <Text style={styles.qtyText}>{quantity}</Text>
          </View>
        )}
      </View>

      {/* Info */}
      <View style={styles.info}>
        <Text style={styles.partName} numberOfLines={2}>{name}</Text>
        <Text style={styles.partNum}>#{partNum}</Text>
        <View style={styles.colorRow}>
          <View style={[styles.colorDot, { backgroundColor: colorHex || '#ccc' }]} />
          <Text style={styles.colorName} numberOfLines={1}>{colorName}</Text>
        </View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  card: {
    flex: 1, backgroundColor: C.white, borderRadius: R.lg,
    overflow: 'hidden', borderWidth: 1, borderColor: C.border,
    ...shadow(1),
  },
  imageWrap: {
    width: '100%', backgroundColor: C.bg,
    alignItems: 'center', justifyContent: 'center',
  },
  image: { width: '90%', height: '90%' },
  imagePlaceholder: { alignItems: 'center', justifyContent: 'center', flex: 1 },

  qtyBadge: {
    position: 'absolute', top: 6, right: 6,
    backgroundColor: C.red, borderRadius: R.full,
    minWidth: 22, height: 22,
    alignItems: 'center', justifyContent: 'center',
    paddingHorizontal: 5,
  },
  qtyText: { color: C.white, fontSize: 11, fontWeight: '800' },

  info: { padding: S.sm, gap: 3 },
  partName: { fontSize: 12, fontWeight: '700', color: C.text, lineHeight: 16 },
  partNum: { fontSize: 10, color: C.textMuted, fontFamily: 'monospace' },
  colorRow: { flexDirection: 'row', alignItems: 'center', gap: 5, marginTop: 2 },
  colorDot: { width: 10, height: 10, borderRadius: 5, borderWidth: 1, borderColor: 'rgba(0,0,0,0.1)' },
  colorName: { fontSize: 10, color: C.textSub, flex: 1, fontWeight: '500' },
});
