import React from 'react';
import { View, Text, Image, TouchableOpacity, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { C, R, S, shadow } from '@/constants/theme';

interface SetCardProps {
  setNum: string;
  name: string;
  year: number;
  numParts: number;
  imgUrl?: string;
  theme: string;
  onPress?: () => void;
}

export const SetCard: React.FC<SetCardProps> = ({
  setNum, name, year, numParts, imgUrl, theme, onPress,
}) => (
  <TouchableOpacity style={styles.card} onPress={onPress} activeOpacity={0.8}>
    {/* Image */}
    <View style={styles.imageWrap}>
      {imgUrl
        ? <Image source={{ uri: imgUrl }} style={styles.image} resizeMode="contain" />
        : (
          <View style={styles.imagePlaceholder}>
            <Ionicons name="layers-outline" size={36} color={C.textMuted} />
          </View>
        )
      }
      {/* Year badge */}
      <View style={styles.yearBadge}>
        <Text style={styles.yearText}>{year}</Text>
      </View>
    </View>

    {/* Info */}
    <View style={styles.info}>
      <Text style={styles.setName} numberOfLines={2}>{name}</Text>

      <View style={styles.metaRow}>
        {theme ? (
          <View style={styles.themePill}>
            <Text style={styles.themeText} numberOfLines={1}>{theme}</Text>
          </View>
        ) : null}
      </View>

      <View style={styles.footer}>
        <Text style={styles.setNum}>#{setNum}</Text>
        <View style={styles.partsBadge}>
          <Ionicons name="cube-outline" size={10} color={C.red} />
          <Text style={styles.partsText}>{numParts.toLocaleString()}</Text>
        </View>
      </View>
    </View>
  </TouchableOpacity>
);

const styles = StyleSheet.create({
  card: {
    flex: 1, backgroundColor: C.white, borderRadius: R.lg,
    overflow: 'hidden', borderWidth: 1, borderColor: C.border,
    ...shadow(1),
  },
  imageWrap: {
    height: 140, backgroundColor: C.bg,
    alignItems: 'center', justifyContent: 'center',
  },
  image: { width: '90%', height: '90%' },
  imagePlaceholder: { flex: 1, alignItems: 'center', justifyContent: 'center' },

  yearBadge: {
    position: 'absolute', bottom: 6, right: 6,
    backgroundColor: 'rgba(0,0,0,0.45)',
    paddingHorizontal: 7, paddingVertical: 2,
    borderRadius: R.full,
  },
  yearText: { fontSize: 10, color: C.white, fontWeight: '700' },

  info: { padding: 10, gap: 4 },
  setName: {
    fontSize: 12, fontWeight: '700', color: C.text,
    lineHeight: 16, minHeight: 32,
  },

  metaRow: { flexDirection: 'row' },
  themePill: {
    backgroundColor: '#FFF5E6', paddingHorizontal: 7, paddingVertical: 2,
    borderRadius: R.full, borderWidth: 1, borderColor: '#FFD700',
    maxWidth: '100%',
  },
  themeText: { fontSize: 10, color: '#92610A', fontWeight: '600' },

  footer: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    paddingTop: 6, borderTopWidth: 1, borderTopColor: C.border, marginTop: 2,
  },
  setNum: { fontSize: 10, color: C.textMuted, fontFamily: 'monospace' },
  partsBadge: { flexDirection: 'row', alignItems: 'center', gap: 3 },
  partsText: { fontSize: 11, fontWeight: '700', color: C.red },
});
