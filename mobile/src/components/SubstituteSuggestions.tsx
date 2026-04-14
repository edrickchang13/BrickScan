import React, { useState, useEffect } from 'react';
import {
  View, Text, Image, TouchableOpacity, ScrollView, StyleSheet, ActivityIndicator,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { apiClient } from '@/services/api';
import type { PartSubstitute } from '@/types';
import { C, R, S, shadow } from '@/constants/theme';

interface Props {
  partNum: string;
  onSelectSubstitute?: (substitute: PartSubstitute) => void;
}

const SubstituteCard: React.FC<{
  substitute: PartSubstitute;
  onPress?: () => void;
}> = ({ substitute, onPress }) => {
  const [imageError, setImageError] = useState(false);
  const similarityPct = Math.round(substitute.similarity * 100);

  return (
    <TouchableOpacity
      style={styles.card}
      onPress={onPress}
      activeOpacity={0.75}
    >
      {/* Image */}
      <View style={styles.imageContainer}>
        {substitute.imageUrl && !imageError ? (
          <Image
            source={{ uri: substitute.imageUrl }}
            style={styles.image}
            resizeMode="contain"
            onError={() => setImageError(true)}
          />
        ) : (
          <View style={styles.imagePlaceholder}>
            <Ionicons name="cube-outline" size={28} color={C.textMuted} />
          </View>
        )}
      </View>

      {/* Similarity Badge */}
      <View style={styles.similarityBadge}>
        <Text style={styles.similarityText}>{similarityPct}%</Text>
      </View>

      {/* Info */}
      <Text style={styles.partName} numberOfLines={2}>{substitute.name}</Text>
      <Text style={styles.partNum} numberOfLines={1}>#{substitute.partNum}</Text>
      <Text style={styles.reason} numberOfLines={2}>{substitute.reason}</Text>

      {/* View Button */}
      <TouchableOpacity style={styles.viewButton} activeOpacity={0.8}>
        <Text style={styles.viewButtonText}>View Part</Text>
        <Ionicons name="chevron-forward" size={13} color={C.red} style={{ marginLeft: 4 }} />
      </TouchableOpacity>
    </TouchableOpacity>
  );
};

export const SubstituteSuggestions: React.FC<Props> = ({
  partNum,
  onSelectSubstitute,
}) => {
  const [substitutes, setSubstitutes] = useState<PartSubstitute[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadSubstitutes();
  }, [partNum]);

  const loadSubstitutes = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const data = await apiClient.getPartSubstitutes(partNum);
      setSubstitutes(data || []);
    } catch (err) {
      console.error('Failed to load substitutes:', err);
      setError('Could not load suggestions');
    } finally {
      setIsLoading(false);
    }
  };

  // If loading or no data, don't show section
  if (isLoading) {
    return (
      <View style={styles.container}>
        <Text style={styles.header}>Finding similar parts…</Text>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="small" color={C.red} />
        </View>
      </View>
    );
  }

  if (!substitutes || substitutes.length === 0) {
    return null; // Hide section if no substitutes
  }

  return (
    <View style={styles.container}>
      <Text style={styles.header}>Similar parts you could use</Text>
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.scrollContent}
      >
        {substitutes.map((substitute) => (
          <SubstituteCard
            key={substitute.partNum}
            substitute={substitute}
            onPress={() => onSelectSubstitute?.(substitute)}
          />
        ))}
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    gap: 12,
  },
  header: {
    fontSize: 14,
    fontWeight: '700',
    color: C.text,
    paddingHorizontal: S.md,
  },
  scrollContent: {
    paddingHorizontal: S.md,
    gap: 10,
  },
  card: {
    width: 140,
    backgroundColor: C.white,
    borderRadius: R.lg,
    padding: S.sm,
    alignItems: 'center',
    ...shadow(1),
  },
  imageContainer: {
    width: 100,
    height: 100,
    borderRadius: R.md,
    backgroundColor: C.bg,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: S.sm,
  },
  image: {
    width: 90,
    height: 90,
  },
  imagePlaceholder: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  similarityBadge: {
    position: 'absolute',
    top: 8,
    right: 8,
    backgroundColor: C.green,
    borderRadius: R.full,
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  similarityText: {
    fontSize: 11,
    fontWeight: '700',
    color: C.white,
  },
  partName: {
    fontSize: 12,
    fontWeight: '600',
    color: C.text,
    textAlign: 'center',
    marginBottom: 2,
  },
  partNum: {
    fontSize: 10,
    color: C.textMuted,
    fontFamily: 'monospace',
    marginBottom: S.xs,
  },
  reason: {
    fontSize: 9,
    color: C.textMuted,
    textAlign: 'center',
    marginBottom: S.sm,
    lineHeight: 12,
  },
  viewButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#FFF5F5',
    borderRadius: R.sm,
    paddingHorizontal: 8,
    paddingVertical: 6,
    borderWidth: 1,
    borderColor: C.red + '30',
  },
  viewButtonText: {
    fontSize: 10,
    fontWeight: '600',
    color: C.red,
  },
  loadingContainer: {
    paddingHorizontal: S.md,
    paddingVertical: S.md,
    alignItems: 'center',
  },
});
