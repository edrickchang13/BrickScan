/**
 * HeatmapExplainer — "Why did the model pick this part?" overlay.
 *
 * Taps into the backend /scan/{scan_id}/heatmap endpoint which returns a
 * Grad-CAM (or occlusion-sensitivity) PNG showing which regions of the
 * scan drove the top prediction.
 *
 * Visible when:
 *   - scanId is non-null (server has persisted the scan thumbnail).
 *   - source === our local model ('distilled_model' or 'contrastive_knn').
 *     Brickognize / Gemini predictions don't have introspectable weights
 *     on our side, so the explainer stays hidden for them.
 *
 * UX:
 *   - Collapsed: a small outlined "Why this?" button.
 *   - Expanded:  Grad-CAM overlay over the original thumbnail + 1-sentence
 *                plain-language explanation.
 */
import React, { useCallback, useState } from 'react';
import { View, Text, Image, TouchableOpacity, ActivityIndicator, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as SecureStore from 'expo-secure-store';
import { C, R, S } from '@/constants/theme';
import { Config } from '@/constants/config';

const TOKEN_KEY = 'brickscan_token';

interface Props {
  scanId: string;
  source: string;
  partName: string;
}

const EXPLAINABLE_SOURCES = new Set([
  'distilled_model',
  'contrastive_knn',
  'onnx_local',
  'efficientnet_local',
]);

export const HeatmapExplainer: React.FC<Props> = ({ scanId, source, partName }) => {
  const [expanded, setExpanded] = useState(false);
  const [heatmapUri, setHeatmapUri] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canExplain = EXPLAINABLE_SOURCES.has(source);

  const fetchHeatmap = useCallback(async () => {
    if (heatmapUri || loading) return;
    setLoading(true);
    setError(null);
    try {
      const token = await SecureStore.getItemAsync(TOKEN_KEY);
      const url = `${Config.API_BASE_URL}/scan/${scanId}/heatmap`;
      const res = await fetch(url, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      if (res.status === 204) {
        setError('No heatmap available for this prediction.');
      } else if (res.ok) {
        // React Native fetch returns a blob we can data-URI-ify
        const blob = await res.blob();
        const reader = new FileReader();
        reader.onloadend = () => setHeatmapUri(reader.result as string);
        reader.readAsDataURL(blob);
      } else {
        setError(`Heatmap request failed (${res.status})`);
      }
    } catch (e: any) {
      setError(e?.message ?? 'Heatmap request failed');
    } finally {
      setLoading(false);
    }
  }, [heatmapUri, loading, scanId]);

  const toggle = useCallback(() => {
    const next = !expanded;
    setExpanded(next);
    if (next) void fetchHeatmap();
  }, [expanded, fetchHeatmap]);

  if (!canExplain) return null;

  return (
    <View style={styles.container}>
      <TouchableOpacity style={styles.toggleBtn} onPress={toggle} activeOpacity={0.8}>
        <Ionicons
          name={expanded ? 'chevron-up' : 'help-circle-outline'}
          size={16}
          color={C.red}
          style={{ marginRight: 6 }}
        />
        <Text style={styles.toggleText}>
          {expanded ? 'Hide explanation' : 'Why this part?'}
        </Text>
      </TouchableOpacity>

      {expanded && (
        <View style={styles.panel}>
          {loading && (
            <View style={styles.loading}>
              <ActivityIndicator color={C.red} />
              <Text style={styles.loadingText}>Analyzing the scan…</Text>
            </View>
          )}

          {!loading && error && (
            <Text style={styles.errorText}>{error}</Text>
          )}

          {!loading && heatmapUri && (
            <>
              <Image source={{ uri: heatmapUri }} style={styles.heatImage} resizeMode="contain" />
              <Text style={styles.caption}>
                The coloured regions show where the model focused when deciding
                this was a {partName}. Warmer (red / yellow) regions had the
                biggest impact; cooler regions were mostly ignored.
              </Text>
            </>
          )}
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginHorizontal: S.md,
    marginTop: S.sm,
  },
  toggleBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
    paddingVertical: 6,
    paddingHorizontal: 10,
    borderWidth: 1,
    borderColor: C.red,
    borderRadius: R.sm,
    backgroundColor: C.white,
  },
  toggleText: {
    fontSize: 13,
    fontWeight: '600',
    color: C.red,
  },
  panel: {
    marginTop: S.sm,
    padding: S.md,
    borderRadius: R.md,
    backgroundColor: C.cardAlt,
  },
  loading: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  loadingText: {
    marginLeft: 10,
    color: C.textMuted,
  },
  errorText: {
    color: C.textMuted,
    fontStyle: 'italic',
  },
  heatImage: {
    width: '100%',
    aspectRatio: 1,
    backgroundColor: C.white,
    borderRadius: R.sm,
  },
  caption: {
    marginTop: S.sm,
    fontSize: 12,
    color: C.textMuted,
    lineHeight: 16,
  },
});
