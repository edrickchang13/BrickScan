import React from 'react';
import { View, ActivityIndicator, Text, StyleSheet } from 'react-native';
import { C, R, S, shadow } from '@/constants/theme';

interface LoadingOverlayProps {
  visible: boolean;
  message?: string;
}

export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({ visible, message }) => {
  if (!visible) return null;

  return (
    <View style={styles.overlay}>
      <View style={styles.card}>
        <ActivityIndicator size="large" color={C.red} />
        {message && <Text style={styles.message}>{message}</Text>}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  overlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: C.overlay,
    alignItems: 'center', justifyContent: 'center',
    zIndex: 50,
  },
  card: {
    backgroundColor: C.white, borderRadius: R.xl,
    paddingVertical: 28, paddingHorizontal: 36,
    alignItems: 'center', gap: 14,
    ...shadow(3),
  },
  message: {
    fontSize: 14, fontWeight: '600', color: C.textSub,
    textAlign: 'center', maxWidth: 200,
  },
});
