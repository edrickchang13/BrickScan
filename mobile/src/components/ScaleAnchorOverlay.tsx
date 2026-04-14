/**
 * ScaleAnchorOverlay
 *
 * Displays a semi-transparent overlay to help users establish physical scale
 * for LEGO brick scanning. Shows an animated finger icon with instructions to
 * place a reference object (finger or credit card) in the frame.
 *
 * Usage:
 *   <ScaleAnchorOverlay
 *     visible={true}
 *     onScaleConfirmed={handleScaleConfirmed}
 *     onSkip={handleSkip}
 *   />
 */

import React, { useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Modal,
  TouchableOpacity,
  Animated,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { C, R, S, shadow } from '@/constants/theme';

// ─── Types ───────────────────────────────────────────────────────────────────

export interface ScaleAnchorOverlayProps {
  visible: boolean;
  onScaleConfirmed: (pixelsPerMm: number) => void;
  onSkip: () => void;
}

// ─── Animated finger icon component ──────────────────────────────────────────

const AnimatedFingerIcon: React.FC = () => {
  const scale = useRef(new Animated.Value(1.0)).current;

  useEffect(() => {
    const anim = Animated.loop(
      Animated.sequence([
        Animated.timing(scale, {
          toValue: 1.2,
          duration: 600,
          useNativeDriver: true,
        }),
        Animated.timing(scale, {
          toValue: 1.0,
          duration: 600,
          useNativeDriver: true,
        }),
      ])
    );
    anim.start();
    return () => anim.stop();
  }, []);

  return (
    <Animated.View style={[{ transform: [{ scale }] }]}>
      <Text style={styles.fingerEmoji}>👆</Text>
    </Animated.View>
  );
};

// ─── Main component ──────────────────────────────────────────────────────────

export const ScaleAnchorOverlay: React.FC<ScaleAnchorOverlayProps> = ({
  visible,
  onScaleConfirmed,
  onSkip,
}) => {
  const [useCrediCard, setUseCrediCard] = React.useState(false);

  const handleDone = () => {
    // Default value: 0 signals server-side detection (actual detection happens server-side)
    onScaleConfirmed(0);
  };

  return (
    <Modal
      visible={visible}
      transparent
      animationType="fade"
    >
      {/* Semi-transparent overlay */}
      <View style={styles.overlay}>
        {/* Centered instruction card */}
        <View style={styles.card}>
          {/* Animated finger icon */}
          <View style={styles.iconWrapper}>
            <AnimatedFingerIcon />
          </View>

          {/* Instructions text */}
          <Text style={styles.instructionTitle}>
            {useCrediCard ? '💳 Credit Card' : '👆 Finger'}
          </Text>
          <Text style={styles.instructionText}>
            {useCrediCard
              ? 'Place a credit card beside the brick for scale reference.'
              : 'Place your index finger beside the brick for scale reference.'}
          </Text>

          {/* Divider */}
          <View style={styles.divider} />

          {/* Action buttons */}
          <View style={styles.buttonGroup}>
            <TouchableOpacity
              style={styles.primaryButton}
              onPress={handleDone}
              activeOpacity={0.8}
            >
              <Ionicons name="checkmark" size={16} color={C.white} />
              <Text style={styles.buttonText}>
                Done — {useCrediCard ? 'card' : 'finger'} in frame
              </Text>
            </TouchableOpacity>

            {/* Alternative option toggle */}
            <TouchableOpacity
              style={styles.secondaryButton}
              onPress={() => setUseCrediCard(!useCrediCard)}
              activeOpacity={0.7}
            >
              <Text style={styles.secondaryButtonText}>
                {useCrediCard ? 'Use finger instead' : 'Use credit card instead'}
              </Text>
            </TouchableOpacity>
          </View>

          {/* Skip button */}
          <TouchableOpacity
            style={styles.skipButton}
            onPress={onSkip}
            activeOpacity={0.7}
          >
            <Text style={styles.skipButtonText}>Skip scale detection</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );
};

// ─── Styles ───────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.65)',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: S.md,
  },
  card: {
    backgroundColor: C.card,
    borderRadius: R.lg,
    paddingHorizontal: S.lg,
    paddingVertical: S.xl,
    alignItems: 'center',
    maxWidth: 400,
    width: '100%',
    ...shadow(2),
  },
  iconWrapper: {
    marginBottom: S.lg,
    alignItems: 'center',
    justifyContent: 'center',
  },
  fingerEmoji: {
    fontSize: 48,
  },
  instructionTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: C.text,
    marginBottom: S.xs,
    textAlign: 'center',
  },
  instructionText: {
    fontSize: 14,
    color: C.textSub,
    textAlign: 'center',
    lineHeight: 20,
    marginBottom: S.lg,
  },
  divider: {
    height: 1,
    backgroundColor: C.border,
    width: '100%',
    marginBottom: S.lg,
  },
  buttonGroup: {
    width: '100%',
    gap: S.sm,
    marginBottom: S.lg,
  },
  primaryButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: C.red,
    paddingVertical: 12,
    paddingHorizontal: S.md,
    borderRadius: R.md,
    gap: S.sm,
  },
  buttonText: {
    color: C.white,
    fontSize: 14,
    fontWeight: '600',
  },
  secondaryButton: {
    paddingVertical: 10,
    paddingHorizontal: S.md,
    borderRadius: R.md,
    backgroundColor: C.cardAlt,
    alignItems: 'center',
  },
  secondaryButtonText: {
    color: C.textSub,
    fontSize: 13,
    fontWeight: '500',
  },
  skipButton: {
    paddingVertical: 8,
  },
  skipButtonText: {
    color: C.textMuted,
    fontSize: 13,
    fontWeight: '500',
  },
});
