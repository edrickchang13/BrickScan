import React, { useRef, useState } from 'react';
import {
  View, Text, TouchableOpacity, ScrollView, Animated,
  Dimensions, StyleSheet, Platform, StatusBar,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useAuthStore } from '@/store/authStore';
import { C, R, S, shadow } from '@/constants/theme';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import type { RootStackParamList } from '@/types';

const { width: SCREEN_W } = Dimensions.get('window');

type SlideFeature = { icon: string; color: string; label: string };
type Slide = {
  id: string;
  icon: string;
  iconColor: string;
  iconBg: string;
  title: string;
  subtitle: string;
  features: SlideFeature[];
};

const SLIDES: Slide[] = [
  {
    id: 'welcome',
    icon: 'scan',
    iconColor: C.red,
    iconBg: '#FFF5F5',
    title: 'Welcome to BrickScan',
    subtitle: 'AI-powered LEGO piece identification and inventory management — right from your phone.',
    features: [
      { icon: 'flash', color: C.red, label: 'Instant AI identification' },
      { icon: 'cube', color: '#6366F1', label: 'Build your inventory' },
      { icon: 'layers', color: '#059669', label: 'Track your sets' },
    ],
  },
  {
    id: 'scan-modes',
    icon: 'camera',
    iconColor: '#6366F1',
    iconBg: '#EEF2FF',
    title: 'Three Ways to Scan',
    subtitle: 'Choose the scanning mode that works best for your situation.',
    features: [
      { icon: 'camera', color: C.red, label: 'Photo — snap a single piece' },
      { icon: 'videocam', color: '#F59E0B', label: 'Video — multi-frame for accuracy' },
      { icon: 'apps', color: '#059669', label: 'Multi — scan a pile at once' },
    ],
  },
  {
    id: 'inventory',
    icon: 'checkmark-circle',
    iconColor: '#059669',
    iconBg: '#ECFDF5',
    title: 'Build & Check Sets',
    subtitle: 'Add scanned pieces to your inventory and instantly check if you have enough parts to build any LEGO set.',
    features: [
      { icon: 'add-circle', color: C.red, label: 'Add pieces with one tap' },
      { icon: 'construct', color: '#F59E0B', label: 'Check set build progress' },
      { icon: 'download-outline', color: '#059669', label: 'Export your inventory as CSV' },
    ],
  },
];

export const OnboardingScreen: React.FC = () => {
  const scrollRef = useRef<ScrollView>(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const scrollX = useRef(new Animated.Value(0)).current;
  const isLoggedIn = useAuthStore((s) => s.isLoggedIn);
  const completeOnboarding = useAuthStore((s) => s.completeOnboarding);
  const navigation = useNavigation<NativeStackNavigationProp<RootStackParamList>>();

  const handleScroll = Animated.event(
    [{ nativeEvent: { contentOffset: { x: scrollX } } }],
    {
      useNativeDriver: false,
      listener: (e: any) => {
        const x = e.nativeEvent.contentOffset.x;
        const idx = Math.round(x / SCREEN_W);
        setCurrentIndex(Math.max(0, Math.min(idx, SLIDES.length - 1)));
      },
    },
  );

  const goToSlide = (idx: number) => {
    scrollRef.current?.scrollTo({ x: idx * SCREEN_W, animated: true });
    setCurrentIndex(idx);
  };

  const handleGetStarted = async () => {
    await completeOnboarding(); // updates Zustand state → RootNavigator re-renders automatically
  };

  const isLast = currentIndex === SLIDES.length - 1;

  return (
    <View style={styles.root}>
      <StatusBar barStyle="dark-content" />

      {!isLast && (
        <TouchableOpacity style={styles.skipBtn} onPress={handleGetStarted}>
          <Text style={styles.skipText}>Skip</Text>
        </TouchableOpacity>
      )}

      <ScrollView
        ref={scrollRef}
        horizontal
        pagingEnabled
        showsHorizontalScrollIndicator={false}
        onScroll={handleScroll}
        scrollEventThrottle={16}
        style={styles.scrollView}
      >
        {SLIDES.map((slide) => (
          <View key={slide.id} style={styles.slide}>
            <View style={[styles.heroIconWrap, { backgroundColor: slide.iconBg }]}>
              <Ionicons name={slide.icon as any} size={72} color={slide.iconColor} />
            </View>

            <Text style={styles.slideTitle}>{slide.title}</Text>
            <Text style={styles.slideSubtitle}>{slide.subtitle}</Text>

            <View style={styles.featuresCard}>
              {slide.features.map((feat, i) => (
                <View
                  key={feat.label}
                  style={[
                    styles.featureRow,
                    i < slide.features.length - 1 && styles.featureRowBorder,
                  ]}
                >
                  <View style={[styles.featureIcon, { backgroundColor: feat.color + '18' }]}>
                    <Ionicons name={feat.icon as any} size={20} color={feat.color} />
                  </View>
                  <Text style={styles.featureLabel}>{feat.label}</Text>
                </View>
              ))}
            </View>
          </View>
        ))}
      </ScrollView>

      <View style={styles.bottomNav}>
        <View style={styles.dotsRow}>
          {SLIDES.map((_, i) => {
            const inputRange = [(i - 1) * SCREEN_W, i * SCREEN_W, (i + 1) * SCREEN_W];
            const dotWidth = scrollX.interpolate({
              inputRange,
              outputRange: [8, 24, 8],
              extrapolate: 'clamp',
            });
            const opacity = scrollX.interpolate({
              inputRange,
              outputRange: [0.35, 1, 0.35],
              extrapolate: 'clamp',
            });
            return (
              <Animated.View key={i} style={[styles.dot, { width: dotWidth, opacity }]} />
            );
          })}
        </View>

        {isLast ? (
          <TouchableOpacity style={styles.getStartedBtn} onPress={handleGetStarted} activeOpacity={0.85}>
            <Ionicons name="rocket-outline" size={20} color={C.white} style={{ marginRight: 8 }} />
            <Text style={styles.getStartedText}>Get Started</Text>
          </TouchableOpacity>
        ) : (
          <TouchableOpacity
            style={styles.nextBtn}
            onPress={() => goToSlide(currentIndex + 1)}
            activeOpacity={0.85}
          >
            <Text style={styles.nextBtnText}>Next</Text>
            <Ionicons name="arrow-forward" size={18} color={C.red} style={{ marginLeft: 6 }} />
          </TouchableOpacity>
        )}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: C.white },
  skipBtn: {
    position: 'absolute', top: Platform.OS === 'ios' ? 56 : 24,
    right: S.md, zIndex: 10, paddingHorizontal: 12, paddingVertical: 6,
    borderRadius: R.full, backgroundColor: C.bg,
  },
  skipText: { fontSize: 14, fontWeight: '600', color: C.textSub },

  scrollView: { flex: 1 },
  slide: {
    width: SCREEN_W, flex: 1,
    alignItems: 'center',
    paddingTop: Platform.OS === 'ios' ? 100 : 80,
    paddingHorizontal: S.lg,
  },

  heroIconWrap: {
    width: 140, height: 140, borderRadius: 40,
    alignItems: 'center', justifyContent: 'center',
    marginBottom: S.lg, ...shadow(2),
  },

  slideTitle: {
    fontSize: 28, fontWeight: '900', color: C.text,
    textAlign: 'center', marginBottom: S.sm, lineHeight: 34,
  },
  slideSubtitle: {
    fontSize: 15, color: C.textSub, textAlign: 'center',
    lineHeight: 22, marginBottom: S.lg, paddingHorizontal: S.sm,
  },

  featuresCard: {
    width: '100%', backgroundColor: C.bg, borderRadius: R.xl,
    overflow: 'hidden', borderWidth: 1, borderColor: C.border,
  },
  featureRow: {
    flexDirection: 'row', alignItems: 'center', padding: S.md, gap: S.md,
  },
  featureRowBorder: { borderBottomWidth: 1, borderBottomColor: C.border },
  featureIcon: {
    width: 40, height: 40, borderRadius: 10,
    alignItems: 'center', justifyContent: 'center',
  },
  featureLabel: { fontSize: 15, fontWeight: '600', color: C.text, flex: 1 },

  bottomNav: {
    paddingHorizontal: S.lg,
    paddingBottom: Platform.OS === 'ios' ? 48 : 32,
    paddingTop: S.md,
    alignItems: 'center', gap: S.md,
    borderTopWidth: 1, borderTopColor: C.border,
    backgroundColor: C.white,
  },

  dotsRow: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  dot: { height: 8, borderRadius: 4, backgroundColor: C.red },

  getStartedBtn: {
    width: '100%',
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    backgroundColor: C.red, borderRadius: R.md, paddingVertical: 16,
    ...shadow(2), shadowColor: C.red, shadowOpacity: 0.35,
  },
  getStartedText: { color: C.white, fontSize: 17, fontWeight: '800' },

  nextBtn: {
    width: '100%',
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    backgroundColor: '#FFF5F5', borderRadius: R.md, paddingVertical: 16,
    borderWidth: 1.5, borderColor: '#E3000B30',
  },
  nextBtnText: { color: C.red, fontSize: 16, fontWeight: '700' },
});
