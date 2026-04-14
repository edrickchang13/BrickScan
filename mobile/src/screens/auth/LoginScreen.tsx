import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
  StatusBar,
} from 'react-native';
import { useAuthStore } from '@/store/authStore';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { AuthStackParamList } from '@/types';
import { C, R, S, shadow } from '@/constants/theme';

type Props = NativeStackScreenProps<AuthStackParamList, 'Login'>;

// Stud pattern decoration
const StudRow = ({ count = 7 }: { count?: number }) => (
  <View style={styles.studRow}>
    {Array.from({ length: count }).map((_, i) => (
      <View key={i} style={styles.stud} />
    ))}
  </View>
);

export const LoginScreen: React.FC<Props> = ({ navigation }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [emailFocused, setEmailFocused] = useState(false);
  const [passwordFocused, setPasswordFocused] = useState(false);
  const passwordRef = useRef<TextInput>(null);

  const login = useAuthStore((state) => state.login);
  const continueAsGuest = useAuthStore((state) => state.continueAsGuest);

  const handleLogin = async () => {
    setError(null);
    if (!email.trim()) { setError('Please enter your email'); return; }
    if (!email.includes('@')) { setError('Please enter a valid email'); return; }
    if (!password.trim()) { setError('Please enter your password'); return; }

    setIsLoading(true);
    try {
      await login(email, password);
    } catch (err: any) {
      setError(err?.response?.data?.detail || err?.message || 'Login failed. Check your credentials.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView
      style={styles.root}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <StatusBar barStyle="light-content" />

      {/* Hero header */}
      <View style={styles.hero}>
        <StudRow count={8} />
        <View style={styles.heroContent}>
          <View style={styles.logoBlock}>
            <Text style={styles.logoText}>BS</Text>
          </View>
          <Text style={styles.appName}>BrickScan</Text>
          <Text style={styles.tagline}>Your LEGO collection, catalogued</Text>
        </View>
        <StudRow count={8} />
      </View>

      {/* Form card */}
      <ScrollView
        style={styles.formScroll}
        contentContainerStyle={styles.formContent}
        keyboardShouldPersistTaps="handled"
        showsVerticalScrollIndicator={false}
      >
        <Text style={styles.formTitle}>Sign In</Text>

        {error ? (
          <View style={styles.errorBox}>
            <Text style={styles.errorText}>⚠ {error}</Text>
          </View>
        ) : null}

        {/* Email */}
        <View style={styles.fieldGroup}>
          <Text style={styles.label}>Email</Text>
          <TextInput
            style={[styles.input, emailFocused && styles.inputFocused]}
            placeholder="you@example.com"
            placeholderTextColor={C.textMuted}
            keyboardType="email-address"
            autoCapitalize="none"
            autoCorrect={false}
            returnKeyType="next"
            editable={!isLoading}
            value={email}
            onChangeText={setEmail}
            onFocus={() => setEmailFocused(true)}
            onBlur={() => setEmailFocused(false)}
            onSubmitEditing={() => passwordRef.current?.focus()}
          />
        </View>

        {/* Password */}
        <View style={styles.fieldGroup}>
          <Text style={styles.label}>Password</Text>
          <TextInput
            ref={passwordRef}
            style={[styles.input, passwordFocused && styles.inputFocused]}
            placeholder="Enter your password"
            placeholderTextColor={C.textMuted}
            secureTextEntry
            returnKeyType="done"
            editable={!isLoading}
            value={password}
            onChangeText={setPassword}
            onFocus={() => setPasswordFocused(true)}
            onBlur={() => setPasswordFocused(false)}
            onSubmitEditing={handleLogin}
          />
        </View>

        {/* Sign In button */}
        <TouchableOpacity
          style={[styles.primaryBtn, isLoading && styles.btnDisabled]}
          onPress={handleLogin}
          disabled={isLoading}
          activeOpacity={0.85}
        >
          {isLoading
            ? <ActivityIndicator color={C.white} />
            : <Text style={styles.primaryBtnText}>Sign In</Text>
          }
        </TouchableOpacity>

        {/* Register link */}
        <View style={styles.switchRow}>
          <Text style={styles.switchText}>Don't have an account? </Text>
          <TouchableOpacity onPress={() => navigation.navigate('Register')} disabled={isLoading}>
            <Text style={styles.switchLink}>Create one</Text>
          </TouchableOpacity>
        </View>

        {/* Divider */}
        <View style={styles.dividerRow}>
          <View style={styles.divider} />
          <Text style={styles.dividerText}>or</Text>
          <View style={styles.divider} />
        </View>

        {/* Guest mode button */}
        <TouchableOpacity
          style={styles.guestBtn}
          onPress={continueAsGuest}
          disabled={isLoading}
          activeOpacity={0.85}
        >
          <Text style={styles.guestBtnText}>Use without account</Text>
          <Text style={styles.guestBtnSub}>Local inventory only</Text>
        </TouchableOpacity>
      </ScrollView>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: C.bg },

  // Hero
  hero: { backgroundColor: C.red, paddingTop: 56, paddingBottom: 0 },
  studRow: {
    flexDirection: 'row',
    paddingHorizontal: 8,
    paddingVertical: 2,
    gap: 2,
    backgroundColor: C.redDark,
  },
  stud: {
    flex: 1,
    height: 10,
    borderRadius: 5,
    backgroundColor: 'rgba(255,255,255,0.15)',
  },
  heroContent: { alignItems: 'center', paddingVertical: 32, paddingHorizontal: S.lg },
  logoBlock: {
    width: 72,
    height: 72,
    borderRadius: 18,
    backgroundColor: C.yellow,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 14,
    ...shadow(2),
  },
  logoText: { fontSize: 28, fontWeight: '900', color: C.red, letterSpacing: -1 },
  appName: { fontSize: 34, fontWeight: '900', color: C.white, letterSpacing: -1, marginBottom: 4 },
  tagline: { fontSize: 14, color: 'rgba(255,255,255,0.75)', fontWeight: '500' },

  // Form
  formScroll: { flex: 1 },
  formContent: { padding: S.lg, paddingTop: S.xl },
  formTitle: { fontSize: 22, fontWeight: '800', color: C.text, marginBottom: S.lg, letterSpacing: -0.5 },

  // Error
  errorBox: {
    backgroundColor: '#FFF0F0',
    borderWidth: 1,
    borderColor: '#FFCCCC',
    borderRadius: R.md,
    padding: 12,
    marginBottom: S.md,
  },
  errorText: { color: '#CC0000', fontSize: 13, fontWeight: '500' },

  // Fields
  fieldGroup: { marginBottom: S.md },
  label: { fontSize: 13, fontWeight: '600', color: C.textSub, marginBottom: 6, letterSpacing: 0.2 },
  input: {
    backgroundColor: C.white,
    borderWidth: 1.5,
    borderColor: C.border,
    borderRadius: R.md,
    paddingHorizontal: S.md,
    paddingVertical: 13,
    fontSize: 15,
    color: C.text,
    ...shadow(1),
  },
  inputFocused: { borderColor: C.red, backgroundColor: C.white },

  // Buttons
  primaryBtn: {
    backgroundColor: C.red,
    borderRadius: R.md,
    paddingVertical: 15,
    alignItems: 'center',
    marginTop: S.sm,
    marginBottom: S.lg,
    ...shadow(2),
  },
  btnDisabled: { opacity: 0.6 },
  primaryBtnText: { color: C.white, fontSize: 16, fontWeight: '700', letterSpacing: 0.3 },

  // Footer
  switchRow: { flexDirection: 'row', justifyContent: 'center', alignItems: 'center' },
  switchText: { fontSize: 14, color: C.textSub },
  switchLink: { fontSize: 14, fontWeight: '700', color: C.red },

  // Divider
  dividerRow: { flexDirection: 'row', alignItems: 'center', marginVertical: S.md },
  divider: { flex: 1, height: 1, backgroundColor: C.border },
  dividerText: { marginHorizontal: S.sm, color: C.textMuted, fontSize: 13 },

  // Guest mode button
  guestBtn: {
    backgroundColor: C.bg,
    borderWidth: 1.5,
    borderColor: C.border,
    borderRadius: R.md,
    paddingVertical: 14,
    alignItems: 'center',
    ...shadow(1),
  },
  guestBtnText: { color: C.text, fontSize: 15, fontWeight: '700' },
  guestBtnSub: { color: C.textMuted, fontSize: 12, marginTop: 2 },
});
