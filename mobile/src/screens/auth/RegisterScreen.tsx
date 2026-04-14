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

type Props = NativeStackScreenProps<AuthStackParamList, 'Register'>;

const StudRow = ({ count = 7 }: { count?: number }) => (
  <View style={styles.studRow}>
    {Array.from({ length: count }).map((_, i) => (
      <View key={i} style={styles.stud} />
    ))}
  </View>
);

export const RegisterScreen: React.FC<Props> = ({ navigation }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [focused, setFocused] = useState<string | null>(null);

  const passwordRef = useRef<TextInput>(null);
  const confirmRef = useRef<TextInput>(null);

  const register = useAuthStore((state) => state.register);

  const validate = () => {
    if (!email.trim()) { setError('Please enter your email'); return false; }
    if (!email.includes('@')) { setError('Please enter a valid email'); return false; }
    if (!password.trim()) { setError('Please enter a password'); return false; }
    if (password.length < 6) { setError('Password must be at least 6 characters'); return false; }
    if (password !== confirmPassword) { setError('Passwords do not match'); return false; }
    return true;
  };

  const handleRegister = async () => {
    setError(null);
    if (!validate()) return;
    setIsLoading(true);
    try {
      await register(email, password);
    } catch (err: any) {
      setError(err?.response?.data?.detail || err?.message || 'Registration failed.');
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

      {/* Hero */}
      <View style={styles.hero}>
        <StudRow count={8} />
        <View style={styles.heroContent}>
          <View style={styles.logoBlock}>
            <Text style={styles.logoText}>BS</Text>
          </View>
          <Text style={styles.appName}>BrickScan</Text>
          <Text style={styles.tagline}>Start building your collection</Text>
        </View>
        <StudRow count={8} />
      </View>

      <ScrollView
        style={styles.formScroll}
        contentContainerStyle={styles.formContent}
        keyboardShouldPersistTaps="handled"
        showsVerticalScrollIndicator={false}
      >
        <Text style={styles.formTitle}>Create Account</Text>

        {error ? (
          <View style={styles.errorBox}>
            <Text style={styles.errorText}>⚠ {error}</Text>
          </View>
        ) : null}

        <View style={styles.fieldGroup}>
          <Text style={styles.label}>Email</Text>
          <TextInput
            style={[styles.input, focused === 'email' && styles.inputFocused]}
            placeholder="you@example.com"
            placeholderTextColor={C.textMuted}
            keyboardType="email-address"
            autoCapitalize="none"
            autoCorrect={false}
            returnKeyType="next"
            editable={!isLoading}
            value={email}
            onChangeText={setEmail}
            onFocus={() => setFocused('email')}
            onBlur={() => setFocused(null)}
            onSubmitEditing={() => passwordRef.current?.focus()}
          />
        </View>

        <View style={styles.fieldGroup}>
          <Text style={styles.label}>Password</Text>
          <TextInput
            ref={passwordRef}
            style={[styles.input, focused === 'password' && styles.inputFocused]}
            placeholder="At least 6 characters"
            placeholderTextColor={C.textMuted}
            secureTextEntry
            returnKeyType="next"
            editable={!isLoading}
            value={password}
            onChangeText={setPassword}
            onFocus={() => setFocused('password')}
            onBlur={() => setFocused(null)}
            onSubmitEditing={() => confirmRef.current?.focus()}
          />
        </View>

        <View style={styles.fieldGroup}>
          <Text style={styles.label}>Confirm Password</Text>
          <TextInput
            ref={confirmRef}
            style={[styles.input, focused === 'confirm' && styles.inputFocused]}
            placeholder="Re-enter your password"
            placeholderTextColor={C.textMuted}
            secureTextEntry
            returnKeyType="done"
            editable={!isLoading}
            value={confirmPassword}
            onChangeText={setConfirmPassword}
            onFocus={() => setFocused('confirm')}
            onBlur={() => setFocused(null)}
            onSubmitEditing={handleRegister}
          />
        </View>

        <TouchableOpacity
          style={[styles.primaryBtn, isLoading && styles.btnDisabled]}
          onPress={handleRegister}
          disabled={isLoading}
          activeOpacity={0.85}
        >
          {isLoading
            ? <ActivityIndicator color={C.white} />
            : <Text style={styles.primaryBtnText}>Create Account</Text>
          }
        </TouchableOpacity>

        <View style={styles.switchRow}>
          <Text style={styles.switchText}>Already have an account? </Text>
          <TouchableOpacity onPress={() => navigation.navigate('Login')} disabled={isLoading}>
            <Text style={styles.switchLink}>Sign in</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: C.bg },
  hero: { backgroundColor: C.red },
  studRow: {
    flexDirection: 'row', paddingHorizontal: 8, paddingVertical: 2, gap: 2,
    backgroundColor: C.redDark,
  },
  stud: { flex: 1, height: 10, borderRadius: 5, backgroundColor: 'rgba(255,255,255,0.15)' },
  heroContent: { alignItems: 'center', paddingVertical: 28, paddingHorizontal: S.lg },
  logoBlock: {
    width: 64, height: 64, borderRadius: 16,
    backgroundColor: C.yellow,
    alignItems: 'center', justifyContent: 'center',
    marginBottom: 12, ...shadow(2),
  },
  logoText: { fontSize: 24, fontWeight: '900', color: C.red, letterSpacing: -1 },
  appName: { fontSize: 30, fontWeight: '900', color: C.white, letterSpacing: -1, marginBottom: 4 },
  tagline: { fontSize: 13, color: 'rgba(255,255,255,0.75)', fontWeight: '500' },

  formScroll: { flex: 1 },
  formContent: { padding: S.lg, paddingTop: S.xl },
  formTitle: { fontSize: 22, fontWeight: '800', color: C.text, marginBottom: S.lg, letterSpacing: -0.5 },

  errorBox: {
    backgroundColor: '#FFF0F0', borderWidth: 1, borderColor: '#FFCCCC',
    borderRadius: R.md, padding: 12, marginBottom: S.md,
  },
  errorText: { color: '#CC0000', fontSize: 13, fontWeight: '500' },

  fieldGroup: { marginBottom: S.md },
  label: { fontSize: 13, fontWeight: '600', color: C.textSub, marginBottom: 6, letterSpacing: 0.2 },
  input: {
    backgroundColor: C.white, borderWidth: 1.5, borderColor: C.border,
    borderRadius: R.md, paddingHorizontal: S.md, paddingVertical: 13,
    fontSize: 15, color: C.text, ...shadow(1),
  },
  inputFocused: { borderColor: C.red },
  primaryBtn: {
    backgroundColor: C.red, borderRadius: R.md, paddingVertical: 15,
    alignItems: 'center', marginTop: S.sm, marginBottom: S.lg, ...shadow(2),
  },
  btnDisabled: { opacity: 0.6 },
  primaryBtnText: { color: C.white, fontSize: 16, fontWeight: '700', letterSpacing: 0.3 },
  switchRow: { flexDirection: 'row', justifyContent: 'center', alignItems: 'center' },
  switchText: { fontSize: 14, color: C.textSub },
  switchLink: { fontSize: 14, fontWeight: '700', color: C.red },
});
