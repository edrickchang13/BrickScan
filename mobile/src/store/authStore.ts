import { create } from 'zustand';
import * as SecureStore from 'expo-secure-store';
import { apiClient } from '@/services/api';
import { User, AuthState } from '@/types';

const TOKEN_KEY = 'brickscan_token';
const USER_KEY = 'brickscan_user';

interface AuthStoreState extends AuthState {
  hasOnboarded: boolean | null;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  loadStoredAuth: () => Promise<void>;
  loadOnboardingState: () => Promise<void>;
  completeOnboarding: () => Promise<void>;
  setUser: (user: User | null) => void;
  continueAsGuest: () => Promise<void>;
}

export const useAuthStore = create<AuthStoreState>((set, get) => ({
  user: null,
  token: null,
  isLoading: false,
  isLoggedIn: false,
  hasOnboarded: null,

  login: async (email: string, password: string) => {
    set({ isLoading: true });
    try {
      const response = await apiClient.login(email, password);
      const { token, user } = response;

      await SecureStore.setItemAsync(TOKEN_KEY, token);
      await SecureStore.setItemAsync(USER_KEY, JSON.stringify(user));

      set({
        token,
        user,
        isLoggedIn: true,
        isLoading: false,
      });
    } catch (error) {
      set({ isLoading: false });
      throw error;
    }
  },

  register: async (email: string, password: string) => {
    set({ isLoading: true });
    try {
      const response = await apiClient.register(email, password);
      const { token, user } = response;

      await SecureStore.setItemAsync(TOKEN_KEY, token);
      await SecureStore.setItemAsync(USER_KEY, JSON.stringify(user));

      set({
        token,
        user,
        isLoggedIn: true,
        isLoading: false,
      });
    } catch (error) {
      set({ isLoading: false });
      throw error;
    }
  },

  logout: async () => {
    try {
      await apiClient.logout();
    } catch (error) {
      console.error('Logout error:', error);
    }

    await SecureStore.deleteItemAsync(TOKEN_KEY);
    await SecureStore.deleteItemAsync(USER_KEY);

    set({
      token: null,
      user: null,
      isLoggedIn: false,
      isLoading: false,
    });
  },

  loadStoredAuth: async () => {
    set({ isLoading: true });
    try {
      const token = await SecureStore.getItemAsync(TOKEN_KEY);
      const userJson = await SecureStore.getItemAsync(USER_KEY);

      if (token && userJson) {
        try {
          const user = JSON.parse(userJson);
          set({
            token,
            user,
            isLoggedIn: true,
            isLoading: false,
          });
        } catch (parseError) {
          console.error('Failed to parse stored user data:', parseError);
          // Clear invalid data and default to logged-out state
          await SecureStore.deleteItemAsync(TOKEN_KEY);
          await SecureStore.deleteItemAsync(USER_KEY);
          set({ isLoading: false });
        }
      } else {
        set({ isLoading: false });
      }
    } catch (error) {
      console.error('Load stored auth error:', error);
      // Default to logged-out state on any error to prevent hanging
      set({
        token: null,
        user: null,
        isLoggedIn: false,
        isLoading: false,
      });
    }
  },

  loadOnboardingState: async () => {
    const val = await (await import('@react-native-async-storage/async-storage')).default.getItem('brickscan_onboarded');
    set({ hasOnboarded: val === 'true' });
  },

  completeOnboarding: async () => {
    await (await import('@react-native-async-storage/async-storage')).default.setItem('brickscan_onboarded', 'true');
    set({ hasOnboarded: true });
  },

  setUser: (user: User | null) => {
    set({ user });
  },

  continueAsGuest: async () => {
    const guestUser: User = { id: 'guest', email: 'guest@local' };
    set({ user: guestUser, token: 'guest', isLoggedIn: true, isLoading: false });
  },
}));
