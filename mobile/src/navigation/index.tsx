import React, { useEffect } from 'react';
import { View, Text, StyleSheet, Platform } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons';
import { useAuthStore } from '@/store/authStore';
import { LoginScreen } from '@/screens/auth/LoginScreen';
import { RegisterScreen } from '@/screens/auth/RegisterScreen';
import { ScanScreen } from '@/screens/ScanScreen';
import { ScanResultScreen } from '@/screens/ScanResultScreen';
import { FeedbackStatsScreen } from '@/screens/FeedbackStatsScreen';
import { ReviewQueueScreen } from '@/screens/ReviewQueueScreen';
import { ContinuousScanScreen } from '@/screens/ContinuousScanScreen';
import { MultiResultScreen } from '@/screens/MultiResultScreen';
import PileScanScreen from '@/screens/PileScanScreen';
import { ScanHistoryScreen } from '@/screens/ScanHistoryScreen';
import { PartDetailScreen } from '@/screens/PartDetailScreen';
import { SettingsScreen } from '@/screens/SettingsScreen';
import { OnboardingScreen } from '@/screens/OnboardingScreen';
import { InventoryScreen } from '@/screens/InventoryScreen';
import { SetsScreen } from '@/screens/SetsScreen';
import { SetDetailScreen } from '@/screens/SetDetailScreen';
import { BuildCheckScreen } from '@/screens/BuildCheckScreen';
import { ProfileScreen } from '@/screens/ProfileScreen';
import { C, R, shadow } from '@/constants/theme';
import type {
  RootStackParamList,
  AuthStackParamList,
  ScanStackParamList,
  SetsStackParamList,
  InventoryStackParamList,
  ProfileStackParamList,
  OnboardingStackParamList,
} from '@/types';

const AuthStack = createNativeStackNavigator<AuthStackParamList>();
const RootStack = createNativeStackNavigator<RootStackParamList>();
const ScanStack = createNativeStackNavigator<ScanStackParamList>();
const SetsStack = createNativeStackNavigator<SetsStackParamList>();
const InventoryStack = createNativeStackNavigator<InventoryStackParamList>();
const ProfileStack = createNativeStackNavigator<ProfileStackParamList>();
const OnboardingStack = createNativeStackNavigator<OnboardingStackParamList>();
const Tab = createBottomTabNavigator();

// ── Custom tab bar icon with label ──────────────────────────────────────────
type TabIconProps = {
  name: React.ComponentProps<typeof Ionicons>['name'];
  label: string;
  focused: boolean;
  isScan?: boolean;
};

const TabIcon = ({ name, label, focused, isScan }: TabIconProps) => {
  if (isScan) {
    return (
      <View style={tabStyles.scanBtn}>
        <Ionicons name={name} size={26} color={C.white} />
      </View>
    );
  }
  return (
    <View style={tabStyles.tabItem}>
      <Ionicons name={name} size={22} color={focused ? C.red : C.textMuted} />
      <Text style={[tabStyles.tabLabel, focused && tabStyles.tabLabelActive]}>{label}</Text>
    </View>
  );
};

// ── Navigators ───────────────────────────────────────────────────────────────
const AuthStackNavigator = () => (
  <AuthStack.Navigator screenOptions={{ headerShown: false }}>
    <AuthStack.Screen name="Login" component={LoginScreen} />
    <AuthStack.Screen name="Register" component={RegisterScreen} />
  </AuthStack.Navigator>
);

const ScanHistoryWrapper: React.FC<any> = ({ navigation }) => (
  <ScanHistoryScreen onSelectScan={(scan) => {
    // Restore full result screen — use saved allPredictions if available,
    // otherwise fall back to a single-prediction list from top-level fields.
    const predictions = scan.allPredictions && scan.allPredictions.length > 0
      ? scan.allPredictions.map((p: any) => ({
          partNum: p.partNum,
          partName: p.partName,
          colorId: p.colorId || '',
          colorName: p.colorName || '',
          colorHex: p.colorHex || '',
          confidence: p.confidence,
          imageUrl: p.imageUrl,
          source: p.source,
        }))
      : [{
          partNum: scan.partNum,
          partName: scan.partName,
          colorId: scan.colorId || '',
          colorName: scan.colorName || '',
          colorHex: scan.colorHex || '',
          confidence: scan.confidence,
          imageUrl: scan.imageUrl,
          source: scan.source,
        }];

    navigation.navigate('ScanResultScreen', {
      predictions,
      scanMode: scan.scanMode,
      framesAnalyzed: scan.framesAnalyzed,
      agreementScore: scan.agreementScore,
    });
  }} />
);

const ScanStackNavigator = () => (
  <ScanStack.Navigator screenOptions={{ headerShown: false }}>
    <ScanStack.Screen name="ScanScreen" component={ScanScreen} />
    <ScanStack.Screen name="ScanResultScreen" component={ScanResultScreen} />
    <ScanStack.Screen name="MultiResultScreen" component={MultiResultScreen} />
    <ScanStack.Screen name="PileScanScreen" component={PileScanScreen} />
    <ScanStack.Screen name="ScanHistoryScreen" component={ScanHistoryWrapper} />
    <ScanStack.Screen name="PartDetailScreen" component={PartDetailScreen} />
    <ScanStack.Screen name="FeedbackStatsScreen" component={FeedbackStatsScreen} />
    <ScanStack.Screen
      name="ReviewQueueScreen"
      component={ReviewQueueScreen}
      options={{ title: 'Review Queue', headerShown: true }}
    />
    <ScanStack.Screen
      name="ContinuousScanScreen"
      component={ContinuousScanScreen}
      options={{ headerShown: false }}
    />
  </ScanStack.Navigator>
);

const SetsStackNavigator = () => (
  <SetsStack.Navigator
    screenOptions={{
      headerShown: true,
      headerBackTitle: 'Back',
      headerStyle: { backgroundColor: C.white },
      headerTitleStyle: { fontWeight: '700', color: C.text },
      headerTintColor: C.red,
    }}
  >
    <SetsStack.Screen name="SetsScreen" component={SetsScreen} options={{ headerShown: false }} />
    <SetsStack.Screen name="SetDetailScreen" component={SetDetailScreen} options={{ headerTitle: 'Set Details' }} />
    <SetsStack.Screen name="BuildCheckScreen" component={BuildCheckScreen} options={{ headerTitle: 'Build Progress' }} />
  </SetsStack.Navigator>
);

const InventoryStackNavigator = () => (
  <InventoryStack.Navigator screenOptions={{ headerShown: false }}>
    <InventoryStack.Screen name="InventoryScreen" component={InventoryScreen} />
    <InventoryStack.Screen name="PartDetailScreen" component={PartDetailScreen} />
  </InventoryStack.Navigator>
);

const ProfileStackNavigator = () => (
  <ProfileStack.Navigator screenOptions={{ headerShown: false }}>
    <ProfileStack.Screen name="ProfileScreen" component={ProfileScreen} />
    <ProfileStack.Screen name="SettingsScreen" component={SettingsScreen} />
  </ProfileStack.Navigator>
);

const OnboardingNavigator = () => (
  <OnboardingStack.Navigator screenOptions={{ headerShown: false }}>
    <OnboardingStack.Screen name="OnboardingScreen" component={OnboardingScreen} />
  </OnboardingStack.Navigator>
);

const MainTabNavigator = () => (
  <Tab.Navigator
    screenOptions={{
      headerShown: false,
      tabBarShowLabel: false,
      tabBarStyle: tabStyles.tabBar,
    }}
  >
    <Tab.Screen
      name="ScanTab"
      component={ScanStackNavigator}
      options={{
        tabBarIcon: ({ focused }) => (
          <TabIcon name="camera" label="Scan" focused={focused} isScan />
        ),
      }}
    />
    <Tab.Screen
      name="InventoryTab"
      component={InventoryStackNavigator}
      options={{
        tabBarIcon: ({ focused }) => (
          <TabIcon
            name={focused ? 'cube' : 'cube-outline'}
            label="Inventory"
            focused={focused}
          />
        ),
      }}
    />
    <Tab.Screen
      name="SetsTab"
      component={SetsStackNavigator}
      options={{
        tabBarIcon: ({ focused }) => (
          <TabIcon
            name={focused ? 'layers' : 'layers-outline'}
            label="Sets"
            focused={focused}
          />
        ),
      }}
    />
    <Tab.Screen
      name="ProfileTab"
      component={ProfileStackNavigator}
      options={{
        tabBarIcon: ({ focused }) => (
          <TabIcon
            name={focused ? 'person' : 'person-outline'}
            label="Profile"
            focused={focused}
          />
        ),
      }}
    />
  </Tab.Navigator>
);

// ── Root ─────────────────────────────────────────────────────────────────────
export const RootNavigator = () => {
  const isLoggedIn = useAuthStore((s) => s.isLoggedIn);
  const loadStoredAuth = useAuthStore((s) => s.loadStoredAuth);
  const loadOnboardingState = useAuthStore((s) => s.loadOnboardingState);
  const isLoading = useAuthStore((s) => s.isLoading);
  const hasOnboarded = useAuthStore((s) => s.hasOnboarded);

  useEffect(() => {
    loadStoredAuth();
    loadOnboardingState();
  }, []);

  if (isLoading || hasOnboarded === null) return null;

  return (
    <NavigationContainer>
      <RootStack.Navigator screenOptions={{ headerShown: false }}>
        {!hasOnboarded ? (
          <RootStack.Screen name="OnboardingStack" component={OnboardingNavigator} />
        ) : isLoggedIn ? (
          <RootStack.Screen name="MainTabs" component={MainTabNavigator} />
        ) : (
          <RootStack.Screen name="AuthStack" component={AuthStackNavigator} />
        )}
      </RootStack.Navigator>
    </NavigationContainer>
  );
};

// ── Tab bar styles ────────────────────────────────────────────────────────────
const tabStyles = StyleSheet.create({
  tabBar: {
    backgroundColor: C.white,
    borderTopWidth: 1,
    borderTopColor: C.border,
    height: Platform.OS === 'ios' ? 82 : 64,
    paddingBottom: Platform.OS === 'ios' ? 24 : 8,
    paddingTop: 8,
    ...shadow(2),
  },
  tabItem: { alignItems: 'center', justifyContent: 'center', gap: 3 },
  tabLabel: { fontSize: 10, fontWeight: '600', color: C.textMuted },
  tabLabelActive: { color: C.red },
  // Scan CTA button — prominent floating style
  scanBtn: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: C.red,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Platform.OS === 'ios' ? 6 : 0,
    ...shadow(3),
    shadowColor: C.red,
    shadowOpacity: 0.4,
  },
});
