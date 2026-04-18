export interface User {
  id: string;
  email: string;
  createdAt?: string;
}

export interface AuthState {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  isLoggedIn: boolean;
}

export interface Part {
  id: string;
  partNum: string;
  name: string;
  category: string;
  imageUrl?: string;
}

export interface ColorInfo {
  id: string;
  name: string;
  hex: string;
  rgb?: string;
}

export interface InventoryItem {
  id: string;
  partNum: string;
  partName: string;
  colorId: string;
  colorName: string;
  colorHex: string;
  quantity: number;
  imageUrl?: string;
  createdAt: string;
  updatedAt: string;
}

export interface SetSummary {
  setNum: string;
  name: string;
  year: number;
  theme: string;
  numParts: number;
  imageUrl?: string;
}

export interface SetPart {
  partNum: string;
  partName: string;
  colorId: string;
  colorName: string;
  colorHex: string;
  quantity: number;
  imageUrl?: string;
}

export interface SetDetail extends SetSummary {
  description?: string;
  parts: SetPart[];
}

export interface PartSubstitute {
  partNum: string;
  name: string;
  similarity: number;
  reason: string;
  imageUrl?: string;
}

export interface ScanPrediction {
  partNum: string;
  partName: string;
  colorId: string;
  colorName: string;
  colorHex: string;
  confidence: number;
  imageUrl?: string;
  source?: string;
}

export interface ScanResult {
  predictions: ScanPrediction[];
  scan_id?: string;
  thumbnail_url?: string;
}

export interface VideoScanResult {
  predictions: ScanPrediction[];
  framesAnalyzed: number;
  agreementScore: number;
  status: string;
}

export interface DetectedPiece {
  pieceIndex: number;
  predictions: ScanPrediction[];
  primaryPrediction: ScanPrediction;
  bbox?: number[];
}

export interface MultiPieceScanResult {
  piecesDetected: number;
  pieces: DetectedPiece[];
  status: string;
}

export interface MissingPart {
  partNum: string;
  partName: string;
  colorId: string;
  colorName: string;
  colorHex: string;
  quantityNeeded: number;
  quantityHave: number;
  imageUrl?: string;
}

export interface BuildCheckResult {
  setNum: string;
  setName: string;
  percentComplete: number;
  have: number;
  total: number;
  missing: number;
  haveParts: SetPart[];
  missingParts: MissingPart[];
}

export interface SetCompletionMissingPart {
  part_num: string;
  part_name: string;
  color_id: string;
  color_name: string;
  color_hex?: string;
  need: number;
  have: number;
  short: number;
}

export interface SetCompletionResult {
  set_num: string;
  set_name: string;
  total_parts: number;
  unique_parts: number;
  have_parts: number;
  have_unique: number;
  completion_pct: number;
  missing: SetCompletionMissingPart[];
  cached_at: string;
}

export interface ValueBreakdownColor {
  color_name: string;
  value_usd: number;
  count: number;
}

export interface TopValuablePart {
  part_num: string;
  part_name: string;
  color_name: string;
  unit_price: number;
  qty: number;
  total: number;
}

export interface InventoryValueResult {
  total_value_usd: number;
  total_parts: number;
  unique_parts: number;
  breakdown_by_color: ValueBreakdownColor[];
  breakdown_by_theme: any[];
  top_valuable_parts: TopValuablePart[];
  cached_at: string;
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export type RootStackParamList = {
  MainTabs: undefined;
  AuthStack: undefined;
  OnboardingStack: undefined;
};

export type AuthStackParamList = {
  Login: undefined;
  Register: undefined;
};

export type MainTabsParamList = {
  ScanTab: undefined;
  InventoryTab: undefined;
  SetsTab: undefined;
  ProfileTab: undefined;
};

export type ScanStackParamList = {
  ScanScreen: undefined;
  ScanResultScreen: {
    predictions: ScanPrediction[];
    scanMode?: 'photo' | 'video' | 'multi';
    framesAnalyzed?: number;
    agreementScore?: number;
  };
  MultiResultScreen: {
    pieces: DetectedPiece[];
  };
  PileScanScreen: undefined;
  ScanHistoryScreen: undefined;
  PartDetailScreen: {
    partNum: string;
    partName?: string;
    imageUrl?: string;
    colorId?: string;
    colorName?: string;
    colorHex?: string;
    confidence?: number;
  };
  FeedbackStatsScreen: undefined;
  ReviewQueueScreen: undefined;
  ContinuousScanScreen: undefined;
};

export type SetsStackParamList = {
  SetsScreen: undefined;
  SetDetailScreen: {
    setNum: string;
  };
  BuildCheckScreen: {
    setNum: string;
  };
};

export type InventoryStackParamList = {
  InventoryScreen: undefined;
  PartDetailScreen: {
    partNum: string;
    partName?: string;
    imageUrl?: string;
    colorId?: string;
    colorName?: string;
    colorHex?: string;
    confidence?: number;
  };
};

export type ProfileStackParamList = {
  ProfileScreen: undefined;
  SettingsScreen: undefined;
};

export type OnboardingStackParamList = {
  OnboardingScreen: undefined;
};
