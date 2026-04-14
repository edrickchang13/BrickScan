# BrickScan iOS App - Setup Guide

## Project Overview

BrickScan is a complete iOS LEGO scanning and inventory management app built with React Native, Expo, TypeScript, React Navigation v6, Zustand, TanStack Query, and NativeWind/Tailwind CSS.

## File Structure

```
brickscan/mobile/
├── App.tsx                          # Root component with providers
├── app.json                         # Expo configuration
├── babel.config.js                  # Babel with NativeWind plugin
├── tailwind.config.js               # Tailwind CSS configuration
├── tsconfig.json                    # TypeScript configuration
├── package.json                     # Dependencies
├── .env.example                     # Environment variables template
│
├── src/
│   ├── navigation/
│   │   └── index.tsx               # Navigation setup (auth/main stacks)
│   │
│   ├── screens/
│   │   ├── auth/
│   │   │   ├── LoginScreen.tsx     # Email/password login
│   │   │   └── RegisterScreen.tsx  # User registration
│   │   │
│   │   ├── ScanScreen.tsx          # Camera interface for scanning
│   │   ├── ScanResultScreen.tsx    # Shows scan results with predictions
│   │   ├── InventoryScreen.tsx     # Lists user's scanned pieces
│   │   ├── SetsScreen.tsx          # Search and browse LEGO sets
│   │   ├── SetDetailScreen.tsx     # Set details and parts list
│   │   ├── BuildCheckScreen.tsx    # Check if you can build a set
│   │   └── ProfileScreen.tsx       # User profile and settings
│   │
│   ├── components/
│   │   ├── PartCard.tsx            # Reusable part display card
│   │   ├── SetCard.tsx             # Reusable set display card
│   │   └── LoadingOverlay.tsx      # Full-screen loading indicator
│   │
│   ├── services/
│   │   └── api.ts                  # Axios API client with interceptors
│   │
│   ├── store/
│   │   ├── authStore.ts            # Zustand auth state management
│   │   └── inventoryStore.ts       # Zustand inventory state with optimistic updates
│   │
│   └── types/
│       └── index.ts                # All TypeScript type definitions
```

## Installation

1. Install dependencies:
```bash
npm install
```

2. Create `.env` file from template:
```bash
cp .env.example .env
```

3. Update `.env` with your API URL:
```
EXPO_PUBLIC_API_URL=http://your-backend.com/api
```

## Running the App

Start the Expo development server:
```bash
npm start
```

Then:
- Press `i` to open in iOS simulator
- Scan QR code with iPhone to open on device
- Use `expo go` app on iPhone

## Architecture

### Authentication Flow
- `LoginScreen` and `RegisterScreen` use `useAuthStore` 
- Tokens stored securely in `expo-secure-store`
- Auto-load stored auth on app launch
- Routes shown based on `isLoggedIn` state

### Data Management
- **Auth State**: Zustand store with secure token persistence
- **Inventory**: Zustand store with optimistic updates and API sync
- **API Calls**: TanStack Query for data fetching and caching
- **Navigation**: React Navigation v6 with bottom tab navigator

### Core Features
1. **Scan Tab**: Full-screen camera with image capture and AI scanning
2. **Inventory Tab**: Grid list of scanned pieces with search/filter
3. **Sets Tab**: Search LEGO sets by name/theme and browse details
4. **Build Check**: Compare inventory to set parts, generate BrickLink lists
5. **Profile**: User info, stats, export, and logout

## Key Screens

### ScanScreen
- Full-screen camera using `expo-camera`
- Circular capture button at bottom
- Overlay instructions and frame guide
- Converts photo to base64 for API
- Navigates to `ScanResultScreen` with predictions

### ScanResultScreen
- Shows top prediction with large image and stats
- Alternative predictions in bottom list
- Quantity selector modal (±1 buttons)
- "Add to Inventory" and "That's Not Right" buttons
- Optimistic inventory updates

### InventoryScreen
- 2-column grid of `PartCard` components
- Search by part name/number
- Filter by color
- Long-press to delete
- Tap to update quantity
- Export CSV button
- Pull-to-refresh

### SetsScreen
- Search bar and horizontal theme filter chips
- 2-column grid of `SetCard` components
- Pull-to-refresh
- Navigate to `SetDetailScreen`

### SetDetailScreen
- Large set image
- Set info (number, year, theme, part count)
- Wishlist toggle
- "Check if I can build this" button
- FlatList of all parts needed

### BuildCheckScreen
- Circular progress indicator (%)
- Stats row (have/need/missing)
- Expandable sections for parts
- "Generate BrickLink List" button
- BrickLink XML modal with copy function

## Styling

All screens use NativeWind Tailwind classes:
- Primary color: `#FF6B00` (orange)
- Secondary: `#2D3436` (dark gray)
- Accent: `#00B894` (green)
- Danger: `#D63031` (red)

Examples:
```tsx
<View className="flex-1 bg-white px-4 py-6">
  <Text className="text-xl font-bold text-primary">Title</Text>
</View>
```

## API Integration

The `apiClient` handles:
- Base URL from env variable
- JWT token injection in request headers
- 401 logout handling
- All endpoints typed with TypeScript

Example usage:
```typescript
const result = await apiClient.scanImage(base64);
// Returns: { predictions: [{partNum, confidence, ...}] }
```

## Environment Setup

### Required for iOS Build:
```json
{
  "ios": {
    "bundleIdentifier": "com.brickscan.app",
    "infoPlist": {
      "NSCameraUsageDescription": "We need camera access to scan LEGO pieces"
    }
  }
}
```

### Permissions Handled:
- Camera (via `expo-camera`)
- Photo library (via `expo-image-picker`)
- Secure storage (via `expo-secure-store`)

## Development Tips

### Hot Reload
Changes auto-reload in simulator/device

### Debugging
- Use `console.log()` visible in Expo CLI
- React DevTools support via Expo
- Redux/Zustand DevTools available

### Common Issues
1. Camera permission denied: Request permission explicitly
2. API 401: Token expired, logout user
3. Image not loading: Check CDN URLs in API responses

## Building for Production

```bash
# Build for iOS
eas build --platform ios

# Submit to App Store
eas submit --platform ios
```

## Dependencies

- **react-native-gesture-handler**: Touch handling
- **react-native-reanimated**: Smooth animations
- **react-native-screens**: Performance optimization
- **react-native-safe-area-context**: Safe area handling
- **expo-camera**: Camera access
- **expo-image-picker**: Photo library access
- **expo-secure-store**: Secure token storage
- **zustand**: State management
- **@tanstack/react-query**: Server state management
- **axios**: HTTP client
- **nativewind**: Tailwind CSS for React Native
- **@react-navigation/v6**: Navigation routing

## Next Steps

1. Update `EXPO_PUBLIC_API_URL` in `.env`
2. Run `npm install` to install dependencies
3. Update app icons in `assets/` directory
4. Test on iOS simulator with `npm start`
5. Connect backend API for full functionality

## Notes

- All code is fully typed with TypeScript
- Uses functional components and hooks throughout
- Optimistic updates for better UX
- Error handling with Alert dialogs
- Loading states for all async operations
- Responsive design with Tailwind classes
