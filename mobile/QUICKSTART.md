# BrickScan Mobile App - Quick Start

## 5-Minute Setup

### 1. Install Dependencies
```bash
cd /sessions/adoring-clever-goodall/mnt/Lego/brickscan/mobile
npm install
```

### 2. Create Environment File
```bash
cp .env.example .env
```

Edit `.env` and set your API URL:
```
EXPO_PUBLIC_API_URL=http://localhost:3000/api
```

### 3. Start Development Server
```bash
npm start
```

### 4. Run on iPhone
- **Simulator**: Press `i`
- **Device**: Scan QR code with iPhone camera, tap link to open in Expo Go app

## File Organization

```
📱 Mobile App (React Native + Expo)
│
├── 🔐 Auth (LoginScreen, RegisterScreen)
│
├── 📸 Scanning
│   ├── ScanScreen (camera interface)
│   └── ScanResultScreen (AI predictions)
│
├── 📦 Inventory
│   └── InventoryScreen (grid of pieces)
│
├── 🧱 Sets
│   ├── SetsScreen (search & browse)
│   ├── SetDetailScreen (parts list)
│   └── BuildCheckScreen (completion %)
│
└── 👤 Profile (user info & export)
```

## Key Technologies

| Tech | Purpose | File |
|------|---------|------|
| **React Native** | Mobile UI framework | `src/screens/*.tsx` |
| **Expo** | Development & deployment | `app.json` |
| **TypeScript** | Type safety | `src/types/index.ts` |
| **React Navigation v6** | Tab & stack navigation | `src/navigation/index.tsx` |
| **Zustand** | State management (auth & inventory) | `src/store/*.ts` |
| **TanStack Query** | Server state (sets & search) | Used in `SetsScreen.tsx` |
| **NativeWind** | Tailwind CSS for React Native | All screen files |
| **Axios** | HTTP client with interceptors | `src/services/api.ts` |
| **expo-camera** | Camera access & photo capture | `ScanScreen.tsx` |
| **expo-secure-store** | Secure token storage | `src/store/authStore.ts` |

## Core Workflows

### 1. Authentication
```
LoginScreen
  ↓ (user enters email/password)
  ↓ apiClient.login()
  ↓ Token stored in secure store
  ↓ User state saved in Zustand
  ↓ Navigation → MainTabs
```

### 2. Scanning a Piece
```
ScanScreen (camera view)
  ↓ (user captures photo)
  ↓ Convert to base64
  ↓ apiClient.scanImage(base64)
  ↓ Get predictions with confidence %
  ↓ ScanResultScreen (show top 3)
  ↓ (user selects quantity)
  ↓ apiClient.addToInventory()
  ↓ Zustand optimistic update
  ↓ Success alert
```

### 3. Checking Set Progress
```
SetsScreen (search)
  ↓ (user finds set)
  ↓ SetDetailScreen (view parts)
  ↓ (user taps "Check if I can build this")
  ↓ BuildCheckScreen (call compareToSet API)
  ↓ Shows % complete with missing parts
  ↓ (user can generate BrickLink list)
```

## Component Map

### Screens (8 main screens)
- `LoginScreen`: Email/password login form
- `RegisterScreen`: Create new account
- `ScanScreen`: Full-screen camera interface
- `ScanResultScreen`: Shows AI predictions
- `InventoryScreen`: Grid of user's pieces
- `SetsScreen`: Search and filter sets
- `SetDetailScreen`: Set info and parts list
- `BuildCheckScreen`: Build progress & missing parts
- `ProfileScreen`: User info and settings

### Components (3 reusable)
- `PartCard`: Displays a LEGO part with image, name, color, quantity
- `SetCard`: Displays a set with image, name, year, piece count
- `LoadingOverlay`: Full-screen spinner with optional message

### Stores (2 state managers)
- `authStore`: Login/logout, token persistence, user info
- `inventoryStore`: Add/update/delete pieces, local caching

### Services (1 API layer)
- `apiClient`: All backend calls, JWT injection, error handling

## API Integration Points

The app expects these endpoints (see `API_ENDPOINTS.md` for details):

**Auth**
- `POST /auth/login` → get JWT token
- `POST /auth/register` → create account

**Scanning**
- `POST /scan` → submit base64 image, get predictions

**Inventory**
- `GET /inventory` → fetch all user pieces
- `POST /inventory` → add new piece
- `PATCH /inventory/{id}` → update quantity
- `DELETE /inventory/{id}` → remove piece
- `GET /inventory/export/csv` → download CSV

**Sets**
- `GET /sets/search?q=...&theme=...` → search sets
- `GET /sets/{setNum}` → get set with all parts
- `POST /builds/check` → check if can build set
- `POST /bricklink/wanted-list` → generate XML

## Styling Guide

All screens use Tailwind CSS via NativeWind:

```tsx
// Container with padding, flex layout
<View className="flex-1 bg-white px-4 py-6">
  {/* Primary button */}
  <TouchableOpacity className="bg-primary rounded-lg py-4">
    <Text className="text-white font-bold text-center">Button</Text>
  </TouchableOpacity>

  {/* Title */}
  <Text className="text-xl font-bold text-gray-800">Title</Text>

  {/* Subtitle */}
  <Text className="text-gray-600">Subtitle</Text>
</View>
```

**Color System:**
- `primary`: `#FF6B00` (orange) - main CTA buttons
- `secondary`: `#2D3436` (dark gray) - secondary buttons
- `accent`: `#00B894` (green) - success/export
- `danger`: `#D63031` (red) - delete/logout

## Common Tasks

### Change API URL
Edit `.env`:
```
EXPO_PUBLIC_API_URL=https://api.brickscan.com
```

### Add a New Screen
1. Create `src/screens/NewScreen.tsx`
2. Add to navigation in `src/navigation/index.tsx`
3. Add TypeScript types in `src/types/index.ts`

### Add API Endpoint
1. Add method to `src/services/api.ts`
2. Add type to `src/types/index.ts`
3. Use with `apiClient.methodName()`

### Update State
Use Zustand hooks:
```tsx
const user = useAuthStore((state) => state.user);
const logout = useAuthStore((state) => state.logout);
```

## Debugging

### View Logs
Logs appear in Expo CLI output:
```
2024-01-15 10:30:45 [info] App started
```

### Access State
React DevTools support via Expo CLI (`shift+m`)

### Network Requests
- All requests logged via axios
- 401 responses trigger logout
- 5xx errors show Alert dialogs

## Building for Production

```bash
# Build iOS app
eas build --platform ios

# Submit to App Store
eas submit --platform ios
```

Update `app.json`:
- Increment `version` (1.0.0 → 1.0.1)
- Update `buildNumber` if needed
- Verify bundle identifier: `com.brickscan.app`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera permission denied | Grant permission in Settings > BrickScan |
| "API_URL not set" | Copy `.env.example` to `.env` |
| Login fails (401) | Check API is running and URL in `.env` |
| Blank screen on startup | Restart Expo CLI with `npm start` |
| Inventory not updating | Check network tab, ensure POST request succeeds |
| Navigation not working | Ensure all screens imported in `src/navigation/index.tsx` |

## Next Steps

1. ✅ Install dependencies (`npm install`)
2. ✅ Set up `.env` with API URL
3. ✅ Run dev server (`npm start`)
4. ✅ Test on simulator/device
5. ✅ Connect to backend API
6. ✅ Test all 4 tabs (Scan, Inventory, Sets, Profile)
7. ✅ Customize app icons in `assets/` folder
8. ✅ Build for production (`eas build --platform ios`)

## File Sizes

- Source code: ~2,500 lines of TypeScript/React Native
- Dependencies: ~500 packages
- Build size: ~100MB (iOS app bundle)

## Performance

- Images lazy-loaded with fallback placeholders
- Inventory items cached with TanStack Query
- Optimistic updates for instant feedback
- Navigation animations use Reanimated for 60fps
