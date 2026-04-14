# BrickScan — Development Guide

This document explains how the dev environment works, what breaks and why, and how to recover from every known failure mode.

---

## Quick Start

```bash
cd mobile
npm install
npm run dev          # starts Metro with auto-detected IP and keepalive
```

Then in Xcode: **Product → Run** (or ⌘R) with your iPhone selected.

> **Phone not plugged in?** That's fine — Xcode will build and install to the most recently connected device. Metro connection happens at runtime.

---

## Architecture: How Metro Connection Works

```
Mac                                   iPhone
────────────────────────────────────────────────────────
start-dev.sh                          AppDelegate.mm
  └─ detect_ip() → 169.254.x.x          └─ resolveMetroHost()
  └─ REACT_NATIVE_PACKAGER_HOSTNAME         ├─ scanForMetroHost()
  └─ expo start --port 8081                 │   └─ getifaddrs → probe all IPs
                                            │       concurrently via HEAD /status
                                            ├─ RCTBundleURLProvider saved host
                                            └─ localhost (simulator fallback)
```

The app **never relies on a baked-in IP**. Instead, at every launch it probes all local network interfaces simultaneously and uses whichever one responds first. This means:

- Reconnecting the USB cable → new IP assigned → app finds it automatically on next launch
- Cable unplugged during development → app falls back to WiFi or localhost
- No rebuild required when your IP changes

---

## Why USB Link-Local Addresses (169.254.x.x)?

When you connect an iPhone via USB, macOS creates a point-to-point (P2P) network interface. The OS assigns a **link-local** IP in the `169.254.0.0/16` range to both ends using APIPA. This IP:

- Is not routable beyond the cable
- Changes on every cable plug/unplug cycle
- Is assigned to whichever `en*` slot macOS has free (en1, en4, en6, en8 — varies)

This is why hardcoding `en6` or a specific `169.254.x.x` IP always breaks eventually.

---

## All Known Failure Modes & Fixes

### 1. "Could not connect to development server" after cable reconnect

**Root cause:** USB link-local IP changed (APIPA reassignment). The app was using a cached/hardcoded IP.

**Fix applied:** `AppDelegate.mm` uses `getifaddrs` to enumerate all live interfaces at app launch, including `ifa_dstaddr` on P2P interfaces (which gives the Mac's side of the USB link). It probes all candidates concurrently.

**Recovery:** Just relaunch the app (no rebuild needed). Metro's `resolveMetroHost` will find the new IP.

---

### 2. Metro builds an stale IP into the binary (MetroHost.h)

**Root cause:** The "Detect USB Metro Host" Xcode build phase ran `detect_metro_host.sh`, which wrote the current `169.254.x.x` into `MetroHost.h`. On next cable reconnect the IP changes but the binary keeps the old one.

**Fix applied:** The "Detect USB Metro Host" build phase has been **removed from `project.pbxproj`** entirely. `MetroHost.h` is no longer generated or used. Runtime detection in `AppDelegate.mm` handles everything.

**Note:** `detect_metro_host.sh` still exists but is no longer called by Xcode. You can ignore it or delete it.

---

### 3. Build fails when phone is disconnected

**Root cause:** Some build phase scripts were querying the attached device and would fail if no device was connected.

**Fix applied:** Removed the USB-detection build phase (see #2). Remaining build phases are device-independent. You can build targeting "Any iOS Device (arm64)" with no phone attached.

---

### 4. App crashes on camera open

**Root cause:** `expo-camera`'s `PreviewView.swift` called `fatalError()` when `AVCaptureVideoPreviewLayer` wasn't the layer class, which can happen in React Native's UIView lifecycle.

**Historical fix:** SDK 51 previously used a local `patch-package` patch for `expo-camera` to avoid this crash.

**Current state:** After upgrading to Expo SDK 55, that legacy patch was removed because it no longer matches the installed `expo-camera` version.

---

### 5. TypeScript errors: `barStyle` on `StatusBar`

**Root cause:** `expo-status-bar`'s `StatusBar` component uses a `style` prop with values `"auto" | "inverted" | "light" | "dark"`, not `barStyle` (which is the React Native core component's API).

**Fix applied:** `App.tsx` uses `<StatusBar style="dark" />`.

---

### 6. TypeScript errors: `describe`/`it`/`expect` undefined

**Root cause:** `@types/jest` missing from devDependencies, and `tsconfig.json` didn't include it in `types`.

**Fix applied:** Installed `@types/jest`, added `"types": ["jest"]` to `tsconfig.json`.

---

### 7. Metro keepalive / Metro dies when you close Terminal

**Root cause:** No process supervision — Metro stops when the shell exits.

**Fix applied:** `start-dev.sh` wraps Metro in a keepalive loop (up to 20 restarts). It re-detects the IP on every restart, so a cable reconnect between restarts is handled automatically.

---

### 8. `detect_metro_host.sh` crashes with `pipefail` + `grep` on no-match

**Root cause:** `set -euo pipefail` causes the script to exit when `grep "inet "` returns exit code 1 (no match), because the pipe propagates the failure.

**Fix applied (historical):** Changed to `set -eu` and added `|| true` guards. This script is no longer run by Xcode, but the fix is preserved so it works if run manually.

---

### 9. Xcode incremental build doesn't recompile after `MetroHost.h` changes

**Root cause:** Xcode's dependency tracker only recompiles `AppDelegate.mm` if it or its listed inputs change. If only `MetroHost.h` changed (written by the build phase), Xcode didn't always trigger a recompile.

**Fix applied:** The build phase is gone; `MetroHost.h` is no longer used. Moot.

---

### 10. Wrong Metro host saved in `NSUserDefaults` / `RCTBundleURLProvider`

**Root cause:** A previous session baked in a stale IP as `jsLocation`, and subsequent launches read it before the runtime scanner ran.

**Fix applied:** `resolveMetroHost` tries the runtime scan *first*, and only falls back to the saved `jsLocation` if the scan finds nothing. The saved value is also checked with a live probe before trusting it.

**Force-clear:** Reinstall the app (`xcrun devicectl device uninstall app --device <UUID> com.yourcompany.BrickScan`) to wipe `NSUserDefaults`.

---

### 11. Metro probe timeouts too short for USB link-local

**Root cause:** USB link-local interfaces take a moment to become routable after cable reconnect. The original 1.5 s timeout was too short; the interface would answer after 2–3 s.

**Fix applied:** Per-probe timeout increased to 3.0 s with a 3.5 s wall-clock guard. Each IP is retried once (with a 1 s pause) before giving up. Group wait increased to 8 s.

---

### 12. Simulator device not found in `xcodebuild`

**Root cause:** iOS 26 Simulator runtime doesn't include "iPhone 15 Pro Max" by default; the simulator model name changed.

**Fix applied:** Use the simulator's UUID directly: `xcodebuild -destination "id=<UUID>"`. Find available simulators with `xcrun simctl list devices available`.

---

### 13. `start-dev.sh` hardcodes `en6` for USB detection

**Root cause:** macOS assigns USB tether to the first free `en*` slot. On different Macs or after different plug sequences, this can be en1, en4, en6, or en8.

**Fix applied:** `detect_ip()` now scans all interfaces for `169.254.x.x` addresses using `ifconfig | grep "inet 169.254."` — completely interface-name-agnostic.

---

## Daily Workflow

### Starting development (phone plugged in)

```bash
# Terminal 1 — Metro
cd mobile
npm run dev

# Xcode — Build & Run
# Product → Run (⌘R)
# The app will find Metro automatically
```

### Phone disconnected (simulator / "Any iOS Device")

```bash
# Metro still runs — the app will probe and fall back to WiFi or localhost
npm run dev
```

```
# Xcode — select "Any iOS Device (arm64)" and build
# This creates an .ipa you can install when the phone is next connected
```

### IP changed after cable reconnect

```
# Nothing to do — just relaunch the app on the phone.
# resolveMetroHost() will scan and find the new IP.
# If Metro is still running on the Mac, it will connect.
```

### Metro died (Terminal closed, etc.)

```bash
npm run dev   # restart Metro; it re-detects the current IP
```

### Full reset (when something is deeply broken)

```bash
# Kill Metro and clear port
lsof -ti tcp:8081 | xargs kill 2>/dev/null || true

# Clean iOS build artifacts
cd mobile/ios
xcodebuild clean -workspace BrickScan.xcworkspace -scheme BrickScan

# Wipe node_modules and reinstall dependencies
cd ..
rm -rf node_modules
npm install

# In Xcode: Product → Clean Build Folder (⌘⇧K), then Build (⌘B)
```

---

## File Reference

| File | Purpose |
|------|---------|
| `ios/BrickScan/AppDelegate.mm` | Metro host resolution at runtime using `getifaddrs` |
| `start-dev.sh` | Metro launcher with IP detection and keepalive |
| `ios/detect_metro_host.sh` | Legacy script, no longer called by Xcode (can be deleted) |
| `ios/BrickScan/MetroHost.h` | Legacy header, no longer generated or used (can be deleted) |

---

## Metro Connection Probe Logic (AppDelegate.mm)

```
resolveMetroHost()
  │
  ├─ scanForMetroHost()            ← always runs first
  │   └─ allLocalIPv4Addresses()
  │       ├─ All non-loopback interface IPs
  │       └─ P2P peer addresses (ifa_dstaddr) — this is the Mac's USB IP
  │
  │   For each candidate (concurrently, dispatch_group):
  │       isMetroReachableAt(ip)
  │           HEAD http://<ip>:8081/status
  │           timeout: 3.0 s per attempt, 2 attempts, 8 s group max
  │
  ├─ RCTBundleURLProvider.sharedSettings.jsLocation  ← if scan found nothing
  │   └─ isMetroReachableAt(savedHost)               ← live-probe before trusting
  │
  └─ "localhost"                   ← final fallback (simulator / WiFi via loopback)
```
