/**
 * Minimal `react-native` mock for the PURE-LOGIC unit tests.
 *
 * The real `react-native` entrypoint ships Flow-typed source that Jest's
 * (deliberately RN-free) Babel transform can't parse, and these suites only
 * touch dependency-free util functions — they never render or call native
 * APIs. We stub the handful of named exports the util modules import at load
 * time (Share, Linking, Platform, NativeModules) so `require('react-native')`
 * resolves without dragging in the RN runtime. Add to this only when a NEW
 * pure test transitively needs another RN named export.
 */
const noop = () => {};

module.exports = {
  Share: { share: async () => ({ action: 'sharedAction' }) },
  Linking: { openURL: async () => {}, canOpenURL: async () => true },
  Platform: { OS: 'ios', select: (o) => (o ? o.ios ?? o.default : undefined) },
  NativeModules: {},
  StyleSheet: { create: (s) => s, absoluteFillObject: {} },
  Alert: { alert: noop },
};
