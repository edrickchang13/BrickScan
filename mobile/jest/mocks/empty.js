/**
 * Empty-module stub for native/Expo modules that the PURE-LOGIC unit tests
 * import only transitively (e.g. expo-image-manipulator, expo-file-system).
 * The tested functions don't call into them at load or in the assertions, so
 * an empty object is enough to let `require(...)` succeed in the node env.
 */
module.exports = {};
