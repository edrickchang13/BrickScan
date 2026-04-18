const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Exclude ios/Pods from the file watcher to avoid ENOENT errors on symlinked boost headers
config.watchFolders = config.watchFolders || [];
config.resolver = config.resolver || {};
config.resolver.blockList = [
  /ios\/Pods\/.*/,
  /android\/\.gradle\/.*/,
];

// Treat .onnx as a bundleable asset so Expo Asset can resolve the
// on-device YOLO model via require(). Without this, Metro throws
// "Unable to resolve module" for require('./yolo_lego.int8.onnx').
const defaultAssetExts = config.resolver.assetExts || [];
if (!defaultAssetExts.includes('onnx')) {
  config.resolver.assetExts = [...defaultAssetExts, 'onnx'];
}

module.exports = config;
