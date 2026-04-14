const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Exclude ios/Pods from the file watcher to avoid ENOENT errors on symlinked boost headers
config.watchFolders = config.watchFolders || [];
config.resolver = config.resolver || {};
config.resolver.blockList = [
  /ios\/Pods\/.*/,
  /android\/\.gradle\/.*/,
];

module.exports = config;
