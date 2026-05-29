/**
 * Jest config for the BrickScan mobile app.
 *
 * The existing suites (src/__tests__/ml/* + utils/*) are PURE TypeScript unit
 * tests — none import react-native, expo-*, or any screen/store/service. So we
 * deliberately AVOID the `jest-expo` preset (its setup.js hard-requires
 * `expo-modules-core`, which isn't a dependency here, and pulls in the whole RN
 * test environment these tests don't need). Instead we run them in the `node`
 * environment with a lightweight Babel transform:
 *   - @babel/preset-typescript  → strip TS types
 *   - babel-plugin-module-resolver → the `@` → ./src alias (mirrors babel.config.js)
 *
 * JSON imports (e.g. colorClassifier's bundled color_model.json) resolve via
 * Jest's native JSON support. If a future test needs the RN runtime, install
 * `expo-modules-core` and switch that test's project over to the `jest-expo`
 * preset rather than burdening these pure ones with it.
 */
module.exports = {
  testEnvironment: 'node',
  setupFiles: ['<rootDir>/jest/setup.js'],
  testMatch: ['**/src/__tests__/**/*.test.ts'],
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
    // Pure-logic suites import these only at module-load (never exercise their
    // native behaviour), so stub them to keep the node env RN-free. See
    // jest/mocks/* for why.
    '^react-native$': '<rootDir>/jest/mocks/react-native.js',
    '^expo-image-manipulator$': '<rootDir>/jest/mocks/empty.js',
    '^expo-file-system(/.*)?$': '<rootDir>/jest/mocks/empty.js',
  },
  transform: {
    '^.+\\.[jt]sx?$': [
      'babel-jest',
      {
        // Self-contained transform — does NOT read babel.config.js (which uses
        // babel-preset-expo / RN plugins). Keep in sync with the `@` alias.
        babelrc: false,
        configFile: false,
        presets: [
          ['@babel/preset-typescript'],
        ],
        plugins: [
          // Jest runs CommonJS; transform ESM import/export down to require().
          '@babel/plugin-transform-modules-commonjs',
          ['module-resolver', { alias: { '@': './src' } }],
        ],
      },
    ],
  },
};
