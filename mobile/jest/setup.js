/**
 * Jest setup — defines the React-Native/Metro global `__DEV__` that some app
 * modules (e.g. constants/config.ts) reference. Metro injects this at build
 * time; the RN-free node test env doesn't, so we set it here. `true` matches a
 * dev build (the branch the tests exercise).
 */
globalThis.__DEV__ = true;
