// Augments React Native's minimal global `URL` interface (declared in
// react-native/src/types/globals.d.ts) with `hostname`, which RN's URL
// polyfill implements at runtime but omits from its type definitions.
// Avoids pulling in the full DOM lib (which would alter other global typings).
// Based on the definition in lib.dom.d.ts.
declare global {
  interface URL {
    hostname: string;
  }
}

export {};
