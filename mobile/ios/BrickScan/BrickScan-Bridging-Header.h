//
//  Use this file to import your target's public headers that you would like to expose to Swift.
//

// React Native bridge types — needed so DepthCaptureModule.swift can use
// RCTPromiseResolveBlock / RCTPromiseRejectBlock / @objc(RCT_EXTERN_MODULE) symbols.
#import <React/RCTBridgeModule.h>
#import <React/RCTEventEmitter.h>
#import <React/RCTViewManager.h>
