#import <React/RCTBridgeModule.h>

@interface RCT_EXTERN_MODULE(DepthCaptureModule, NSObject)

RCT_EXTERN_METHOD(
  isDepthAvailable:(RCTPromiseResolveBlock)resolve
  reject:(RCTPromiseRejectBlock)reject
)

RCT_EXTERN_METHOD(
  captureRGBD:(RCTPromiseResolveBlock)resolve
  reject:(RCTPromiseRejectBlock)reject
)

@end
