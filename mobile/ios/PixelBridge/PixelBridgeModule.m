#import <React/RCTBridgeModule.h>

@interface RCT_EXTERN_MODULE(PixelBridgeModule, NSObject)

RCT_EXTERN_METHOD(
  prepareTensorFromJpegUri:(NSString *)uri
  targetSize:(nonnull NSNumber *)targetSize
  fillR:(nonnull NSNumber *)fillR
  fillG:(nonnull NSNumber *)fillG
  fillB:(nonnull NSNumber *)fillB
  resolver:(RCTPromiseResolveBlock)resolve
  rejecter:(RCTPromiseRejectBlock)reject
)

@end
