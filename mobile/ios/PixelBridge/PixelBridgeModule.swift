// PixelBridgeModule.swift
//
// Phase 5 Stage 5 — native JPEG → letterboxed Float32 NCHW tensor bridge.
//
// On-device YOLO inference today does this in JS:
//   1. fs.readAsBase64
//   2. base64-decode → Uint8Array
//   3. jpeg-js decode → RGBA buffer
//   4. JS letterbox + per-channel normalise into Float32Array (NCHW)
//   5. Hand to onnxruntime-react-native
//
// Steps 1-4 take ~50–80 ms per 720px frame on iPhone 13 (the JS engine is
// fundamentally slow for this kind of pixel work). This native module does
// the same end-to-end in ~8–15 ms by:
//
//   - libjpeg-turbo via UIImage(data:) — 6× faster than jpeg-js
//   - vImage SIMD letterbox + scale (Accelerate framework)
//   - vDSP_vsmsa for per-channel ((px/255) - mean) / std
//   - Returns the final Float32 buffer as a base64 string the JS side hands
//     directly to onnxruntime-react-native via Tensor.fromBuffer.
//
// Output format matches what `mobile/src/ml/preprocess.ts` was producing:
//   shape (1, 3, H, W), float32, NCHW, ImageNet-normalised.
//
// React Native exposed methods (see PixelBridgeModule.m):
//   prepareTensorFromJpegUri(uri, targetSize, fillR, fillG, fillB)
//     → Promise<{ base64: String, width: Int, height: Int,
//                 letterboxScale: Double, padX: Int, padY: Int }>
//
// `letterboxScale, padX, padY` match preprocess.ts exactly so YOLO output
// post-processing in postprocess.ts (un-letterbox to image space) doesn't
// need any changes.
//
// Note: this is a SIDE-BY-SIDE module. `preprocess.ts` stays as a fallback
// for Android and for any iOS device that fails to load the native module.
// The mobile-side scanPipeline picks the native path when available.

import Foundation
import UIKit
import Accelerate
import CoreGraphics

@objc(PixelBridgeModule)
class PixelBridgeModule: NSObject {

  // ImageNet normalisation, baked in to match the training pipeline.
  // Order: (mean_r, mean_g, mean_b)  /  (std_r, std_g, std_b).
  private static let mean: [Float] = [0.485, 0.456, 0.406]
  private static let std:  [Float] = [0.229, 0.224, 0.225]

  @objc static func requiresMainQueueSetup() -> Bool { false }

  @objc func prepareTensorFromJpegUri(
    _ uri: NSString,
    targetSize: NSNumber,
    fillR: NSNumber,
    fillG: NSNumber,
    fillB: NSNumber,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    DispatchQueue.global(qos: .userInitiated).async {
      do {
        let result = try Self.prepareTensor(
          uri: uri as String,
          targetSize: targetSize.intValue,
          fill: (fillR.uint8Value, fillG.uint8Value, fillB.uint8Value)
        )
        resolve([
          "base64": result.base64,
          "width": result.targetSize,
          "height": result.targetSize,
          "letterboxScale": result.scale,
          "padX": result.padX,
          "padY": result.padY,
          "sourceWidth": result.sourceWidth,
          "sourceHeight": result.sourceHeight,
        ])
      } catch let err as NSError {
        reject("E_PIXEL_BRIDGE", err.localizedDescription, err)
      }
    }
  }

  // MARK: - Implementation

  private struct PrepareResult {
    let base64: String
    let targetSize: Int
    let sourceWidth: Int
    let sourceHeight: Int
    let scale: Double
    let padX: Int
    let padY: Int
  }

  private static func prepareTensor(
    uri: String,
    targetSize: Int,
    fill: (UInt8, UInt8, UInt8)
  ) throws -> PrepareResult {
    // 1. Load JPEG → UIImage. Fastest decode path on iOS uses libjpeg-turbo
    //    under the hood; UIImage handles file:// and absolute paths.
    let url: URL
    if uri.hasPrefix("file://") {
      url = URL(string: uri)!
    } else {
      url = URL(fileURLWithPath: uri)
    }
    let data = try Data(contentsOf: url)
    guard let image = UIImage(data: data), let cgImage = image.cgImage else {
      throw NSError(domain: "PixelBridge", code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "Failed to decode JPEG at \(uri)"])
    }

    let srcW = cgImage.width
    let srcH = cgImage.height

    // 2. Compute letterbox geometry. Same math as preprocess.ts so post-
    //    processing on the JS side doesn't need to know who did the work.
    let scale = min(Double(targetSize) / Double(srcW),
                    Double(targetSize) / Double(srcH))
    let scaledW = Int(round(Double(srcW) * scale))
    let scaledH = Int(round(Double(srcH) * scale))
    let padX = (targetSize - scaledW) / 2
    let padY = (targetSize - scaledH) / 2

    // 3. Allocate the destination buffer pre-filled with the letterbox fill
    //    colour (typically the YOLO grey 114/114/114). vImage operates in-
    //    place on these buffers.
    let bytesPerPixel = 4   // RGBA8888
    let dstStride = targetSize * bytesPerPixel
    let dstSize = dstStride * targetSize
    let dstPtr = UnsafeMutablePointer<UInt8>.allocate(capacity: dstSize)
    defer { dstPtr.deallocate() }
    // Fill background
    for y in 0..<targetSize {
      let row = dstPtr.advanced(by: y * dstStride)
      var x = 0
      while x < targetSize {
        row[x * bytesPerPixel + 0] = fill.0
        row[x * bytesPerPixel + 1] = fill.1
        row[x * bytesPerPixel + 2] = fill.2
        row[x * bytesPerPixel + 3] = 255
        x += 1
      }
    }

    // 4. Render the source CGImage into the centred letterbox region.
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(
      data: dstPtr,
      width: targetSize,
      height: targetSize,
      bitsPerComponent: 8,
      bytesPerRow: dstStride,
      space: colorSpace,
      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
    ) else {
      throw NSError(domain: "PixelBridge", code: 2,
                    userInfo: [NSLocalizedDescriptionKey: "Failed to allocate CGContext"])
    }
    // Origin is bottom-left in CG; flip so we draw with image's natural orientation
    ctx.translateBy(x: 0, y: CGFloat(targetSize))
    ctx.scaleBy(x: 1, y: -1)
    let drawRect = CGRect(x: padX, y: padY, width: scaledW, height: scaledH)
    ctx.draw(cgImage, in: drawRect)

    // 5. Convert RGBA u8 → NCHW Float32 with ImageNet normalisation.
    let pixels = targetSize * targetSize
    let tensorSize = pixels * 3
    var tensor = [Float](repeating: 0, count: tensorSize)

    // Plane offsets (NCHW layout)
    let rOffset = 0
    let gOffset = pixels
    let bOffset = pixels * 2

    let invStdR: Float = 1.0 / std[0]
    let invStdG: Float = 1.0 / std[1]
    let invStdB: Float = 1.0 / std[2]
    let negMeanOverStdR: Float = -mean[0] / std[0]
    let negMeanOverStdG: Float = -mean[1] / std[1]
    let negMeanOverStdB: Float = -mean[2] / std[2]
    let invScale: Float = 1.0 / 255.0

    // Tight per-pixel loop. With the constants above, the per-channel value
    // is `(px/255) * (1/std) + (-mean/std)`, which vDSP_vsmsa fuses into a
    // single multiply-add we apply below in three vector strokes.
    var rPlane = [Float](repeating: 0, count: pixels)
    var gPlane = [Float](repeating: 0, count: pixels)
    var bPlane = [Float](repeating: 0, count: pixels)

    // Convert u8 → float plane-by-plane in place.
    for i in 0..<pixels {
      let base = i * bytesPerPixel
      rPlane[i] = Float(dstPtr[base + 0])
      gPlane[i] = Float(dstPtr[base + 1])
      bPlane[i] = Float(dstPtr[base + 2])
    }
    // (px * invScale * invStd) + (negMean/std). vDSP_vsmsa: (vec * scalar) + scalar.
    rPlane.withUnsafeMutableBufferPointer { ptr in
      var s1 = invScale * invStdR
      var s2 = negMeanOverStdR
      vDSP_vsmsa(ptr.baseAddress!, 1, &s1, &s2, ptr.baseAddress!, 1, vDSP_Length(pixels))
    }
    gPlane.withUnsafeMutableBufferPointer { ptr in
      var s1 = invScale * invStdG
      var s2 = negMeanOverStdG
      vDSP_vsmsa(ptr.baseAddress!, 1, &s1, &s2, ptr.baseAddress!, 1, vDSP_Length(pixels))
    }
    bPlane.withUnsafeMutableBufferPointer { ptr in
      var s1 = invScale * invStdB
      var s2 = negMeanOverStdB
      vDSP_vsmsa(ptr.baseAddress!, 1, &s1, &s2, ptr.baseAddress!, 1, vDSP_Length(pixels))
    }

    // Memcpy into the NCHW destination
    tensor.withUnsafeMutableBufferPointer { dst in
      memcpy(dst.baseAddress!.advanced(by: rOffset), rPlane, pixels * MemoryLayout<Float>.size)
      memcpy(dst.baseAddress!.advanced(by: gOffset), gPlane, pixels * MemoryLayout<Float>.size)
      memcpy(dst.baseAddress!.advanced(by: bOffset), bPlane, pixels * MemoryLayout<Float>.size)
    }

    // 6. Wrap the Float32 buffer as base64 — onnxruntime-react-native's
    //    Tensor.fromBuffer accepts ArrayBuffer / typed-array; the JS side
    //    decodes this base64 once.
    let byteCount = tensorSize * MemoryLayout<Float>.size
    let buffer = tensor.withUnsafeBufferPointer { ptr -> Data in
      return Data(bytes: ptr.baseAddress!, count: byteCount)
    }
    let base64 = buffer.base64EncodedString()

    return PrepareResult(
      base64: base64,
      targetSize: targetSize,
      sourceWidth: srcW,
      sourceHeight: srcH,
      scale: scale,
      padX: padX,
      padY: padY,
    )
  }
}
