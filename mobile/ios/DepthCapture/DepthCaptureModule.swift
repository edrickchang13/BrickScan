import Foundation
import AVFoundation
import ARKit
import Photos
import UIKit          // UIImage
import Accelerate     // vImage_Buffer, vImage_CGImageFormat, vImagePixelCount
import CoreGraphics

@objc(DepthCaptureModule)
class DepthCaptureModule: NSObject {

  private var captureSession: AVCaptureSession?
  private var depthDataOutput: AVCaptureDepthDataOutput?
  private var videoDataOutput: AVCaptureVideoDataOutput?
  private var lastRGBFrame: CVImageBuffer?
  private var lastDepthFrame: CVImageBuffer?
  private var captureComplete = false
  private let captureQueue = DispatchQueue(label: "com.brickscan.depth-capture")

  // MARK: - Public Methods

  @objc func isDepthAvailable(
    _ resolve: @escaping RCTPromiseResolveBlock,
    reject: @escaping RCTPromiseRejectBlock
  ) {
    // Check if ARKit with depth is available.
    // `supportsFrameSemantics(_:)` is a class method — don't instantiate.
    // For LiDAR RGBD we need `.sceneDepth` (iPhone 12 Pro / iPad Pro 2020+).
    let isAvailable = ARWorldTrackingConfiguration.isSupported &&
                      ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)

    resolve(isAvailable)
  }

  @objc func captureRGBD(
    _ resolve: @escaping RCTPromiseResolveBlock,
    reject: @escaping RCTPromiseRejectBlock
  ) {
    // Perform depth capture on background queue
    captureQueue.async { [weak self] in
      self?.performRGBDCapture(resolve: resolve, reject: reject)
    }
  }

  // MARK: - Private Methods

  private func performRGBDCapture(
    resolve: @escaping RCTPromiseResolveBlock,
    reject: @escaping RCTPromiseRejectBlock
  ) {
    do {
      // Setup capture session
      let session = AVCaptureSession()
      session.sessionPreset = .high

      // Add input device
      guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
            device.supportsSessionPreset(.high) else {
        reject("DEPTH_ERROR", "Camera device not available", nil)
        return
      }

      try device.lockForConfiguration()
      // Request depth data if available
      if device.activeDepthDataFormat == nil {
        let depthFormats = device.activeFormat.supportedDepthDataFormats
        if let depthFormat = depthFormats.first {
          device.activeDepthDataFormat = depthFormat
        }
      }
      device.unlockForConfiguration()

      let input = try AVCaptureDeviceInput(device: device)
      guard session.canAddInput(input) else {
        reject("DEPTH_ERROR", "Cannot add camera input", nil)
        return
      }
      session.addInput(input)

      // Setup depth data output
      let depthOutput = AVCaptureDepthDataOutput()
      depthOutput.isFilteringEnabled = true
      depthOutput.setDelegate(self, callbackQueue: captureQueue)

      guard session.canAddOutput(depthOutput) else {
        reject("DEPTH_ERROR", "Cannot add depth output", nil)
        return
      }
      session.addOutput(depthOutput)

      // Setup video data output (for RGB frames)
      let videoOutput = AVCaptureVideoDataOutput()
      videoOutput.videoSettings = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
      ]
      videoOutput.setSampleBufferDelegate(self, queue: captureQueue)

      guard session.canAddOutput(videoOutput) else {
        reject("DEPTH_ERROR", "Cannot add video output", nil)
        return
      }
      session.addOutput(videoOutput)

      // Synchronize video and depth outputs — we only need to assert the
      // connection exists; the reference itself isn't used here.
      guard videoOutput.connection(with: .video) != nil else {
        reject("DEPTH_ERROR", "Cannot setup video connection", nil)
        return
      }

      // Start capturing
      lastRGBFrame = nil
      lastDepthFrame = nil
      captureComplete = false
      self.captureSession = session
      self.depthDataOutput = depthOutput
      self.videoDataOutput = videoOutput

      session.startRunning()

      // Wait for synchronized frames (timeout after 5 seconds)
      let deadline = Date().addingTimeInterval(5.0)
      while !captureComplete && Date() < deadline {
        Thread.sleep(forTimeInterval: 0.05)
      }

      session.stopRunning()

      // Process frames
      guard let rgbFrame = lastRGBFrame, let depthFrame = lastDepthFrame else {
        reject("DEPTH_ERROR", "Failed to capture synchronized RGB/depth frames", nil)
        return
      }

      do {
        let rgbPath = try saveRGBFrame(rgbFrame)
        let depthPath = try saveDepthFrame(depthFrame)

        DispatchQueue.main.async {
          resolve([
            "rgbPath": rgbPath,
            "depthPath": depthPath
          ])
        }
      } catch {
        reject("DEPTH_ERROR", "Failed to save frames: \(error.localizedDescription)", nil)
      }
    } catch {
      reject("DEPTH_ERROR", "Capture setup failed: \(error.localizedDescription)", nil)
    }
  }

  private func saveRGBFrame(_ pixelBuffer: CVImageBuffer) throws -> String {
    // Convert CVImageBuffer (BGRA) to UIImage
    let ciImage = CIImage(cvImageBuffer: pixelBuffer)
    let context = CIContext()
    guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
      throw NSError(domain: "DepthCapture", code: 1, userInfo: nil)
    }

    let uiImage = UIImage(cgImage: cgImage)
    guard let jpegData = uiImage.jpegData(compressionQuality: 0.7) else {
      throw NSError(domain: "DepthCapture", code: 2, userInfo: nil)
    }

    // Write to temp file
    let tempDir = NSTemporaryDirectory()
    let fileName = "depth_rgb_\(UUID().uuidString).jpg"
    let filePath = (tempDir as NSString).appendingPathComponent(fileName)

    try jpegData.write(to: URL(fileURLWithPath: filePath))
    return filePath
  }

  private func saveDepthFrame(_ depthBuffer: CVImageBuffer) throws -> String {
    // Ensure depth buffer is in 16-bit format
    let depthPixelFormat = CVPixelBufferGetPixelFormatType(depthBuffer)

    // Convert depth to 16-bit grayscale if needed
    var depthImageBuffer = depthBuffer
    if depthPixelFormat != kCVPixelFormatType_DepthFloat32 &&
       depthPixelFormat != kCVPixelFormatType_DisparityFloat32 {
      // Already in appropriate format
    }

    let width = CVPixelBufferGetWidth(depthBuffer)
    let height = CVPixelBufferGetHeight(depthBuffer)

    // Lock base address for reading
    CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }

    // Create 16-bit depth PNG
    guard width > 0, height > 0 else {
      throw NSError(
        domain: "DepthCapture", code: -10,
        userInfo: [NSLocalizedDescriptionKey: "Empty depth buffer dimensions"]
      )
    }
    guard let depthData = NSMutableData(capacity: width * height * 2) else {
      throw NSError(
        domain: "DepthCapture", code: -11,
        userInfo: [NSLocalizedDescriptionKey: "Failed to allocate depth buffer (out of memory?)"]
      )
    }

    guard let baseAddress = CVPixelBufferGetBaseAddress(depthBuffer) else {
      throw NSError(
        domain: "DepthCapture", code: -12,
        userInfo: [NSLocalizedDescriptionKey: "Depth pixel buffer has no base address"]
      )
    }
    let bytesPerRow = CVPixelBufferGetBytesPerRow(depthBuffer)

    for y in 0..<height {
      let rowBase = baseAddress.advanced(by: y * bytesPerRow)

      if depthPixelFormat == kCVPixelFormatType_DepthFloat32 {
        // Convert float32 depth to uint16 (in mm, clamped to 0-4000mm)
        let row = rowBase.assumingMemoryBound(to: Float32.self)
        for x in 0..<width {
          let depthMM = min(max(row[x] * 1000.0, 0), 4000)
          var uint16Value = UInt16(depthMM)
          depthData.append(&uint16Value, length: 2)
        }
      } else {
        // Assume it's already 16-bit
        depthData.append(rowBase, length: width * 2)
      }
    }

    // Create grayscale image and encode as PNG.
    // vImage_CGImageFormat init is failable on iOS 13+ so we use `guard`.
    // The type for vImage_Buffer width/height is `vImagePixelCount` (not `vUInt`).
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageByteOrderInfo.order16Little.rawValue)
    guard var format = vImage_CGImageFormat(
      bitsPerComponent: 16,
      bitsPerPixel: 16,
      colorSpace: CGColorSpaceCreateDeviceGray(),
      bitmapInfo: bitmapInfo
    ) else {
      throw NSError(domain: "DepthCapture", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to build vImage format"])
    }

    var buffer = vImage_Buffer(
      data: UnsafeMutableRawPointer(mutating: depthData.bytes),
      height: vImagePixelCount(height),
      width: vImagePixelCount(width),
      rowBytes: width * 2
    )
    var vErr: vImage_Error = kvImageNoError
    guard let cgImage = vImageCreateCGImageFromBuffer(
      &buffer, &format, nil, nil, vImage_Flags(kvImageNoFlags), &vErr
    )?.takeRetainedValue() else {
      throw NSError(domain: "DepthCapture", code: 3,
                    userInfo: [NSLocalizedDescriptionKey: "vImageCreateCGImageFromBuffer failed (\(vErr))"])
    }

    let uiImage = UIImage(cgImage: cgImage)
    guard let pngData = uiImage.pngData() else {
      throw NSError(domain: "DepthCapture", code: 4, userInfo: nil)
    }

    // Write to temp file
    let tempDir = NSTemporaryDirectory()
    let fileName = "depth_map_\(UUID().uuidString).png"
    let filePath = (tempDir as NSString).appendingPathComponent(fileName)

    try pngData.write(to: URL(fileURLWithPath: filePath))
    return filePath
  }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension DepthCaptureModule: AVCaptureVideoDataOutputSampleBufferDelegate {
  func captureOutput(
    _ output: AVCaptureOutput,
    didOutput sampleBuffer: CMSampleBuffer,
    from connection: AVCaptureConnection
  ) {
    guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
    lastRGBFrame = pixelBuffer
    checkCaptureComplete()
  }
}

// MARK: - AVCaptureDepthDataOutputDelegate

extension DepthCaptureModule: AVCaptureDepthDataOutputDelegate {
  func depthDataOutput(
    _ output: AVCaptureDepthDataOutput,
    didOutput depthData: AVDepthData,
    timestamp: CMTime,
    connection: AVCaptureConnection
  ) {
    let depthPixelBuffer = depthData.depthDataMap
    lastDepthFrame = depthPixelBuffer
    checkCaptureComplete()
  }

  private func checkCaptureComplete() {
    if lastRGBFrame != nil && lastDepthFrame != nil && !captureComplete {
      captureComplete = true
    }
  }
}
