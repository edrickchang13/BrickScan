import CoreML
import Vision
import UIKit
import Foundation

@available(iOS 16.0, *)
class BrickClassifier {
    private let model: MLModel
    private let queue = DispatchQueue(label: "com.brickscan.classifier", qos: .userInitiated)

    // MARK: - Initialization

    init(modelName: String = "BrickClassifier") throws {
        guard let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlpackage") else {
            throw ClassifierError.modelNotFound(modelName)
        }

        let compiledModelURL = try MLModel.compileModel(at: modelURL)
        self.model = try MLModel(contentsOf: compiledModelURL)
    }

    init(modelURL: URL) throws {
        let compiledModelURL = try MLModel.compileModel(at: modelURL)
        self.model = try MLModel(contentsOf: compiledModelURL)
    }

    // MARK: - Prediction

    struct BrickPrediction {
        let partNum: String
        let colorId: Int
        let colorName: String
        let confidence: Float
    }

    func classify(image: UIImage, topK: Int = 5) throws -> [BrickPrediction] {
        var result: [BrickPrediction] = []

        queue.sync {
            do {
                // Convert UIImage to CVPixelBuffer
                guard let pixelBuffer = image.toCVPixelBuffer() else {
                    throw ClassifierError.pixelBufferConversionFailed
                }

                // Create input
                let input = BrickClassifierInput(image: pixelBuffer)

                // Run prediction
                let output = try model.prediction(from: input)

                // Extract logits
                guard let partLogits = output.featureValue(for: "part_logits")?.multiArrayValue,
                      let colorLogits = output.featureValue(for: "color_logits")?.multiArrayValue else {
                    throw ClassifierError.invalidModelOutput
                }

                // Convert to arrays
                let partArray = partLogits.toArray()
                let colorArray = colorLogits.toArray()

                // Load class maps from metadata
                let partMap = parsePartMap()
                let colorMap = parseColorMap()

                // Softmax both outputs
                let partProbs = softmax(partArray)
                let colorProbs = softmax(colorArray)

                // Get top-K part predictions
                let topIndices = partProbs.enumerated()
                    .map { ($0.offset, $0.element) }
                    .sorted { $0.1 > $1.1 }
                    .prefix(topK)
                    .map { $0.0 }

                // For each top part, pair with highest color confidence
                let bestColorIdx = colorProbs.enumerated()
                    .max(by: { $0.element < $1.element })?
                    .offset ?? 0

                for partIdx in topIndices {
                    let partNum = partMap[String(partIdx)] ?? "UNKNOWN"
                    let colorId = colorMap[String(bestColorIdx)] ?? -1
                    let colorName = getColorName(colorId: colorId)
                    let confidence = partProbs[partIdx]

                    result.append(BrickPrediction(
                        partNum: partNum,
                        colorId: colorId,
                        colorName: colorName,
                        confidence: confidence
                    ))
                }
            } catch {
                print("Classification error: \(error)")
            }
        }

        return result
    }

    // MARK: - Helpers

    private func parsePartMap() -> [String: String] {
        guard let metadata = model.modelDescription.metadata as? [String: String],
              let partMapJSON = metadata["part_map"],
              let data = partMapJSON.data(using: .utf8),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: String] else {
            return [:]
        }
        return dict
    }

    private func parseColorMap() -> [String: Int] {
        guard let metadata = model.modelDescription.metadata as? [String: String],
              let colorMapJSON = metadata["color_map"],
              let data = colorMapJSON.data(using: .utf8),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Int] else {
            return [:]
        }
        return dict
    }

    private func getColorName(colorId: Int) -> String {
        let colorMap: [Int: String] = [
            0: "Black", 1: "White", 2: "Red", 3: "Green", 4: "Blue",
            5: "Yellow", 6: "Brown", 7: "Gray", 8: "Orange", 9: "Pink"
        ]
        return colorMap[colorId] ?? "Unknown"
    }

    private func softmax(_ logits: [Float]) -> [Float] {
        guard !logits.isEmpty else { return [] }
        let maxLogit = logits.max() ?? 0
        let expLogits = logits.map { exp($0 - maxLogit) }
        let sum = expLogits.reduce(0, +)
        // Guard against divide-by-zero / NaN if all logits are -inf or array is degenerate.
        guard sum > 0, sum.isFinite else {
            return [Float](repeating: 0, count: logits.count)
        }
        return expLogits.map { $0 / sum }
    }
}

// MARK: - UIImage Extension

extension UIImage {
    func toCVPixelBuffer() -> CVPixelBuffer? {
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary

        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(self.size.width),
            Int(self.size.height),
            kCVPixelFormatType_32ARGB,
            attrs,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let pixelBuffer = pixelBuffer else {
            return nil
        }

        // CVPixelBufferLockFlags has only `.readOnly`. For read-write access,
        // pass an empty option set (== default behavior, writable).
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(pixelBuffer),
            width: Int(self.size.width),
            height: Int(self.size.height),
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else {
            return nil
        }

        guard let cgImage = self.cgImage else {
            // UIImage can be backed by CIImage (e.g. filtered images); no CGImage available.
            return nil
        }
        context.draw(cgImage, in: CGRect(origin: .zero, size: self.size))

        return pixelBuffer
    }
}

// MARK: - MLMultiArray Extension

extension MLMultiArray {
    func toArray() -> [Float] {
        let pointer = UnsafeMutablePointer<Float>(OpaquePointer(dataPointer))
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }
}

// MARK: - BrickClassifierInput

// NOTE: MLFeatureProvider is a class-only (NSObjectProtocol) protocol, so this
// must be a `class`, not a `struct`. Marking as `final` to keep the lightweight
// value-type intent.
@available(iOS 16.0, *)
final class BrickClassifierInput: NSObject, MLFeatureProvider {
    let image: CVPixelBuffer

    init(image: CVPixelBuffer) {
        self.image = image
        super.init()
    }

    var featureNames: Set<String> {
        return ["image"]
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        guard featureName == "image" else { return nil }
        return MLFeatureValue(pixelBuffer: image)
    }
}

// MARK: - Errors

enum ClassifierError: Error {
    case modelNotFound(String)
    case pixelBufferConversionFailed
    case invalidModelOutput
}
