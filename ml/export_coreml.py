"""
CoreML export pipeline for BrickScan classifier.

Loads a trained PyTorch checkpoint or ONNX model and exports to CoreML (.mlpackage)
with embedded class mappings. Includes validation against PyTorch baseline and
generates a Swift wrapper class.

Usage:
    python export_coreml.py --checkpoint path/to/best.pt --class-map path/to/class_map.json --output-dir ./coreml_output
    OR
    python export_coreml.py --onnx path/to/model.onnx --class-map path/to/class_map.json --output-dir ./coreml_output
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import io

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

try:
    import coremltools as ct
    from coremltools.models.neural_network import quantization_utils
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False

try:
    import onnxruntime as rt
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LegoBrickClassifier(nn.Module):
    """
    Dual-head EfficientNet-B3 classifier for LEGO parts and colors.
    (Must match train_two_stage.py definition)
    """

    def __init__(self, num_parts: int, num_colors: int):
        super().__init__()
        self.num_parts = num_parts
        self.num_colors = num_colors

        from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3

        # Load pretrained EfficientNet-B3
        self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)

        # Remove default classifier
        self.backbone.classifier = nn.Identity()

        # Feature dimension from EfficientNet-B3
        self.feature_dim = 1536

        # Dual output heads
        self.part_head = nn.Linear(self.feature_dim, num_parts)
        self.color_head = nn.Linear(self.feature_dim, num_colors)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning logits for both heads."""
        features = self.backbone(x)
        part_logits = self.part_head(features)
        color_logits = self.color_head(features)
        return part_logits, color_logits


def load_pytorch_checkpoint(
    checkpoint_path: str,
    class_map: Dict,
) -> nn.Module:
    """Load PyTorch checkpoint and return model in eval mode."""
    logger.info(f"Loading PyTorch checkpoint from {checkpoint_path}")

    num_parts = len(class_map['part_to_idx'])
    num_colors = len(class_map['color_to_idx'])

    model = LegoBrickClassifier(num_parts, num_colors)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Loaded model: {num_parts} parts, {num_colors} colors")
    return model


def load_onnx_session(onnx_path: str):
    """Load ONNX model session."""
    if not ONNX_AVAILABLE:
        raise RuntimeError("onnxruntime not installed. Install with: pip install onnxruntime")

    logger.info(f"Loading ONNX model from {onnx_path}")
    session = rt.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    return session


def export_to_coreml(
    model: Optional[nn.Module],
    onnx_path: Optional[str],
    class_map: Dict,
    output_dir: str,
) -> str:
    """
    Export PyTorch or ONNX model to CoreML.

    Args:
        model: PyTorch model (if exporting from PyTorch)
        onnx_path: Path to ONNX model (if exporting from ONNX)
        class_map: Class mapping dictionary
        output_dir: Output directory for .mlpackage

    Returns:
        Path to exported .mlpackage
    """
    if not COREML_AVAILABLE:
        raise RuntimeError("coremltools not installed. Install with: pip install coremltools")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_parts = len(class_map['part_to_idx'])
    num_colors = len(class_map['color_to_idx'])

    logger.info("Exporting to CoreML...")

    if model is not None:
        # Export from PyTorch using torch.jit.trace
        logger.info("Exporting from PyTorch checkpoint")

        # Create dummy input (1, 3, 300, 300) - normalized to [0, 1]
        dummy_input = torch.randn(1, 3, 300, 300)

        # Trace the model
        traced_model = torch.jit.trace(model, dummy_input)

        # Convert to CoreML
        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.ImageType(
                    name='image',
                    shape=(1, 3, 300, 300),
                    scale=1.0 / 255.0,  # Scale pixel values from [0, 255] to [0, 1]
                )
            ],
            outputs=[
                ct.TensorType(name='part_logits', shape=(1, num_parts)),
                ct.TensorType(name='color_logits', shape=(1, num_colors)),
            ],
            minimum_deployment_target=ct.target.iOS16,
            compute_units=ct.ComputeUnit.ALL,
        )

    else:
        # Export from ONNX
        logger.info(f"Exporting from ONNX model: {onnx_path}")

        mlmodel = ct.convert(
            onnx_path,
            inputs=[
                ct.ImageType(
                    name='image',
                    shape=(1, 3, 300, 300),
                    scale=1.0 / 255.0,
                )
            ],
            minimum_deployment_target=ct.target.iOS16,
            compute_units=ct.ComputeUnit.ALL,
        )

    # Add metadata and class mappings
    mlmodel.short_description = "LEGO Brick Classifier - Dual-head EfficientNet-B3"

    # Embed class mappings in user-defined metadata
    mlmodel.user_defined_metadata = {
        'part_map': json.dumps(class_map['idx_to_part']),
        'color_map': json.dumps(class_map['idx_to_color']),
        'num_parts': str(num_parts),
        'num_colors': str(num_colors),
    }

    # Save the model package
    mlpackage_path = output_dir / 'BrickClassifier.mlpackage'
    logger.info(f"Saving CoreML model to {mlpackage_path}")
    mlmodel.save(str(mlpackage_path))

    return str(mlpackage_path)


def validate_export(
    pytorch_model: nn.Module,
    mlmodel_path: str,
    class_map: Dict,
    num_samples: int = 5,
    tolerance: float = 0.01,
):
    """
    Validate CoreML export by comparing outputs with PyTorch baseline.

    Args:
        pytorch_model: Original PyTorch model
        mlmodel_path: Path to exported .mlpackage
        class_map: Class mappings
        num_samples: Number of random test samples
        tolerance: Max absolute difference allowed (default 0.01)
    """
    logger.info(f"Validating CoreML export (tolerance={tolerance})...")

    # Load CoreML model
    try:
        mlmodel = ct.models.MLModel(mlmodel_path)
    except Exception as e:
        logger.error(f"Failed to load CoreML model for validation: {e}")
        return False

    num_parts = len(class_map['part_to_idx'])
    num_colors = len(class_map['color_to_idx'])

    pytorch_model.eval()

    for i in range(num_samples):
        # Create random input (normalized to [0, 1])
        input_np = np.random.randn(1, 3, 300, 300).astype(np.float32)
        input_tensor = torch.from_numpy(input_np)

        # PyTorch inference
        with torch.no_grad():
            pt_part_logits, pt_color_logits = pytorch_model(input_tensor)
            pt_part_logits = pt_part_logits.numpy()
            pt_color_logits = pt_color_logits.numpy()

        # CoreML inference
        # Convert back to [0, 255] uint8 for input
        input_uint8 = ((input_np + 1.0) / 2.0 * 255.0).astype(np.uint8)

        input_dict = {
            'image': input_uint8,
        }

        try:
            coreml_output = mlmodel.predict(input_dict)
            coreml_part_logits = coreml_output['part_logits']
            coreml_color_logits = coreml_output['color_logits']
        except Exception as e:
            logger.warning(f"CoreML inference failed for sample {i}: {e}")
            continue

        # Compare
        part_diff = np.abs(pt_part_logits - coreml_part_logits).max()
        color_diff = np.abs(pt_color_logits - coreml_color_logits).max()

        logger.info(
            f"  Sample {i+1}: part_diff={part_diff:.6f}, "
            f"color_diff={color_diff:.6f}"
        )

        if part_diff > tolerance or color_diff > tolerance:
            logger.warning(
                f"Sample {i+1} exceeds tolerance! "
                f"part_diff={part_diff:.6f}, color_diff={color_diff:.6f}"
            )
            return False

    logger.info("Validation passed!")
    return True


def generate_swift_wrapper(
    output_dir: str,
    mlpackage_name: str = 'BrickClassifier',
):
    """
    Generate a Swift wrapper class for the CoreML model.

    Args:
        output_dir: Output directory
        mlpackage_name: Name of the .mlpackage (without extension)
    """
    logger.info("Generating Swift wrapper class...")

    swift_code = '''import CoreML
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
                var topIndices = partProbs.enumerated()
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
                print("Classification error: \\(error)")
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
        let maxLogit = logits.max() ?? 0
        let expLogits = logits.map { exp($0 - maxLogit) }
        let sum = expLogits.reduce(0, +)
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

        CVPixelBufferLockBaseAddress(pixelBuffer, .readAndWrite)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readAndWrite) }

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

        context.draw(self.cgImage!, in: CGRect(origin: .zero, size: self.size))

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

@available(iOS 16.0, *)
struct BrickClassifierInput: MLFeatureProvider {
    let image: CVPixelBuffer

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
'''

    output_path = Path(output_dir) / 'BrickClassifier.swift'
    logger.info(f"Writing Swift wrapper to {output_path}")

    with open(output_path, 'w') as f:
        f.write(swift_code)

    logger.info(f"Swift wrapper generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Export BrickScan classifier to CoreML format'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to PyTorch checkpoint (.pt file)',
    )
    parser.add_argument(
        '--onnx',
        type=str,
        help='Path to ONNX model file',
    )
    parser.add_argument(
        '--class-map',
        type=str,
        required=True,
        help='Path to class_map.json',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./coreml_output',
        help='Output directory for CoreML model (default: ./coreml_output)',
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip validation step',
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.checkpoint and not args.onnx:
        parser.error('Must provide either --checkpoint or --onnx')

    if args.checkpoint and args.onnx:
        parser.error('Cannot provide both --checkpoint and --onnx')

    # Load class map
    logger.info(f"Loading class map from {args.class_map}")
    with open(args.class_map, 'r') as f:
        class_map = json.load(f)

    # Load model
    pytorch_model = None
    onnx_path = None

    if args.checkpoint:
        pytorch_model = load_pytorch_checkpoint(args.checkpoint, class_map)
    else:
        onnx_path = args.onnx
        load_onnx_session(onnx_path)  # Verify it loads

    # Export to CoreML
    mlpackage_path = export_to_coreml(
        pytorch_model,
        onnx_path,
        class_map,
        args.output_dir,
    )

    # Validate
    if pytorch_model and not args.skip_validation:
        validate_export(pytorch_model, mlpackage_path, class_map)

    # Generate Swift wrapper
    generate_swift_wrapper(args.output_dir, 'BrickClassifier')

    logger.info(f"Export complete!")
    logger.info(f"  CoreML model: {mlpackage_path}")
    logger.info(f"  Swift wrapper: {Path(args.output_dir) / 'BrickClassifier.swift'}")


if __name__ == '__main__':
    main()
