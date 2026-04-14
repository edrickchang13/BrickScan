#!/usr/bin/env python3
"""
Convert DINOv2 ONNX model to CoreML (.mlpackage) for on-device iPhone inference.

Deployment target: iOS 17  (Neural Engine + ANE-optimized ViT ops)
Compute units:     ALL  (CPU + GPU + Neural Engine)
Precision:         FP16 weights, FP32 accumulation (automatic via CoreML)

Run this on a Mac (Apple Silicon recommended for fast conversion):

    python export_coreml_dinov2.py \
        --onnx    /path/to/dinov2_lego.onnx \
        --labels  /path/to/part_labels.json \
        --output  BrickScanDINOv2.mlpackage \
        [--quantize int8]   # optional INT8 quantization for smaller bundle

Output files:
    BrickScanDINOv2.mlpackage/           ← drag into Xcode project
    BrickScanDINOv2_metadata.json        ← label map for Swift lookup
    coreml_export_info.json              ← conversion metadata

Integration in Swift:
    let model = try BrickScanDINOv2(configuration: .init())
    let input = BrickScanDINOv2Input(image: pixelBuffer)
    let output = try model.prediction(input: input)
    // output.logits is a [1, num_classes] MLMultiArray — take argmax
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

INPUT_SIZE = 518   # DINOv2 ViT-B/14 native resolution

# ImageNet stats
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


# ──────────────────────────────────────────────────────────────────────────────
# Conversion
# ──────────────────────────────────────────────────────────────────────────────

def convert_to_coreml(onnx_path: Path, output_path: Path, num_classes: int) -> any:
    """
    Convert ONNX → CoreML mlpackage with built-in image preprocessing.

    The CoreML model accepts a CVPixelBuffer (camera frame) directly — no
    manual normalisation needed on the Swift side.
    """
    try:
        import coremltools as ct
    except ImportError:
        raise SystemExit(
            "coremltools not installed.\n"
            "Install: pip install coremltools>=7.0\n"
            "Requires macOS with Xcode installed."
        )

    log.info("coremltools version: %s", ct.__version__)

    # ── Step 1: Load ONNX ────────────────────────────────────────────────────
    log.info("Loading ONNX model from %s …", onnx_path)
    t0 = time.perf_counter()

    # CoreML conversion via the ONNX → MIL route
    mlmodel = ct.convert(
        str(onnx_path),
        convert_to="mlprogram",                  # .mlpackage (not legacy .mlmodel)
        minimum_deployment_target=ct.target.iOS17,
        compute_units=ct.ComputeUnit.ALL,        # CPU + GPU + Neural Engine
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, INPUT_SIZE, INPUT_SIZE),
                # ImageNet normalisation baked in — Swift sends raw CVPixelBuffer
                bias=[-m / s for m, s in zip(_MEAN, _STD)],
                scale=1.0 / (255.0 * np.mean(_STD)),   # approximate joint scale
                color_layout=ct.colorlayout.RGB,
            )
        ],
        outputs=[
            ct.TensorType(name="logits"),
        ],
    )

    elapsed = time.perf_counter() - t0
    log.info("Conversion completed in %.1fs", elapsed)

    return mlmodel


def add_metadata(mlmodel: any, labels: dict, num_classes: int) -> any:
    """Attach class labels and description to the CoreML model."""
    try:
        import coremltools as ct
    except ImportError:
        return mlmodel

    spec = mlmodel.get_spec()

    # Model description
    spec.description.metadata.shortDescription = (
        "BrickScan DINOv2 ViT-B/14 — LEGO part classifier. "
        f"Identifies {num_classes} part types from a 518×518 RGB image."
    )
    spec.description.metadata.author = "BrickScan"
    spec.description.metadata.license = "Proprietary"
    spec.description.metadata.versionString = "2.0-dinov2"

    mlmodel = ct.models.MLModel(spec, weights_dir=mlmodel.weights_dir)
    return mlmodel


def quantize_int8(mlmodel: any, output_path: Path) -> any:
    """Apply INT8 weight quantization (~4× model size reduction)."""
    try:
        import coremltools as ct
        from coremltools.optimize.coreml import (
            OpLinearQuantizerConfig,
            OptimizationConfig,
            linearly_quantize_weights,
        )
    except ImportError:
        log.warning("CoreML optimization tools not available — skipping INT8 quantization")
        return mlmodel

    log.info("Applying INT8 weight quantization …")
    t0 = time.perf_counter()

    op_config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
    config = OptimizationConfig(global_config=op_config)
    mlmodel_int8 = linearly_quantize_weights(mlmodel, config)

    elapsed = time.perf_counter() - t0
    log.info("INT8 quantization completed in %.1fs", elapsed)
    return mlmodel_int8


def run_coreml_benchmark(mlmodel: any, num_classes: int, n: int = 20) -> dict:
    """Run a quick latency benchmark using random inputs (Mac only)."""
    try:
        import coremltools as ct
        from PIL import Image as PILImage
    except ImportError:
        return {}

    log.info("Running CoreML benchmark (%d inferences) …", n)
    dummy_pil = PILImage.fromarray(
        np.random.randint(0, 255, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    )

    times = []
    for i in range(n):
        t = time.perf_counter()
        try:
            mlmodel.predict({"image": dummy_pil})
        except Exception as e:
            if i == 0:
                log.warning("CoreML predict failed (normal on non-Mac): %s", e)
            break
        times.append((time.perf_counter() - t) * 1000)

    if not times:
        return {}

    return {
        "lat_p50_ms":  round(float(np.percentile(times, 50)), 2),
        "lat_p95_ms":  round(float(np.percentile(times, 95)), 2),
        "lat_mean_ms": round(float(np.mean(times)), 2),
        "n_runs":      len(times),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Swift integration helpers
# ──────────────────────────────────────────────────────────────────────────────

SWIFT_INTEGRATION_SNIPPET = '''
// ── BrickScan CoreML Integration (Swift) ──────────────────────────────────
//
// 1. Drag BrickScanDINOv2.mlpackage into your Xcode project.
// 2. Xcode auto-generates BrickScanDINOv2.swift with typed I/O.
// 3. Use the snippet below in your ScanViewController.

import CoreML
import Vision

class BrickScanPredictor {
    private let model: VNCoreMLModel

    // Load label map from BrickScanDINOv2_metadata.json bundled in app
    private let idx2part: [Int: String]

    init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all   // ANE + GPU + CPU

        let mlModel = try BrickScanDINOv2(configuration: config).model
        self.model = try VNCoreMLModel(for: mlModel)

        // Load label JSON bundled in app target
        let url = Bundle.main.url(forResource: "BrickScanDINOv2_metadata", withExtension: "json")!
        let data = try Data(contentsOf: url)
        let parsed = try JSONDecoder().decode([String: [String: String]].self, from: data)
        self.idx2part = parsed["idx2part"]!.reduce(into: [:]) { dict, pair in
            if let idx = Int(pair.key) { dict[idx] = pair.value }
        }
    }

    /// Returns top-3 (partNum, confidence) sorted by descending confidence.
    func predict(pixelBuffer: CVPixelBuffer) async throws -> [(partNum: String, confidence: Float)] {
        return try await withCheckedThrowingContinuation { continuation in
            let request = VNCoreMLRequest(model: model) { [weak self] request, error in
                guard let self else { return }
                if let error { continuation.resume(throwing: error); return }

                guard let results = request.results as? [VNCoreMLFeatureValueObservation],
                      let logitArray = results.first?.featureValue.multiArrayValue
                else {
                    continuation.resume(returning: []); return
                }

                // Softmax over logits
                let count = logitArray.count
                var logits = [Float](repeating: 0, count: count)
                for i in 0..<count { logits[i] = logitArray[i].floatValue }

                let maxLogit = logits.max() ?? 0
                var expValues = logits.map { exp($0 - maxLogit) }
                let sumExp = expValues.reduce(0, +)
                let probs = expValues.map { $0 / sumExp }

                // Top-3
                let topIndices = probs.indices
                    .sorted { probs[$0] > probs[$1] }
                    .prefix(3)

                let predictions = topIndices.compactMap { idx -> (String, Float)? in
                    guard let partNum = self.idx2part[idx] else { return nil }
                    return (partNum, probs[idx])
                }

                continuation.resume(returning: predictions)
            }
            request.imageCropAndScaleOption = .centerCrop

            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
            try? handler.perform([request])
        }
    }
}
// ─────────────────────────────────────────────────────────────────────────────
'''


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx",     required=True, help="Path to dinov2_lego.onnx")
    parser.add_argument("--labels",   required=True, help="Path to part_labels.json")
    parser.add_argument("--output",   default="BrickScanDINOv2.mlpackage",
                        help="Output .mlpackage path")
    parser.add_argument("--quantize", choices=["int8", "none"], default="none",
                        help="Weight quantization (int8 cuts size ~4×)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run latency benchmark after conversion")
    args = parser.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        sys.exit(f"ONNX model not found: {onnx_path}")

    labels_raw = json.loads(Path(args.labels).read_text())
    idx2part = labels_raw.get("idx2part", labels_raw)
    num_classes = labels_raw.get("num_classes", len(idx2part))
    log.info("Labels: %d part classes", num_classes)

    output_path = Path(args.output)

    # ── Convert ──────────────────────────────────────────────────────────────
    mlmodel = convert_to_coreml(onnx_path, output_path, num_classes)
    mlmodel = add_metadata(mlmodel, labels_raw, num_classes)

    if args.quantize == "int8":
        mlmodel = quantize_int8(mlmodel, output_path)

    # ── Save .mlpackage ───────────────────────────────────────────────────────
    log.info("Saving CoreML model → %s", output_path)
    mlmodel.save(str(output_path))

    size_mb = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / 1e6
    log.info("Package size: %.1f MB", size_mb)

    # ── Save metadata JSON for Swift label lookup ─────────────────────────────
    meta_json_path = output_path.parent / (output_path.stem + "_metadata.json")
    meta_json_path.write_text(json.dumps({"idx2part": idx2part}, indent=2))
    log.info("Swift label map → %s", meta_json_path.name)

    # ── Benchmark ─────────────────────────────────────────────────────────────
    bench = {}
    if args.benchmark:
        bench = run_coreml_benchmark(mlmodel, num_classes)
        if bench:
            log.info(
                "CoreML latency  p50=%.1fms  p95=%.1fms  (n=%d)",
                bench["lat_p50_ms"], bench["lat_p95_ms"], bench["n_runs"],
            )

    # ── Export info ────────────────────────────────────────────────────────────
    info = {
        "model": "DINOv2",
        "backbone": "vit_base_patch14_dinov2",
        "format": "mlpackage",
        "minimum_deployment_target": "iOS17",
        "compute_units": "ALL",
        "input": {
            "name": "image",
            "shape": [1, 3, INPUT_SIZE, INPUT_SIZE],
            "preprocessing": "ImageNet (built-in bias/scale)",
            "color_layout": "RGB",
        },
        "output": {"name": "logits", "shape": [1, num_classes]},
        "quantization": args.quantize,
        "size_mb": round(size_mb, 1),
        "num_classes": num_classes,
        "source_onnx": str(onnx_path),
        "benchmark": bench,
    }
    info_path = output_path.parent / "coreml_export_info.json"
    info_path.write_text(json.dumps(info, indent=2))

    # ── Swift snippet ──────────────────────────────────────────────────────────
    snippet_path = output_path.parent / "BrickScanPredictor.swift"
    snippet_path.write_text(SWIFT_INTEGRATION_SNIPPET)

    log.info(
        "\n"
        "══════════════════════════════════════════════════════════\n"
        "  CoreML export complete!\n"
        "\n"
        "  %-40s %.1f MB\n"
        "  %-40s (bundle in app target)\n"
        "  %-40s (reference Swift integration)\n"
        "\n"
        "  Xcode steps:\n"
        "    1. Drag %s into your Xcode project\n"
        "    2. Add %s to app bundle resources\n"
        "    3. Copy BrickScanPredictor.swift into your source tree\n"
        "    4. Run on device — target < 100ms on iPhone 14 (A15 ANE)\n"
        "══════════════════════════════════════════════════════════",
        str(output_path.name),             size_mb,
        str(meta_json_path.name),
        str(snippet_path.name),
        str(output_path.name),
        str(meta_json_path.name),
    )


if __name__ == "__main__":
    main()
