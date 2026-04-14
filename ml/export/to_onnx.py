"""Export BrickScan PyTorch model to ONNX format."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from training.model import load_from_checkpoint


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    class_to_idx: Dict[str, int],
    config: dict,
) -> None:
    """Export PyTorch model to ONNX format.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_path: Path to save ONNX model
        class_to_idx: Dictionary mapping part numbers to class indices
        config: Model configuration
    """
    device = torch.device("cpu")

    # Load model
    print(f"Loading model from {checkpoint_path}")
    model = load_from_checkpoint(
        checkpoint_path,
        num_classes=len(class_to_idx),
        architecture=config["model"]["architecture"],
        dropout=config["model"]["dropout"],
    )
    model = model.to(device)
    model.eval()

    # Create output directory
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Create example input
    input_shape = (1, 3, config["model"]["input_size"], config["model"]["input_size"])
    example_input = torch.randn(*input_shape)

    print(f"Model input shape: {input_shape}")
    print(f"Number of classes: {len(class_to_idx)}")

    # Export to ONNX
    print(f"Exporting model to ONNX...")
    torch.onnx.export(
        model,
        example_input,
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
        verbose=False,
    )

    print(f"Model exported to {output_path}")

    # Verify with ONNX Runtime
    print("\nVerifying model with ONNX Runtime...")
    try:
        session = ort.InferenceSession(
            str(output_path),
            providers=["CPUExecutionProvider"],
        )

        # Get model info
        input_names = [input.name for input in session.get_inputs()]
        output_names = [output.name for output in session.get_outputs()]

        print(f"Input names: {input_names}")
        print(f"Output names: {output_names}")

        # Run inference with test input
        test_input = np.random.randn(1, 3, config["model"]["input_size"], config["model"]["input_size"]).astype(
            np.float32
        )
        outputs = session.run(output_names, {input_names[0]: test_input})
        logits = outputs[0]

        print(f"\nTest inference:")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {logits.shape}")
        print(f"  Output value range: [{logits.min():.4f}, {logits.max():.4f}]")

        # Check softmax output
        softmax_probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        top5_indices = np.argsort(softmax_probs[0])[-5:][::-1]
        print(f"  Top-5 class indices: {top5_indices}")

    except Exception as e:
        print(f"Error verifying model: {e}")
        raise

    # Get model file size
    model_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nModel file size: {model_size_mb:.2f} MB")

    # Estimate inference time (rough estimate for single batch)
    print(f"\nEstimated inference time (single batch on CPU): ~50-200ms")
    print(f"Estimated inference time (batch size 16 on CPU): ~200-800ms")

    # Save class mapping JSON alongside model
    mapping_path = output_path_obj.parent / "class_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"\nClass mapping saved to: {mapping_path}")

    # Save model info
    info = {
        "model_name": "BrickScan",
        "model_type": "ONNX",
        "input_shape": list(input_shape),
        "num_classes": len(class_to_idx),
        "architecture": config["model"]["architecture"],
        "file_size_mb": model_size_mb,
        "opset_version": 14,
        "input_names": input_names,
        "output_names": output_names,
    }
    info_path = output_path_obj.parent / "model_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Model info saved to: {info_path}")


def main(
    checkpoint_path: str = "./checkpoints/best_model.pt",
    output_path: str = "./models/brickscan.onnx",
) -> None:
    """Main export function.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_path: Path to save ONNX model
    """
    # Load checkpoint to get config and class mapping
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    class_to_idx = checkpoint["class_to_idx"]

    export_to_onnx(checkpoint_path, output_path, class_to_idx, config)

    print("\n" + "=" * 50)
    print("EXPORT COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export BrickScan model to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/best_model.pt",
        help="Path to PyTorch checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/brickscan.onnx",
        help="Path to save ONNX model",
    )
    args = parser.parse_args()

    main(args.checkpoint, args.output)
