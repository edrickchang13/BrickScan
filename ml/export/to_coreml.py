"""Export BrickScan PyTorch model to Core ML format for iOS."""

import argparse
import json
from pathlib import Path
from typing import Dict

import coremltools as ct
import torch
import torch.nn as nn

from training.model import load_from_checkpoint


def export_to_coreml(
    checkpoint_path: str,
    output_path: str,
    class_to_idx: Dict[str, int],
    config: dict,
) -> None:
    """Export PyTorch model to Core ML format.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_path: Path to save Core ML model (.mlpackage directory)
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

    # Create example input
    input_shape = (1, 3, config["model"]["input_size"], config["model"]["input_size"])
    example_input = torch.randn(*input_shape)

    print(f"Model input shape: {input_shape}")
    print(f"Number of classes: {len(class_to_idx)}")

    # Trace the model
    print("Tracing model with torch.jit.trace...")
    traced_model = torch.jit.trace(model, example_input)

    # Create class labels list (sorted by index)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_labels = [idx_to_class[i] for i in range(len(class_to_idx))]

    # Convert to Core ML
    print("Converting to Core ML...")
    ml_model = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                name="image",
                shape=input_shape,
                scale=1.0 / 255.0,
                bias=[0, 0, 0],
            )
        ],
        outputs=[
            ct.TensorType(name="logits", shape=(1, len(class_to_idx)))
        ],
        classifier_config=ct.ClassifierConfig(class_labels),
        minimum_deployment_target=ct.target.iOS16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )

    # Set model metadata
    ml_model.author = "BrickScan"
    ml_model.short_description = "LEGO piece classifier"
    ml_model.input_description["image"] = "Input image of LEGO piece"
    ml_model.output_description["classLabel"] = "Predicted LEGO part number"
    ml_model.output_description["logits"] = "Raw logits for each class"
    ml_model.version = "1.0.0"

    # Save model
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving Core ML model to {output_path}")
    ml_model.save(output_path)

    # Print model info
    print("\n" + "=" * 50)
    print("CORE ML MODEL INFO")
    print("=" * 50)
    print(f"Model saved to: {output_path}")
    print(f"\nModel Inputs:")
    for inp in ml_model.input_description:
        print(f"  {inp}: {ml_model.input_description[inp]}")

    print(f"\nModel Outputs:")
    for out in ml_model.output_description:
        print(f"  {out}: {ml_model.output_description[out]}")

    print(f"\nDeployment Target: iOS 16+")
    print(f"Compute Units: CPU and Neural Engine")
    print(f"Number of Classes: {len(class_to_idx)}")

    # Save class mapping JSON alongside model
    mapping_path = output_path_obj.parent / "class_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"\nClass mapping saved to: {mapping_path}")


def main(
    checkpoint_path: str = "./checkpoints/best_model.pt",
    output_path: str = "./models/BrickScan.mlpackage",
) -> None:
    """Main export function.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_path: Path to save Core ML model
    """
    # Load checkpoint to get config and class mapping
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    class_to_idx = checkpoint["class_to_idx"]

    export_to_coreml(checkpoint_path, output_path, class_to_idx, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export BrickScan model to Core ML")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/best_model.pt",
        help="Path to PyTorch checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/BrickScan.mlpackage",
        help="Path to save Core ML model",
    )
    args = parser.parse_args()

    main(args.checkpoint, args.output)
