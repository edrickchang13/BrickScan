"""Export trained model to ONNX format."""

import argparse
import json
from pathlib import Path

import onnx
import onnxruntime as ort
import torch

from model import LegoBrickClassifier


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    labels_dir: str,
    device: torch.device = None
):
    """
    Export a trained checkpoint to ONNX format.

    Args:
        checkpoint_path: Path to the checkpoint file
        output_path: Path where ONNX model will be saved
        labels_dir: Directory containing part_labels.json and color_labels.json
        device: torch.device to use (defaults to cuda if available)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_dir = Path(labels_dir)

    # Load label encoders
    with open(labels_dir / 'part_labels.json', 'r') as f:
        part_labels = json.load(f)

    with open(labels_dir / 'color_labels.json', 'r') as f:
        color_labels = json.load(f)

    num_parts = len(part_labels)
    num_colors = len(color_labels)

    print(f"Number of parts: {num_parts}")
    print(f"Number of colors: {num_colors}")

    # Create model
    model = LegoBrickClassifier(num_parts=num_parts, num_colors=num_colors)
    model = model.to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    print(f"Loaded checkpoint from {checkpoint_path}")

    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    # Export to ONNX using the dedicated forward_onnx method
    # (concatenates part_logits + color_logits into a single output tensor)
    print(f"Exporting to ONNX: {output_path}")

    class _OnnxWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m.forward_onnx(x)

    wrapped = _OnnxWrapper(model).to(device)
    wrapped.eval()

    torch.onnx.export(
        wrapped,
        dummy_input,
        str(output_path),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=18,
        do_constant_folding=True,
        verbose=False
    )

    # Verify ONNX model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid")

    # Test inference with onnxruntime
    print("Testing inference with onnxruntime...")
    sess = ort.InferenceSession(str(output_path), providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

    # Dummy input for onnxruntime (numpy)
    import numpy as np
    dummy_input_np = np.random.randn(1, 3, 224, 224).astype(np.float32)

    outputs = sess.run(None, {'input': dummy_input_np})
    output_shape = outputs[0].shape

    print(f"Output shape: {output_shape}")
    print(f"Expected shape: (1, {num_parts + num_colors})")

    if output_shape == (1, num_parts + num_colors):
        print("Output shape is correct!")
    else:
        print(f"WARNING: Output shape mismatch!")

    print(f"\n=== ONNX Export Summary ===")
    print(f"Model path: {output_path}")
    print(f"Output shape: {output_shape}")
    print(f"Number of part classes: {num_parts}")
    print(f"Number of color classes: {num_colors}")
    print(f"Total output dimensions: {output_shape[1]}")


def main():
    parser = argparse.ArgumentParser(description='Export model to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--output', type=str, required=True, help='Path to save ONNX model')
    parser.add_argument('--labels-dir', type=str, required=True, help='Directory containing label JSON files')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        labels_dir=args.labels_dir,
        device=device
    )


if __name__ == '__main__':
    main()
