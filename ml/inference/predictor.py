"""Inference predictor class for BrickScan model."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import onnxruntime as ort
from PIL import Image


class LegoPredictor:
    """LEGO piece predictor using ONNX model.

    Supports inference from image bytes with configurable top-k predictions.
    """

    def __init__(
        self,
        model_path: str,
        class_map_path: str,
        top_k: int = 5,
        providers: Optional[List[str]] = None,
    ):
        """Initialize predictor.

        Args:
            model_path: Path to ONNX model
            class_map_path: Path to class_to_idx JSON mapping
            top_k: Number of top predictions to return
            providers: ONNX Runtime execution providers
                (defaults to ["CPUExecutionProvider"])
        """
        self.model_path = model_path
        self.class_map_path = class_map_path
        self.top_k = top_k

        # Load ONNX model
        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            str(model_path),
            providers=providers,
        )

        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        input_shape = self.session.get_inputs()[0].shape
        self.input_size = input_shape[2] if len(input_shape) > 2 else 224

        # Load class mapping
        with open(class_map_path) as f:
            self.class_to_idx = json.load(f)

        # Create inverse mapping
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        print(f"Model loaded from {model_path}")
        print(f"Input size: {self.input_size}x{self.input_size}")
        print(f"Number of classes: {len(self.idx_to_class)}")

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess image bytes to model input.

        Args:
            image_bytes: Image data as bytes

        Returns:
            Normalized image tensor (1, 3, H, W) as float32 numpy array
        """
        # Load image
        img = Image.open(type("obj", (object,), {"read": lambda: image_bytes})())
        if isinstance(image_bytes, bytes):
            from io import BytesIO
            img = Image.open(BytesIO(image_bytes))

        # Convert to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize
        img = img.resize((self.input_size, self.input_size), Image.Resampling.LANCZOS)

        # Convert to numpy
        img_array = np.array(img).astype(np.float32) / 255.0

        # Normalize with ImageNet mean/std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        img_array = (img_array - mean) / std

        # Convert to CHW format
        img_array = np.transpose(img_array, (2, 0, 1))

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def predict(self, image_bytes: bytes) -> List[Dict]:
        """Predict top-k LEGO pieces.

        Args:
            image_bytes: Image data as bytes

        Returns:
            List of dicts with keys:
                - 'part_num': LEGO part number (string)
                - 'confidence': Confidence score (0-1)
        """
        # Preprocess
        img_array = self.preprocess(image_bytes)

        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: img_array})
        logits = outputs[0][0]  # (num_classes,)

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        probs = exp_logits / np.sum(exp_logits)

        # Get top-k
        top_indices = np.argsort(probs)[-self.top_k:][::-1]
        top_probs = probs[top_indices]

        # Convert to results
        results = []
        for idx, prob in zip(top_indices, top_probs):
            part_num = self.idx_to_class[int(idx)]
            results.append({
                "part_num": part_num,
                "confidence": float(prob),
            })

        return results

    def predict_with_details(self, image_bytes: bytes) -> List[Dict]:
        """Predict with additional details.

        Args:
            image_bytes: Image data as bytes

        Returns:
            List of dicts with keys:
                - 'part_num': LEGO part number
                - 'confidence': Confidence score
                - 'rank': Rank (1-indexed)
                - 'logit': Raw logit value
        """
        # Preprocess
        img_array = self.preprocess(image_bytes)

        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: img_array})
        logits = outputs[0][0]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        # Get top-k
        top_indices = np.argsort(probs)[-self.top_k:][::-1]
        top_probs = probs[top_indices]
        top_logits = logits[top_indices]

        # Convert to results
        results = []
        for rank, (idx, prob, logit) in enumerate(zip(top_indices, top_probs, top_logits), 1):
            part_num = self.idx_to_class[int(idx)]
            results.append({
                "part_num": part_num,
                "confidence": float(prob),
                "rank": rank,
                "logit": float(logit),
            })

        return results

    def batch_predict(self, image_list: List[bytes]) -> List[List[Dict]]:
        """Batch predict on multiple images.

        Args:
            image_list: List of image bytes

        Returns:
            List of prediction lists (one per image)
        """
        # Preprocess all images
        imgs = np.vstack([self.preprocess(img) for img in image_list])

        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: imgs})
        all_logits = outputs[0]

        # Process each image
        results_list = []
        for logits in all_logits:
            # Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)

            # Get top-k
            top_indices = np.argsort(probs)[-self.top_k:][::-1]
            top_probs = probs[top_indices]

            # Convert to results
            results = []
            for idx, prob in zip(top_indices, top_probs):
                part_num = self.idx_to_class[int(idx)]
                results.append({
                    "part_num": part_num,
                    "confidence": float(prob),
                })

            results_list.append(results)

        return results_list


# Example usage
if __name__ == "__main__":
    # This is an example of how to use the LegoPredictor class
    # In practice, you would load a real image file

    # Initialize predictor
    # predictor = LegoPredictor(
    #     model_path="./models/brickscan.onnx",
    #     class_map_path="./models/class_mapping.json",
    #     top_k=5,
    # )

    # Read image file
    # with open("path/to/image.jpg", "rb") as f:
    #     image_bytes = f.read()

    # Make prediction
    # predictions = predictor.predict(image_bytes)
    # for pred in predictions:
    #     print(f"{pred['part_num']}: {pred['confidence']:.2%}")

    # Or with details
    # predictions_detailed = predictor.predict_with_details(image_bytes)
    # for pred in predictions_detailed:
    #     print(f"Rank {pred['rank']}: {pred['part_num']} "
    #           f"({pred['confidence']:.2%}, logit: {pred['logit']:.2f})")

    print("LegoPredictor class ready for use")
