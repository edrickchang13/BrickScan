"""
Production-quality inference module for BrickScan stage 2 classifier.

Loads ONNX model and class mappings, provides high-level classify() function
with test-time augmentation (TTA) support. Importable by FastAPI backend.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import onnxruntime as rt
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    rt = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BrickClassifier:
    """
    High-level interface for LEGO brick classification.

    Wraps ONNX model loading and inference with TTA and class mapping.
    """

    def __init__(
        self,
        model_path: str,
        class_map_path: str,
        use_onnx: bool = True,
        providers: Optional[List[str]] = None,
    ):
        """
        Initialize classifier.

        Args:
            model_path: Path to ONNX model file (or .pt for PyTorch)
            class_map_path: Path to class_map.json
            use_onnx: Whether to use ONNX (True) or PyTorch (False)
            providers: ONNX Runtime execution providers (default: auto-select)
        """
        self.model_path = Path(model_path)
        self.class_map_path = Path(class_map_path)
        self.use_onnx = use_onnx and ONNX_AVAILABLE
        self.device = 'cpu'

        # Load class mappings
        logger.info(f"Loading class map from {class_map_path}")
        with open(class_map_path, 'r') as f:
            class_map = json.load(f)

        self.part_to_idx = class_map['part_to_idx']
        self.color_to_idx = class_map['color_to_idx']
        self.idx_to_part = {int(k): v for k, v in class_map['idx_to_part'].items()}
        self.idx_to_color = {int(k): v for k, v in class_map['idx_to_color'].items()}

        self.num_parts = len(self.part_to_idx)
        self.num_colors = len(self.color_to_idx)

        logger.info(f"Classes: {self.num_parts} parts, {self.num_colors} colors")

        # Load model
        if self.use_onnx:
            self._load_onnx(providers)
        else:
            self._load_pytorch()

    def _load_onnx(self, providers: Optional[List[str]] = None):
        """Load ONNX model."""
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime not installed. Install with: pip install onnxruntime")

        logger.info(f"Loading ONNX model from {self.model_path}")

        if providers is None:
            # Auto-select providers
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        try:
            self.session = rt.InferenceSession(
                str(self.model_path),
                providers=providers
            )
            actual_providers = self.session.get_providers()
            logger.info(f"ONNX Runtime using providers: {actual_providers}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise

    def _load_pytorch(self):
        """Load PyTorch model (for testing)."""
        import torch
        from train_two_stage import LegoBrickClassifier

        logger.info(f"Loading PyTorch model from {self.model_path}")

        self.model = LegoBrickClassifier(self.num_parts, self.num_colors)
        checkpoint = torch.load(self.model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def _preprocess_image(self, image: Union[str, Image.Image]) -> np.ndarray:
        """
        Load and preprocess image to 300x300 numpy array.

        Args:
            image: Path or PIL Image

        Returns:
            Preprocessed image as numpy array [3, 300, 300]
        """
        if isinstance(image, str):
            image = Image.open(image)
        elif not isinstance(image, Image.Image):
            raise TypeError("image must be str (path) or PIL.Image")

        image = image.convert('RGB')
        image_np = np.array(image)

        # Resize to 300x300
        image_pil = Image.fromarray(image_np)
        image_np = np.array(image_pil.resize((300, 300), Image.BILINEAR))

        return image_np

    def _apply_tta_transforms(
        self,
        image_np: np.ndarray
    ) -> List[np.ndarray]:
        """
        Apply 7 test-time augmentation transforms.

        Args:
            image_np: Input image as numpy array [H, W, 3]

        Returns:
            List of 7 augmented images [H, W, 3]
        """
        # Define 7 TTA variants
        tta_transforms = [
            # Original
            A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            # Horizontal flip
            A.Compose([
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            # Vertical flip
            A.Compose([
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            # Rotate +15 degrees
            A.Compose([
                A.Rotate(limit=15, p=1.0, border_mode=0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            # Rotate -15 degrees
            A.Compose([
                A.Rotate(limit=-15, p=1.0, border_mode=0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            # Brightness +10%
            A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            # Brightness -10%
            A.Compose([
                A.RandomBrightnessContrast(brightness_limit=-0.1, contrast_limit=0, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        ]

        augmented_images = []
        for transform in tta_transforms:
            aug = transform(image=image_np)
            augmented_images.append(aug['image'])

        return augmented_images

    def _onnx_inference(self, images_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run ONNX inference on batch of images.

        Args:
            images_np: Batch of images [B, 3, 300, 300]

        Returns:
            (part_logits, color_logits) each [B, num_classes]
        """
        input_name = self.session.get_inputs()[0].name

        # Run inference
        outputs = self.session.run(None, {input_name: images_np.astype(np.float32)})

        # Unpack outputs
        part_logits = outputs[0]  # [B, num_parts]
        color_logits = outputs[1]  # [B, num_colors]

        return part_logits, color_logits

    def _pytorch_inference(self, images_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run PyTorch inference on batch.

        Args:
            images_tensor: Batch of images [B, 3, 300, 300]

        Returns:
            (part_logits, color_logits) each [B, num_classes]
        """
        with torch.no_grad():
            part_logits, color_logits = self.model(images_tensor)

        return part_logits, color_logits

    def classify(
        self,
        image: Union[str, Image.Image],
        top_k: int = 5,
        use_tta: bool = True,
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Classify a LEGO brick image.

        Returns top-k predictions for parts with their confidence scores.

        Args:
            image: Path or PIL Image of brick
            top_k: Number of top predictions to return
            use_tta: Whether to use test-time augmentation

        Returns:
            List of dicts [{part_num, color_id, confidence}, ...]
            Sorted by confidence descending.
        """
        # Preprocess image
        image_np = self._preprocess_image(image)

        if use_tta:
            # Apply TTA
            tta_images = self._apply_tta_transforms(image_np)

            # Convert to tensor batch
            if self.use_onnx:
                # ONNX expects [B, 3, H, W]
                batch_np = np.stack([
                    np.transpose(img, (2, 0, 1))
                    for img in tta_images
                ], axis=0).astype(np.float32)

                part_logits, color_logits = self._onnx_inference(batch_np)

            else:
                # PyTorch
                batch_tensor = torch.stack([
                    torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
                    for img in tta_images
                ])
                part_logits, color_logits = self._pytorch_inference(batch_tensor)
                part_logits = part_logits.numpy()
                color_logits = color_logits.numpy()

            # Average predictions across TTA
            part_logits = part_logits.mean(axis=0)  # [num_parts]
            color_logits = color_logits.mean(axis=0)  # [num_colors]

        else:
            # Single forward pass
            img_normalized = A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(image=image_np)['image']

            if self.use_onnx:
                img_array = np.transpose(img_normalized, (2, 0, 1))[np.newaxis, :].astype(np.float32)
                part_logits, color_logits = self._onnx_inference(img_array)
                part_logits = part_logits[0]  # [num_parts]
                color_logits = color_logits[0]  # [num_colors]

            else:
                img_tensor = torch.from_numpy(np.transpose(img_normalized, (2, 0, 1))).float().unsqueeze(0)
                part_logits_tensor, color_logits_tensor = self._pytorch_inference(img_tensor)
                part_logits = part_logits_tensor[0].numpy()
                color_logits = color_logits_tensor[0].numpy()

        # Convert logits to probabilities
        part_probs = self._softmax(part_logits)
        color_probs = self._softmax(color_logits)

        # Get top-k part predictions
        top_k_indices = np.argsort(part_probs)[::-1][:top_k]

        results = []
        for part_idx in top_k_indices:
            part_num = self.idx_to_part[int(part_idx)]
            confidence = float(part_probs[part_idx])

            # Get most confident color for this part
            best_color_idx = np.argmax(color_probs)
            color_id = self.idx_to_color[int(best_color_idx)]

            results.append({
                'part_num': part_num,
                'color_id': color_id,
                'confidence': confidence,
            })

        return results

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)

    def classify_batch(
        self,
        images: List[Union[str, Image.Image]],
        top_k: int = 5,
        use_tta: bool = False,
    ) -> List[List[Dict[str, Union[str, float]]]]:
        """
        Classify multiple images in batch.

        Args:
            images: List of paths or PIL Images
            top_k: Number of top predictions per image
            use_tta: Whether to use TTA (slower, typically disabled for batch inference)

        Returns:
            List of classification results, one list per image
        """
        results = []
        for image in images:
            result = self.classify(image, top_k=top_k, use_tta=use_tta)
            results.append(result)

        return results


def load_classifier(
    model_dir: str,
    use_onnx: bool = True,
) -> BrickClassifier:
    """
    Convenience function to load classifier from a directory.

    Expects:
        <model_dir>/brickscan_classifier.onnx
        <model_dir>/class_map.json

    Args:
        model_dir: Directory containing model and class map
        use_onnx: Whether to use ONNX (True) or PyTorch (False)

    Returns:
        Initialized BrickClassifier
    """
    model_dir = Path(model_dir)

    if use_onnx:
        model_path = model_dir / 'brickscan_classifier.onnx'
    else:
        model_path = model_dir / 'best_model.pt'

    class_map_path = model_dir / 'class_map.json'

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not class_map_path.exists():
        raise FileNotFoundError(f"Class map not found: {class_map_path}")

    return BrickClassifier(
        model_path=str(model_path),
        class_map_path=str(class_map_path),
        use_onnx=use_onnx,
    )


# FastAPI integration example (for reference)
def create_fastapi_app(model_dir: str):
    """
    Create a FastAPI app for brick classification.

    Example usage:
        from inference_classifier import create_fastapi_app
        app = create_fastapi_app('./outputs')
    """
    try:
        from fastapi import FastAPI, File, UploadFile
        from fastapi.responses import JSONResponse
    except ImportError:
        logger.warning("FastAPI not installed. Skipping app creation.")
        return None

    app = FastAPI(title="BrickScan Classifier")

    # Load classifier once at startup
    classifier = load_classifier(model_dir, use_onnx=True)

    @app.post("/classify")
    async def classify_image(file: UploadFile = File(...)):
        """Classify a single brick image."""
        try:
            # Read image from upload
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))

            # Classify
            results = classifier.classify(image, top_k=5, use_tta=True)

            return JSONResponse({
                'success': True,
                'predictions': results
            })
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return JSONResponse({
                'success': False,
                'error': str(e)
            }, status_code=500)

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {'status': 'healthy'}

    return app


if __name__ == '__main__':
    # Example usage
    import sys

    if len(sys.argv) < 3:
        print("Usage: python inference_classifier.py <model_dir> <image_path> [--no-tta]")
        sys.exit(1)

    model_dir = sys.argv[1]
    image_path = sys.argv[2]
    use_tta = '--no-tta' not in sys.argv

    logger.info(f"Loading model from {model_dir}")
    classifier = load_classifier(model_dir, use_onnx=True)

    logger.info(f"Classifying {image_path} (TTA: {use_tta})")
    results = classifier.classify(image_path, top_k=5, use_tta=use_tta)

    logger.info("Classification results:")
    for i, result in enumerate(results):
        logger.info(f"  {i+1}. {result['part_num']} - {result['color_id']} "
                   f"(confidence: {result['confidence']:.4f})")
