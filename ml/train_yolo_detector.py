#!/usr/bin/env python3
"""
YOLOv8 Training Script for LEGO Piece Detection

This script:
1. Generates synthetic detection training data by compositing rendered LEGO parts
   onto randomized backgrounds
2. Trains YOLOv8m for single-class detection ("lego_piece")
3. Uses strong augmentation (mosaic, mixup, HSV)
4. Trains for 100 epochs with early stopping (patience=15)
5. Exports the best model to ONNX format
6. Saves training metrics and plots

Hardware: NVIDIA DGX Spark with GB10 Blackwell GPU (130.6GB VRAM)
Python Environment: ~/brickscan/ml/venv with torch, torchvision, PIL, numpy, pandas, tqdm
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple
import random
import shutil
import subprocess

import cv2
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================

class BackgroundGenerator:
    """Generate varied backgrounds for synthetic training data."""

    @staticmethod
    def solid_color(size: Tuple[int, int]) -> Image.Image:
        """Generate a solid color background."""
        colors = [
            (255, 255, 255),  # white
            (200, 200, 200),  # light gray
            (128, 128, 128),  # medium gray
            (64, 64, 64),     # dark gray
            (240, 240, 240),  # off-white
            (220, 220, 220),  # lighter gray
        ]
        color = random.choice(colors)
        return Image.new('RGB', size, color)

    @staticmethod
    def gradient(size: Tuple[int, int]) -> Image.Image:
        """Generate a gradient background."""
        img = Image.new('RGB', size)
        draw = ImageDraw.Draw(img)

        # Random color range
        r1, g1, b1 = random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)
        r2, g2, b2 = random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)

        for i in range(size[1]):
            ratio = i / size[1]
            r = int(r1 * (1 - ratio) + r2 * ratio)
            g = int(g1 * (1 - ratio) + g2 * ratio)
            b = int(b1 * (1 - ratio) + b2 * ratio)
            draw.line([(0, i), (size[0], i)], fill=(r, g, b))

        return img

    @staticmethod
    def noise(size: Tuple[int, int]) -> Image.Image:
        """Generate a noisy background."""
        base_color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
        img = Image.new('RGB', size, base_color)

        # Add noise
        pixels = img.load()
        for i in range(size[0]):
            for j in range(size[1]):
                if random.random() < 0.1:
                    noise_val = random.randint(-30, 30)
                    r = max(0, min(255, base_color[0] + noise_val))
                    g = max(0, min(255, base_color[1] + noise_val))
                    b = max(0, min(255, base_color[2] + noise_val))
                    pixels[i, j] = (r, g, b)

        return img

    @staticmethod
    def texture(size: Tuple[int, int]) -> Image.Image:
        """Generate a textured background (wood-like)."""
        img = Image.new('RGB', size)
        pixels = np.array(img)

        # Wood-like texture
        for i in range(size[0]):
            base = np.sin(i / 50) * 50 + 150
            for j in range(size[1]):
                val = int(base + np.sin(j / 30) * 30 + random.randint(-20, 20))
                val = max(0, min(255, val))
                pixels[j, i] = [val, val - 20, val - 40]

        return Image.fromarray(pixels.astype('uint8'))

    @classmethod
    def generate(cls, size: Tuple[int, int]) -> Image.Image:
        """Generate a random background."""
        method = random.choice([cls.solid_color, cls.gradient, cls.noise, cls.texture])
        return method(size)


class SyntheticDataGenerator:
    """Generate synthetic YOLO detection training data."""

    def __init__(self,
                 parts_dir: str,
                 output_dir: str,
                 image_size: int = 640,
                 num_train: int = 5000,
                 num_val: int = 1000):
        """
        Initialize the synthetic data generator.

        Args:
            parts_dir: Directory containing rendered LEGO part images
            output_dir: Output directory for YOLO dataset
            image_size: Size of generated images (square)
            num_train: Number of training images to generate
            num_val: Number of validation images to generate
        """
        self.parts_dir = Path(parts_dir)
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.num_train = num_train
        self.num_val = num_val

        # Setup output directories
        self.train_dir = self.output_dir / 'images' / 'train'
        self.train_labels_dir = self.output_dir / 'labels' / 'train'
        self.val_dir = self.output_dir / 'images' / 'val'
        self.val_labels_dir = self.output_dir / 'labels' / 'val'

        for d in [self.train_dir, self.train_labels_dir, self.val_dir, self.val_labels_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Load available part images
        self.part_images = self._load_part_images()
        logger.info(f"Found {len(self.part_images)} part images")

        if len(self.part_images) == 0:
            raise ValueError(f"No part images found in {parts_dir}")

    def _load_part_images(self) -> List[Path]:
        """Load all PNG images from parts directory."""
        parts = []
        if self.parts_dir.exists():
            parts = list(self.parts_dir.glob('*.png'))
        return parts

    def _load_image_with_alpha(self, image_path: Path) -> Tuple[Image.Image, Image.Image]:
        """
        Load image and ensure it has alpha channel for transparency.

        Returns:
            (image_RGB, alpha_mask)
        """
        img = Image.open(image_path)

        if img.mode == 'RGBA':
            alpha = img.split()[3]
            img_rgb = img.convert('RGB')
        elif img.mode == 'RGB':
            img_rgb = img
            alpha = Image.new('L', img.size, 255)
        else:
            img = img.convert('RGBA')
            alpha = img.split()[3]
            img_rgb = img.convert('RGB')

        return img_rgb, alpha

    def _get_part_bbox(self, alpha_mask: Image.Image,
                       offset_x: int, offset_y: int) -> Tuple[int, int, int, int]:
        """
        Get bounding box from alpha mask, adjusted for offset.
        Returns (x_min, y_min, x_max, y_max) in absolute image coordinates.
        """
        alpha_array = np.array(alpha_mask)
        coords = np.where(alpha_array > 127)

        if len(coords[0]) == 0:
            return None

        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()

        return (x_min + offset_x, y_min + offset_y, x_max + offset_x, y_max + offset_y)

    def _apply_random_transformations(self, img: Image.Image) -> Image.Image:
        """Apply random transformations to part image."""
        # Random rotation
        if random.random() < 0.7:
            angle = random.uniform(-15, 15)
            img = img.rotate(angle, expand=False, fillcolor=(255, 255, 255))

        # Random scale
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        return img

    def generate_image(self, is_train: bool = True) -> Tuple[Image.Image, List[str]]:
        """
        Generate a single synthetic image with LEGO parts.

        Returns:
            (image, list of YOLO format annotation strings)
        """
        # Create background
        bg = BackgroundGenerator.generate((self.image_size, self.image_size))

        # Random number of parts (3-15)
        num_parts = random.randint(3, 15)
        annotations = []

        # Track placed bboxes to detect overlap
        placed_bboxes = []

        for _ in range(num_parts):
            if len(self.part_images) == 0:
                break

            # Random part image
            part_path = random.choice(self.part_images)
            part_img, part_alpha = self._load_image_with_alpha(part_path)

            # Apply transformations
            part_img = self._apply_random_transformations(part_img)
            part_alpha = self._apply_random_transformations(part_alpha)

            # Random scale (parts take up 5-25% of image)
            max_scale = min(self.image_size / max(part_img.width, part_img.height) * 0.25,
                           self.image_size / max(part_img.width, part_img.height) * 0.95)
            scale = random.uniform(0.05, max_scale)
            new_size = (int(part_img.width * scale), int(part_img.height * scale))

            part_img = part_img.resize(new_size, Image.Resampling.LANCZOS)
            part_alpha = part_alpha.resize(new_size, Image.Resampling.LANCZOS)

            # Random position
            max_x = max(0, self.image_size - part_img.width)
            max_y = max(0, self.image_size - part_img.height)

            if max_x <= 0 or max_y <= 0:
                continue

            offset_x = random.randint(0, max_x)
            offset_y = random.randint(0, max_y)

            # Paste on background
            bg.paste(part_img, (offset_x, offset_y), part_alpha)

            # Get bbox
            bbox = self._get_part_bbox(part_alpha, offset_x, offset_y)
            if bbox:
                placed_bboxes.append(bbox)

                # Convert to YOLO format (normalized center_x, center_y, width, height)
                x_min, y_min, x_max, y_max = bbox
                center_x = (x_min + x_max) / 2 / self.image_size
                center_y = (y_min + y_max) / 2 / self.image_size
                width = (x_max - x_min) / self.image_size
                height = (y_max - y_min) / self.image_size

                # Clip to valid range
                center_x = max(0, min(1, center_x))
                center_y = max(0, min(1, center_y))
                width = max(0, min(1, width))
                height = max(0, min(1, height))

                annotations.append(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")

        # Apply slight augmentation to the final image
        if random.random() < 0.3:
            bg = ImageOps.autocontrast(bg)

        if random.random() < 0.3:
            bg = bg.filter(ImageFilter.GaussianBlur(radius=0.5))

        return bg, annotations

    def generate_dataset(self):
        """Generate entire training and validation dataset."""
        logger.info(f"Generating {self.num_train} training images...")
        for i in tqdm(range(self.num_train), desc="Training data"):
            img, annots = self.generate_image(is_train=True)

            # Save image
            img_path = self.train_dir / f"img_{i:05d}.jpg"
            img.save(img_path, quality=95)

            # Save annotations
            if annots:
                label_path = self.train_labels_dir / f"img_{i:05d}.txt"
                with open(label_path, 'w') as f:
                    f.write('\n'.join(annots))

        logger.info(f"Generating {self.num_val} validation images...")
        for i in tqdm(range(self.num_val), desc="Validation data"):
            img, annots = self.generate_image(is_train=False)

            # Save image
            img_path = self.val_dir / f"img_{i:05d}.jpg"
            img.save(img_path, quality=95)

            # Save annotations
            if annots:
                label_path = self.val_labels_dir / f"img_{i:05d}.txt"
                with open(label_path, 'w') as f:
                    f.write('\n'.join(annots))

        logger.info("Synthetic dataset generation complete!")
        self._create_yolo_yaml()

    def _create_yolo_yaml(self):
        """Create YAML file for YOLO dataset configuration."""
        yaml_content = f"""path: {self.output_dir.absolute()}
train: images/train
val: images/val

nc: 1
names: ['lego_piece']
"""
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        logger.info(f"Created dataset config: {yaml_path}")


# ============================================================================
# YOLO TRAINING
# ============================================================================

def ensure_ultralytics_installed():
    """Ensure ultralytics (YOLOv8) is installed."""
    try:
        import ultralytics
        logger.info(f"ultralytics version {ultralytics.__version__} is installed")
    except ImportError:
        logger.info("Installing ultralytics...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
        logger.info("ultralytics installed successfully")


def train_yolo_model(dataset_config: str,
                     output_dir: str,
                     epochs: int = 100,
                     patience: int = 15,
                     batch_size: int = 32,
                     imgsz: int = 640,
                     device: str = '0'):
    """
    Train YOLOv8m model for LEGO piece detection.

    Args:
        dataset_config: Path to YOLO dataset.yaml
        output_dir: Output directory for results
        epochs: Number of training epochs
        patience: Early stopping patience
        batch_size: Batch size for training
        imgsz: Image size for training
        device: CUDA device ID
    """
    from ultralytics import YOLO

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Loading YOLOv8m model...")
    model = YOLO('yolov8m.pt')

    logger.info("Starting training...")
    results = model.train(
        data=dataset_config,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        patience=patience,
        device=device,
        project=output_dir,
        name='yolo_detector',
        exist_ok=True,

        # Strong augmentation
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        # Training parameters
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_bias_lr=0.1,

        # Augmentation
        degrees=10,
        translate=0.1,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        perspective=0.0,

        # Save and log
        save=True,
        save_period=10,
        verbose=True,
    )

    logger.info("Training complete!")
    return results


def export_to_onnx(model_path: str, output_dir: str):
    """
    Export trained YOLOv8 model to ONNX format.

    Args:
        model_path: Path to trained model (best.pt)
        output_dir: Output directory for ONNX model
    """
    from ultralytics import YOLO

    logger.info(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Exporting to ONNX...")
    export_path = model.export(format='onnx', half=False)

    logger.info(f"Model exported to: {export_path}")
    return export_path


def save_training_metrics(results, output_dir: str):
    """Save training metrics to JSON and create plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_file = output_path / 'training_metrics.json'
    metrics = {
        'epochs': len(results.epoch) if hasattr(results, 'epoch') else 0,
        'best_fitness': float(results.best_fitness) if hasattr(results, 'best_fitness') else None,
    }

    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics saved to {metrics_file}")


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main(args):
    """Main training pipeline."""
    logger.info("=" * 80)
    logger.info("YOLO v8 LEGO Piece Detector Training Pipeline")
    logger.info("=" * 80)

    # Ensure ultralytics is installed
    ensure_ultralytics_installed()

    # Setup directories
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    dataset_output = output_base / 'dataset'
    training_output = output_base / 'training'
    export_output = output_base / 'models'

    # Step 1: Generate synthetic training data
    if args.generate_data:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: GENERATING SYNTHETIC TRAINING DATA")
        logger.info("=" * 80)

        if not Path(args.parts_dir).exists():
            logger.error(f"Parts directory not found: {args.parts_dir}")
            sys.exit(1)

        generator = SyntheticDataGenerator(
            parts_dir=args.parts_dir,
            output_dir=str(dataset_output),
            image_size=args.image_size,
            num_train=args.num_train,
            num_val=args.num_val
        )
        generator.generate_dataset()
    else:
        logger.info("Skipping data generation (--no-generate-data flag set)")

    # Step 2: Train YOLOv8 model
    if args.train:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: TRAINING YOLO v8m MODEL")
        logger.info("=" * 80)

        dataset_yaml = dataset_output / 'data.yaml'
        if not dataset_yaml.exists():
            logger.error(f"Dataset config not found: {dataset_yaml}")
            logger.error("Run with --generate-data first")
            sys.exit(1)

        results = train_yolo_model(
            dataset_config=str(dataset_yaml),
            output_dir=str(training_output),
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            imgsz=args.image_size,
            device=args.device
        )

        save_training_metrics(results, str(export_output))
    else:
        logger.info("Skipping training (--no-train flag set)")

    # Step 3: Export to ONNX
    if args.export_onnx:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: EXPORTING MODEL TO ONNX")
        logger.info("=" * 80)

        model_path = training_output / 'yolo_detector' / 'weights' / 'best.pt'
        if not model_path.exists():
            logger.error(f"Trained model not found: {model_path}")
            logger.error("Run training first")
            sys.exit(1)

        export_to_onnx(str(model_path), str(export_output))
    else:
        logger.info("Skipping ONNX export (--no-export flag set)")

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_base}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='YOLOv8 Training Script for LEGO Piece Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:

1. Full pipeline (generate data + train + export):
   python train_yolo_detector.py \\
       --parts-dir ~/brickscan/ml/data/test_renders \\
       --output-dir ./yolo_results

2. Train only (skip data generation):
   python train_yolo_detector.py \\
       --parts-dir ~/brickscan/ml/data/test_renders \\
       --output-dir ./yolo_results \\
       --no-generate-data

3. Generate data only:
   python train_yolo_detector.py \\
       --parts-dir ~/brickscan/ml/data/test_renders \\
       --output-dir ./yolo_results \\
       --no-train \\
       --no-export

4. Custom hyperparameters:
   python train_yolo_detector.py \\
       --parts-dir ~/brickscan/ml/data/test_renders \\
       --output-dir ./yolo_results \\
       --epochs 50 \\
       --batch-size 64 \\
       --patience 10
        """
    )

    # Data generation arguments
    parser.add_argument('--parts-dir',
                       type=str,
                       default='~/brickscan/ml/data/test_renders',
                       help='Directory containing rendered LEGO part images')

    parser.add_argument('--output-dir',
                       type=str,
                       default='./yolo_detector',
                       help='Output directory for dataset, training, and models')

    parser.add_argument('--image-size',
                       type=int,
                       default=640,
                       help='Training image size (square)')

    parser.add_argument('--num-train',
                       type=int,
                       default=5000,
                       help='Number of training images to generate')

    parser.add_argument('--num-val',
                       type=int,
                       default=1000,
                       help='Number of validation images to generate')

    # Training arguments
    parser.add_argument('--epochs',
                       type=int,
                       default=100,
                       help='Number of training epochs')

    parser.add_argument('--batch-size',
                       type=int,
                       default=32,
                       help='Batch size for training')

    parser.add_argument('--patience',
                       type=int,
                       default=15,
                       help='Early stopping patience (epochs without improvement)')

    parser.add_argument('--device',
                       type=str,
                       default='0',
                       help='CUDA device ID (default: 0)')

    # Pipeline control
    parser.add_argument('--no-generate-data',
                       action='store_true',
                       help='Skip synthetic data generation')

    parser.add_argument('--no-train',
                       action='store_true',
                       help='Skip model training')

    parser.add_argument('--no-export',
                       action='store_true',
                       help='Skip ONNX export')

    args = parser.parse_args()

    # Expand user paths
    args.parts_dir = os.path.expanduser(args.parts_dir)
    args.output_dir = os.path.expanduser(args.output_dir)

    return args


if __name__ == '__main__':
    args = parse_arguments()

    try:
        main(args)
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
