"""Data augmentation pipelines for synthetic-to-real domain adaptation."""

import os
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm


def get_synthetic_to_real_transform() -> A.Compose:
    """Augmentations to make synthetic renders look like real iPhone photos.

    These augmentations simulate:
    - Lighting variations
    - Camera noise (especially in lower light)
    - Compression artifacts from JPEG
    - Shadows and occlusions
    - Slight perspective distortion

    Returns:
        Albumentations Compose transform
    """
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.7),
        A.GaussNoise(var_limit=(10, 50), p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.ImageCompression(quality_lower=70, quality_upper=100, p=0.4),
        A.Perspective(scale=(0.05, 0.1), p=0.5),
        A.RandomShadow(p=0.3),
        A.CoarseDropout(max_holes=3, max_height=20, max_width=20, p=0.2),
    ])


def get_train_transform(input_size: int = 224) -> A.Compose:
    """Full training augmentation pipeline with geometric transforms.

    Args:
        input_size: Input image size

    Returns:
        Albumentations Compose transform
    """
    return A.Compose([
        A.LongestMaxSize(max_size=input_size + 32),
        A.PadIfNeeded(
            min_height=input_size + 32,
            min_width=input_size + 32,
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.RandomCrop(height=input_size, width=input_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=45, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.6),
        A.GaussNoise(var_limit=(10, 50), p=0.4),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transform(input_size: int = 224) -> A.Compose:
    """Validation/test transform with minimal augmentation.

    Args:
        input_size: Input image size

    Returns:
        Albumentations Compose transform
    """
    return A.Compose([
        A.LongestMaxSize(max_size=input_size),
        A.PadIfNeeded(
            min_height=input_size,
            min_width=input_size,
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.CenterCrop(height=input_size, width=input_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def apply_augmentation_to_image(
    image_path: str,
    output_path: str,
    transform: Optional[A.Compose] = None,
) -> None:
    """Apply augmentation to a single image and save.

    Args:
        image_path: Path to input image
        output_path: Path to save augmented image
        transform: Albumentations transform to apply (uses synthetic-to-real by default)
    """
    if transform is None:
        transform = get_synthetic_to_real_transform()

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply augmentation
    augmented = transform(image=img)
    augmented_img = augmented["image"]

    # Convert back to BGR for cv2.imwrite
    augmented_img = cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR)

    # Save
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(output_path, augmented_img)


def augment_dataset(
    input_dir: str,
    output_dir: str,
    multiplier: int = 3,
    use_synthetic_to_real: bool = True,
) -> None:
    """Augment all images in a dataset by creating multiple copies.

    For each image in input_dir/{class}/{img}.png, generates `multiplier`
    augmented copies in output_dir/{class}/ with different random augmentations.

    Args:
        input_dir: Root directory with structure: {class}/{image.jpg}
        output_dir: Output directory (same structure will be created)
        multiplier: Number of augmented versions per image
        use_synthetic_to_real: Use synthetic-to-real augmentation
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise ValueError(f"Input directory not found: {input_dir}")

    # Choose augmentation
    if use_synthetic_to_real:
        augmentation = get_synthetic_to_real_transform()
    else:
        augmentation = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.Rotate(limit=45, p=0.5),
        ])

    # Iterate through classes
    class_dirs = [d for d in input_path.iterdir() if d.is_dir()]

    for class_dir in tqdm(class_dirs, desc="Augmenting classes"):
        class_name = class_dir.name
        output_class_dir = output_path / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        # Process each image
        image_files = sorted([
            f for f in class_dir.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        for img_file in tqdm(image_files, desc=f"Class {class_name}", leave=False):
            # Load image
            img = cv2.imread(str(img_file))
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Save original
            output_img_path = output_class_dir / img_file.name
            output_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_img_path), output_img)

            # Create augmented versions
            for aug_idx in range(multiplier):
                augmented = augmentation(image=img)
                augmented_img = augmented["image"]

                # Create output filename
                stem = img_file.stem
                suffix = img_file.suffix
                aug_filename = f"{stem}_aug_{aug_idx}{suffix}"
                aug_path = output_class_dir / aug_filename

                # Save
                aug_img_bgr = cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(aug_path), aug_img_bgr)

    print(f"Augmentation complete! Output saved to {output_dir}")


def apply_synthetic_to_real_to_dataset(input_dir: str, output_dir: str) -> None:
    """Apply synthetic-to-real augmentation to dataset for domain adaptation.

    This is useful when you have purely synthetic data and want to make it
    look more like real-world photos before training.

    Args:
        input_dir: Input dataset directory with structure: {class}/{image.jpg}
        output_dir: Output directory where augmented images will be saved
    """
    augment_dataset(input_dir, output_dir, multiplier=1, use_synthetic_to_real=True)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Augment LEGO dataset")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--multiplier", type=int, default=3, help="Augmentation multiplier")
    parser.add_argument(
        "--synthetic_to_real",
        action="store_true",
        help="Use synthetic-to-real augmentation",
    )

    args = parser.parse_args()

    augment_dataset(
        args.input_dir,
        args.output_dir,
        args.multiplier,
        args.synthetic_to_real,
    )
