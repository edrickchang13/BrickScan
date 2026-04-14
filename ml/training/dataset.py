"""PyTorch Dataset for LEGO brick part + color classification."""

import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def build_label_encoders(csv_path: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Build label encoders from the index.csv file.

    Args:
        csv_path: Path to the index.csv file with columns:
                  image_path, part_num, color_id, color_name, color_r, color_g, color_b

    Returns:
        Tuple of (part_encoder, color_encoder) where each is a dict mapping label to index
    """
    df = pd.read_csv(csv_path)

    # Create encoders from unique values
    part_encoder = {part_num: idx for idx, part_num in enumerate(sorted(df['part_num'].unique()))}
    color_encoder = {color_name: idx for idx, color_name in enumerate(sorted(df['color_name'].unique()))}

    return part_encoder, color_encoder


def save_label_encoders(
    part_encoder: Dict[str, int],
    color_encoder: Dict[str, int],
    output_dir: str
) -> None:
    """
    Save label encoders to JSON files.

    Args:
        part_encoder: Dict mapping part_num to index
        color_encoder: Dict mapping color_name to index
        output_dir: Directory to save the JSON files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'part_labels.json', 'w') as f:
        json.dump(part_encoder, f, indent=2)

    with open(output_dir / 'color_labels.json', 'w') as f:
        json.dump(color_encoder, f, indent=2)


class LegoPartsDataset(Dataset):
    """PyTorch Dataset for LEGO brick classification."""

    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        part_encoder: Dict[str, int],
        color_encoder: Dict[str, int],
        is_train: bool = True,
        image_size: int = 224
    ):
        """
        Initialize the dataset.

        Args:
            csv_path: Path to index.csv file
            images_dir: Base directory containing images
            part_encoder: Dict mapping part_num to index
            color_encoder: Dict mapping color_name to index
            is_train: Whether to use training or validation augmentations
            image_size: Size to resize images to (default: 224 for ImageNet)
        """
        self.df = pd.read_csv(csv_path)
        self.images_dir = Path(images_dir)
        self.part_encoder = part_encoder
        self.color_encoder = color_encoder
        self.is_train = is_train
        self.image_size = image_size

        # ImageNet normalization constants
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]

        # Define augmentation pipelines
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.333),
                    interpolation=Image.BILINEAR
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.imagenet_mean, std=self.imagenet_std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(int(image_size * 1.14)),  # 256 for 224x224
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.imagenet_mean, std=self.imagenet_std)
            ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        Get a sample from the dataset.

        Returns:
            Tuple of (image_tensor, part_label_idx, color_label_idx)
        """
        row = self.df.iloc[idx]

        # Load and augment image
        image_path = self.images_dir / row['image_path']
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)

        # Get label indices
        part_label_idx = self.part_encoder[row['part_num']]
        color_label_idx = self.color_encoder[row['color_name']]

        return image_tensor, part_label_idx, color_label_idx
