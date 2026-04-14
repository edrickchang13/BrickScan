"""Dual-head EfficientNet classifier for LEGO parts and colors."""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


class LegoBrickClassifier(nn.Module):
    """
    Dual-head EfficientNet-B3 classifier for LEGO parts and colors.

    Architecture:
        Input: 224x224 RGB image
        Backbone: EfficientNet-B3 (pretrained on ImageNet)
        Shared features (1536 dim) -> GlobalAvgPool
          ├─ part_head:  Linear(1536 → 512 → num_parts)   + dropout(0.4)
          └─ color_head: Linear(1536 → 256 → num_colors)  + dropout(0.3)
    """

    def __init__(self, num_parts: int, num_colors: int):
        """
        Initialize the classifier.

        Args:
            num_parts: Number of distinct LEGO part types
            num_colors: Number of distinct LEGO colors
        """
        super().__init__()
        self.num_parts = num_parts
        self.num_colors = num_colors

        # Load pretrained EfficientNet-B3
        self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)

        # Remove original classifier
        self.backbone.classifier = nn.Identity()

        # Feature dimension from EfficientNet-B3 (after avgpool)
        self.feature_dim = 1536

        # Part classification head
        self.part_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_parts)
        )

        # Color classification head
        self.color_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_colors)
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass for training.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Tuple of (part_logits, color_logits)
                - part_logits: (batch_size, num_parts)
                - color_logits: (batch_size, num_colors)
        """
        # Extract features from backbone
        features = self.backbone(x)

        # Pass through heads
        part_logits = self.part_head(features)
        color_logits = self.color_head(features)

        return part_logits, color_logits

    def forward_onnx(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ONNX export.

        Concatenates logits from both heads into a single output tensor.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Concatenated logits of shape (batch_size, num_parts + num_colors)
        """
        # Extract features from backbone
        features = self.backbone(x)

        # Pass through heads
        part_logits = self.part_head(features)
        color_logits = self.color_head(features)

        # Concatenate: [batch_size, num_parts + num_colors]
        return torch.cat([part_logits, color_logits], dim=1)
