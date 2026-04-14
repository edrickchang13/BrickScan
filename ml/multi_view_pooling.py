"""
Multi-View Attention Pooling for LEGO brick classification.

Takes N frame feature vectors from the EfficientNet backbone (1536-dim each)
and produces a single pooled representation via self-attention with a CLS token.
This replaces the simple weighted vote accumulation in the live scan.

Architecture:
- Positional encoding: learnable frame-index embedding (max 16 frames)
- Self-attention: 4 heads, 1 layer (lightweight for iPhone inference)
- CLS token: prepend a learned CLS token, use its output as pooled repr
- Final classifier head: Linear(1536 → num_parts) + Linear(1536 → num_colors)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from PIL import Image
import numpy as np


class MultiViewPooling(nn.Module):
    """
    Multi-view feature pooling via self-attention.

    Attributes:
        num_parts: Number of LEGO part classes
        num_colors: Number of LEGO color classes
        feature_dim: Feature dimension from backbone (default 1536 for EfficientNet)
        max_frames: Maximum number of frames to process (default 16)
        num_heads: Number of attention heads (default 4)
        num_layers: Number of attention layers (default 1)
    """

    def __init__(
        self,
        num_parts: int,
        num_colors: int,
        feature_dim: int = 1536,
        max_frames: int = 16,
        num_heads: int = 4,
        num_layers: int = 1,
        backbone_checkpoint: Optional[str] = None,
    ):
        super().__init__()
        self.num_parts = num_parts
        self.num_colors = num_colors
        self.feature_dim = feature_dim
        self.max_frames = max_frames
        self.num_heads = num_heads
        self.num_layers = num_layers

        # CLS token: learnable parameter prepended to sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        # Positional encoding for frame index (learnable)
        self.pos_embedding = nn.Embedding(max_frames + 1, feature_dim)

        # Multi-head self-attention (lightweight for iPhone)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification heads
        self.part_classifier = nn.Linear(feature_dim, num_parts)
        self.color_classifier = nn.Linear(feature_dim, num_colors)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

        # Load backbone checkpoint if provided
        self.backbone = None
        if backbone_checkpoint:
            self.backbone = self._load_backbone(backbone_checkpoint)

    def _load_backbone(self, checkpoint_path: str):
        """Load EfficientNet backbone from checkpoint (stub for training)."""
        # In production, this would load the trained EfficientNet model
        # For now, return a placeholder that subclasses should implement
        try:
            from efficientnet_pytorch import EfficientNet
            backbone = EfficientNet.from_pretrained('efficientnet-b0')
            # Remove classification head, keep only features
            backbone = nn.Sequential(*list(backbone.children())[:-1])
            return backbone
        except ImportError:
            print("Warning: EfficientNet not available; backbone inference will not work")
            return None

    def encode_frames(
        self,
        frames: List[Image.Image],
    ) -> torch.Tensor:
        """
        Extract feature vectors from frames using EfficientNet backbone.

        Args:
            frames: List of PIL Images

        Returns:
            Tensor of shape [N, 1536] where N is number of frames
        """
        if self.backbone is None:
            raise RuntimeError("Backbone not loaded. Please provide backbone_checkpoint")

        device = next(self.parameters()).device
        self.backbone.to(device)
        self.backbone.eval()

        features_list = []
        with torch.no_grad():
            for frame in frames:
                # Convert PIL Image to tensor
                # Assume EfficientNet expects normalized ImageNet input
                frame_np = np.array(frame.convert('RGB'))
                # Simple normalization (ImageNet stats)
                frame_t = torch.from_numpy(frame_np).permute(2, 0, 1).float() / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                frame_t = (frame_t - mean) / std

                frame_t = frame_t.unsqueeze(0).to(device)

                # Extract features
                features = self.backbone(frame_t)
                # Flatten to [1, 1536]
                features = features.view(1, -1)
                features_list.append(features)

        # Concatenate all frame features
        return torch.cat(features_list, dim=0)

    def forward(self, frame_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pool multi-frame features and classify.

        Args:
            frame_features: Tensor of shape [N, 1536] where N <= max_frames

        Returns:
            (part_logits, color_logits) where each is [1, num_classes]
        """
        N = frame_features.shape[0]
        assert N <= self.max_frames, f"Too many frames: {N} > {self.max_frames}"

        # Add CLS token at the beginning
        batch_size = frame_features.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)  # [N, 1, 1536]

        # Add positional embeddings
        frame_indices = torch.arange(N, device=frame_features.device)
        pos_embed = self.pos_embedding(frame_indices)  # [N, 1536]

        # Combine frame features with positional embeddings
        frame_features = frame_features + pos_embed  # [N, 1536]

        # Concatenate CLS token with frame features
        # CLS gets pos embedding index 0 (special)
        cls_pos = self.pos_embedding(torch.tensor([0], device=frame_features.device))
        cls = cls + cls_pos.unsqueeze(0)  # [1, 1536]

        x = torch.cat([cls, frame_features.unsqueeze(0)], dim=1) \
            if cls.shape[0] == 1 \
            else torch.cat([cls, frame_features], dim=1)  # [N, 1+N, 1536]

        # Actually, let me fix this — cls should be added to each batch
        # For single batch inference:
        x = torch.cat([cls.squeeze(0).unsqueeze(0), frame_features.unsqueeze(0)], dim=1)
        # x is now [1, N+1, 1536]

        # Apply transformer
        x = self.transformer(x)  # [1, N+1, 1536]

        # Extract CLS token output (first position)
        cls_output = x[:, 0, :]  # [1, 1536]

        # Apply dropout
        cls_output = self.dropout(cls_output)

        # Classify
        part_logits = self.part_classifier(cls_output)  # [1, num_parts]
        color_logits = self.color_classifier(cls_output)  # [1, num_colors]

        return part_logits, color_logits


class MultiViewPoolingInference:
    """
    Wrapper for inference-time multi-view pooling.

    Handles batching, device management, and post-processing.
    """

    def __init__(
        self,
        model: MultiViewPooling,
        device: str = 'cpu',
    ):
        self.model = model.to(device).eval()
        self.device = device

    def predict(
        self,
        frames: List[Image.Image],
        top_k: int = 3,
    ) -> dict:
        """
        Classify frames and return top-k predictions.

        Args:
            frames: List of PIL Images
            top_k: Return top K predictions for each class

        Returns:
            dict with keys:
                - part_num: predicted part number
                - confidence: confidence for part prediction
                - top_parts: list of top-k predictions
                - top_colors: list of top-k color predictions
        """
        with torch.no_grad():
            # Extract features
            frame_features = self.model.encode_frames(frames)
            frame_features = frame_features.to(self.device)

            # Forward pass
            part_logits, color_logits = self.model(frame_features)

            # Get top-k predictions
            part_probs = F.softmax(part_logits, dim=1)
            color_probs = F.softmax(color_logits, dim=1)

            part_top = torch.topk(part_probs, k=min(top_k, self.model.num_parts))
            color_top = torch.topk(color_probs, k=min(top_k, self.model.num_colors))

        return {
            'part_logits': part_logits.cpu().numpy(),
            'color_logits': color_logits.cpu().numpy(),
            'part_probs': part_probs.cpu().numpy(),
            'color_probs': color_probs.cpu().numpy(),
            'top_part_scores': part_top.values.cpu().numpy()[0],
            'top_part_indices': part_top.indices.cpu().numpy()[0],
            'top_color_scores': color_top.values.cpu().numpy()[0],
            'top_color_indices': color_top.indices.cpu().numpy()[0],
        }
