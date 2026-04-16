"""
Grad-CAM overlay generation for ONNX classifier predictions.

Produces a heatmap showing which regions of the input image drove the top
prediction — per the EuroPython 2020 paper on LEGO multi-label LIME vs Grad-CAM
(https://arxiv.org/abs/2008.01584), users trust predictions with visible
heatmaps far more than bare scores.

Implementation note: ONNX doesn't expose intermediate activations the way
PyTorch hooks do, so we do one of:

    1. If a PyTorch .pt / .pth checkpoint is present alongside the ONNX model,
       load it, attach a hook to the last conv block, compute Grad-CAM
       classically. (Preferred, accurate.)

    2. If only an ONNX model is present, fall back to "occlusion
       sensitivity" — slide a gray patch over the image, measure confidence
       drop per position. Approximate but model-agnostic.

Returns a PIL RGBA overlay matching input dimensions that the frontend can
alpha-blend over the scan photo.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

@dataclass
class GradCAMResult:
    """Container for the explanation output."""
    heatmap: np.ndarray        # H×W float32 in [0, 1]
    overlay_png: bytes         # RGBA PNG bytes ready to ship to the client
    method: str                # "gradcam" | "occlusion"


def generate_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.55,
    cmap: str = "jet",
) -> bytes:
    """
    Alpha-blend `heatmap` (H×W float in [0, 1]) onto `image` and return PNG bytes.
    """
    from matplotlib import cm
    import matplotlib.pyplot as plt  # noqa: F401  (for cmap registration)

    cmap_fn = cm.get_cmap(cmap)
    heatmap_norm = np.clip(heatmap, 0.0, 1.0)
    heat_rgba = (cmap_fn(heatmap_norm) * 255).astype(np.uint8)  # H×W×4
    # Modulate the alpha channel by heatmap intensity so cool regions are fully transparent
    heat_rgba[..., 3] = (heatmap_norm * 255 * alpha).astype(np.uint8)

    # Resize heatmap to match the source image
    heat_img = Image.fromarray(heat_rgba, mode="RGBA").resize(
        image.size, resample=Image.Resampling.BILINEAR
    )
    base = image.convert("RGBA")
    out = Image.alpha_composite(base, heat_img)

    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Path 1: proper Grad-CAM via PyTorch
# ---------------------------------------------------------------------------

def gradcam_pytorch(
    image: Image.Image,
    checkpoint_path: Path,
    target_layer_name: str = "features.16",  # MobileNetV3-Large last conv block
    target_class: Optional[int] = None,
    img_size: int = 224,
) -> Optional[GradCAMResult]:
    """
    Classical Grad-CAM on a PyTorch checkpoint.
    Returns None if torch isn't installed or the checkpoint can't be loaded.
    """
    try:
        import torch
        import torch.nn.functional as F
        from torchvision import models, transforms
    except ImportError:
        logger.info("torch not installed; skipping PyTorch Grad-CAM path")
        return None

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        logger.warning("Could not load checkpoint %s: %s", checkpoint_path, e)
        return None

    # Construct a MobileNetV3-Large with the right number of classes
    state_dict = ckpt.get("model_state_dict", ckpt)
    num_classes = None
    for key, val in state_dict.items():
        if key.endswith("classifier.3.weight") or key.endswith("classifier.weight"):
            num_classes = val.shape[0]
            break
    if num_classes is None:
        logger.warning("Could not infer num_classes; skipping Grad-CAM")
        return None

    model = models.mobilenet_v3_large(weights=None, num_classes=num_classes)
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        logger.warning("state_dict load failed: %s", e)
        return None
    model.eval()

    # Transforms
    tfm = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = tfm(image.convert("RGB")).unsqueeze(0).requires_grad_(True)

    # Hook the target layer
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def fwd_hook(_m, _in, out): activations.append(out)
    def bwd_hook(_m, _gin, gout): gradients.append(gout[0])

    # Find the layer
    layer = model
    for part in target_layer_name.split("."):
        if part.isdigit():
            layer = layer[int(part)]
        else:
            layer = getattr(layer, part)
    fh = layer.register_forward_hook(fwd_hook)
    bh = layer.register_full_backward_hook(bwd_hook)

    try:
        logits = model(x)
        cls = int(target_class) if target_class is not None else int(logits.argmax(dim=1).item())
        model.zero_grad()
        logits[0, cls].backward()

        acts = activations[0]      # [1, C, h, w]
        grads = gradients[0]        # [1, C, h, w]
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * acts).sum(dim=1).squeeze(0)    # [h, w]
        cam = F.relu(cam)
        cam = cam.detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    finally:
        fh.remove()
        bh.remove()

    overlay = generate_overlay(image.convert("RGB"), cam)
    return GradCAMResult(heatmap=cam, overlay_png=overlay, method="gradcam")


# ---------------------------------------------------------------------------
# Path 2: occlusion sensitivity via ONNX
# ---------------------------------------------------------------------------

def occlusion_sensitivity_onnx(
    image: Image.Image,
    onnx_session,  # onnxruntime.InferenceSession
    target_class: int,
    img_size: int = 224,
    patch: int = 32,
    stride: int = 16,
) -> GradCAMResult:
    """
    Slide a gray patch over the image, record confidence drop at each position.
    Model-agnostic but O(img_size^2 / stride^2) forward passes — expensive.
    Use stride=16 for a 14x14 map on a 224 image (196 fwd passes).
    """
    img = image.convert("RGB").resize((img_size, img_size))
    arr = np.asarray(img).astype(np.float32) / 255.0
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normed = (arr - mean) / std
    x_base = normed.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)  # 1×3×H×W

    input_name = onnx_session.get_inputs()[0].name
    base_logits = onnx_session.run(None, {input_name: x_base})[0]
    base_conf = _softmax(base_logits[0])[target_class]

    heat = np.zeros((img_size, img_size), dtype=np.float32)
    count = np.zeros_like(heat) + 1e-8
    gray = 0.5  # mid-gray occluder (post-normalization this is near-zero, fine)

    for y in range(0, img_size - patch + 1, stride):
        for x in range(0, img_size - patch + 1, stride):
            x_occ = x_base.copy()
            x_occ[0, :, y:y + patch, x:x + patch] = gray
            logits = onnx_session.run(None, {input_name: x_occ})[0]
            conf = _softmax(logits[0])[target_class]
            drop = base_conf - conf
            heat[y:y + patch, x:x + patch] += drop
            count[y:y + patch, x:x + patch] += 1.0

    heat /= count
    heat = np.clip(heat, 0.0, None)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    overlay = generate_overlay(img, heat)
    return GradCAMResult(heatmap=heat, overlay_png=overlay, method="occlusion")


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


# ---------------------------------------------------------------------------
# Smart dispatcher
# ---------------------------------------------------------------------------

def explain(
    image: Image.Image,
    target_class: int,
    pytorch_checkpoint: Optional[Path] = None,
    onnx_session=None,
    img_size: int = 224,
) -> Optional[GradCAMResult]:
    """
    Preferred Grad-CAM when a PyTorch ckpt is available; otherwise occlusion
    sensitivity via ONNX. Returns None if neither path is viable.
    """
    if pytorch_checkpoint and Path(pytorch_checkpoint).exists():
        result = gradcam_pytorch(image, Path(pytorch_checkpoint), target_class=target_class, img_size=img_size)
        if result is not None:
            return result
    if onnx_session is not None:
        return occlusion_sensitivity_onnx(image, onnx_session, target_class=target_class, img_size=img_size)
    return None
