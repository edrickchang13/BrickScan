"""
Depth channel processing module for RGBD and RGBD+normals pipelines.

Handles loading 16-bit depth PNGs, computing surface normals, and combining
depth with RGB for multi-channel model inputs.
"""

import numpy as np
from typing import Optional, Tuple
from PIL import Image
import cv2


def load_depth_png(path: str) -> np.ndarray:
    """
    Load a 16-bit depth PNG and return as float32 array in millimeters.

    Args:
        path: Path to the 16-bit grayscale depth PNG.

    Returns:
        float32 numpy array with depth values in millimeters.
        Shape: (H, W)
    """
    img = Image.open(path)
    depth_16 = np.array(img, dtype=np.uint16)
    # Convert from uint16 (0-65535) directly to float32 (depth in mm)
    depth_mm = depth_16.astype(np.float32)
    return depth_mm


def compute_surface_normals(
    depth_mm: np.ndarray,
    fx: float = 500.0,
    fy: float = 500.0
) -> np.ndarray:
    """
    Compute surface normals from a depth map using Sobel gradients.

    The normal at each pixel is computed from the depth gradients:
    - nx = -dZ/dx (normalized)
    - ny = -dZ/dy (normalized)
    - nz = 1.0 (normalized)

    Args:
        depth_mm: Depth map in millimeters, shape (H, W).
        fx: Focal length in x (used for scaling gradients). Default 500.
        fy: Focal length in y (used for scaling gradients). Default 500.

    Returns:
        Surface normals as float32 array, shape (H, W, 3).
        Each normal is a unit vector [nx, ny, nz].
    """
    H, W = depth_mm.shape

    # Compute depth gradients using Sobel operators
    # Treat invalid/missing depth (0 or NaN) as no gradient
    depth_valid = depth_mm.copy()
    depth_valid[depth_valid <= 0] = 0  # Invalid depths

    # Sobel gradients
    gx = cv2.Sobel(depth_valid, cv2.CV_32F, 1, 0, ksize=3)  # dZ/dx
    gy = cv2.Sobel(depth_valid, cv2.CV_32F, 0, 1, ksize=3)  # dZ/dy

    # Scale by focal length to convert pixel gradients to 3D gradients
    gx = gx / fx
    gy = gy / fy

    # Normal components: surface normal = [-dZ/dx, -dZ/dy, 1]
    nx = -gx
    ny = -gy
    nz = np.ones((H, W), dtype=np.float32)

    # Stack into (H, W, 3)
    normals = np.stack([nx, ny, nz], axis=2)

    # Normalize each normal to unit length
    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    norms[norms == 0] = 1.0  # Avoid division by zero
    normals = normals / norms

    return normals.astype(np.float32)


def depth_to_4channel(
    rgb: np.ndarray,
    depth_mm: np.ndarray,
    target_size: Tuple[int, int] = (300, 300)
) -> np.ndarray:
    """
    Combine RGB + depth into a 4-channel image for model input.

    Depth is normalized to [0, 1] by clipping to [0, 1000mm] brick range.

    Args:
        rgb: RGB image, shape (H, W, 3), uint8 or float in [0, 255].
        depth_mm: Depth map in millimeters, shape (H, W).
        target_size: Target output size (H, W). Default (300, 300).

    Returns:
        4-channel float32 image [R, G, B, D_norm], shape (H, W, 4).
        All values normalized to [0, 1].
    """
    # Ensure RGB is float and in [0, 1]
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255.0
    else:
        rgb = rgb.astype(np.float32)
        if rgb.max() > 1.5:  # Likely in [0, 255] range
            rgb = rgb / 255.0

    # Resize RGB to target size if needed
    if rgb.shape[0] != target_size[0] or rgb.shape[1] != target_size[1]:
        rgb = cv2.resize(rgb, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

    # Resize depth to target size
    if depth_mm.shape[0] != target_size[0] or depth_mm.shape[1] != target_size[1]:
        depth_resized = cv2.resize(depth_mm, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    else:
        depth_resized = depth_mm.copy()

    # Normalize depth to [0, 1]
    # Clip to brick range [0, 1000mm]
    depth_normalized = np.clip(depth_resized, 0, 1000) / 1000.0
    depth_normalized = depth_normalized.astype(np.float32)

    # Stack into 4-channel
    rgba = np.concatenate([rgb, depth_normalized[..., np.newaxis]], axis=2)

    return rgba.astype(np.float32)


def depth_and_normals_to_6channel(
    rgb: np.ndarray,
    depth_mm: np.ndarray,
    target_size: Tuple[int, int] = (300, 300),
    fx: float = 500.0,
    fy: float = 500.0
) -> np.ndarray:
    """
    Combine RGB + depth + surface normals into 6-channel image.

    Output channels: [R, G, B, D_norm, Nx, Ny]
    (Nz is omitted since it's redundant for unit normals)

    Args:
        rgb: RGB image, shape (H, W, 3), uint8 or float in [0, 255].
        depth_mm: Depth map in millimeters, shape (H, W).
        target_size: Target output size (H, W). Default (300, 300).
        fx: Focal length in x. Default 500.
        fy: Focal length in y. Default 500.

    Returns:
        6-channel float32 image [R, G, B, D_norm, Nx, Ny], shape (H, W, 6).
        All values normalized to [0, 1] where applicable.
    """
    # Ensure RGB is float and in [0, 1]
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255.0
    else:
        rgb = rgb.astype(np.float32)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0

    # Resize RGB to target size
    if rgb.shape[0] != target_size[0] or rgb.shape[1] != target_size[1]:
        rgb = cv2.resize(rgb, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

    # Resize depth to target size
    if depth_mm.shape[0] != target_size[0] or depth_mm.shape[1] != target_size[1]:
        depth_resized = cv2.resize(depth_mm, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    else:
        depth_resized = depth_mm.copy()

    # Normalize depth to [0, 1]
    depth_normalized = np.clip(depth_resized, 0, 1000) / 1000.0
    depth_normalized = depth_normalized.astype(np.float32)

    # Compute surface normals
    normals = compute_surface_normals(depth_resized, fx=fx, fy=fy)

    # Extract Nx, Ny (drop Nz)
    nx = normals[..., 0]
    ny = normals[..., 1]

    # Normalize Nx, Ny from [-1, 1] to [0, 1] for network input
    nx_normalized = (nx + 1.0) / 2.0
    ny_normalized = (ny + 1.0) / 2.0

    # Stack into 6-channel
    output = np.stack([
        rgb[..., 0],  # R
        rgb[..., 1],  # G
        rgb[..., 2],  # B
        depth_normalized,  # D_norm
        nx_normalized,  # Nx
        ny_normalized,  # Ny
    ], axis=2)

    return output.astype(np.float32)
