"""Backend-package shim for the Grad-CAM / occlusion-sensitivity helpers.

The actual implementation lives under ml/inference/gradcam.py so it can be
reused by offline tooling. We re-export the public surface here so the API
layer can import it with `from app.ml.gradcam import explain`.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Path: backend/app/ml/gradcam.py -> repo root is 4 levels up; ml/inference sits
# next to the backend directory.
_repo_root = Path(__file__).resolve().parent.parent.parent.parent
_ml_inference = _repo_root / "ml" / "inference"
if _ml_inference.exists() and str(_ml_inference) not in sys.path:
    sys.path.insert(0, str(_ml_inference))

try:
    from gradcam import (  # type: ignore  # noqa: F401
        GradCAMResult,
        explain,
        generate_overlay,
        gradcam_pytorch,
        occlusion_sensitivity_onnx,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Could not locate ml/inference/gradcam.py. Ensure the ml directory "
        "is present alongside the backend package."
    ) from e

__all__ = [
    "GradCAMResult",
    "explain",
    "generate_overlay",
    "gradcam_pytorch",
    "occlusion_sensitivity_onnx",
]
