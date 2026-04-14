"""
ML inference service for BrickScan LEGO part classifier.

Supports two model families, auto-detected from export_info.json or ONNX output shape:

  EfficientNet-B3 (dual-head, legacy):
    Input:  [batch, 3, 224, 224]
    Output: [batch, num_parts + num_colors] — concatenated logits
      parts  = output[:, :num_parts]  → softmax → top-k part indices
      colors = output[:, num_parts:]  → softmax → argmax color index

  DINOv2 ViT-B/14 (single-head, current):
    Input:  [batch, 3, 518, 518]
    Output: [batch, num_parts] — part logits only
    Color:  not predicted by this model (returns None)

Label files (JSON) live alongside the .onnx file:
  part_labels.json   {"idx2part": {"0": "3001", ...}, "num_classes": N}
  color_labels.json  {"idx2color": {"0": {"id":1, "name":"White", "hex":"F2F3F2"}, ...}}
  export_info.json   written by export_dinov2_onnx.py — used for model-type detection
"""

from __future__ import annotations

import json
import os
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Module-level singletons ────────────────────────────────────────────────────
_session: Optional[Any] = None          # onnxruntime.InferenceSession
_part_labels: Optional[Dict] = None     # decoded from part_labels.json
_color_labels: Optional[Dict] = None    # decoded from color_labels.json
_part_names: Dict[str, str] = {}        # part_num -> human-readable name
_num_parts: int = 0
_input_size: int = 224                  # 224 for EfficientNet, 518 for DINOv2
_is_single_head: bool = False           # True for DINOv2 (parts only, no color head)

# ImageNet normalization constants (match training augmentation)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _load_json(path: str) -> Optional[Dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not load %s: %s", path, e)
        return None


def _try_load_part_names(model_dir: str) -> None:
    """Try to load part names from JSON cache. Silent fail if not found."""
    global _part_names
    names_path = os.path.join(model_dir, "part_names.json")
    data = _load_json(names_path)
    if data:
        _part_names = data
        logger.info("Loaded %d part names from cache", len(_part_names))


def _normalize_labels(raw: Dict) -> tuple:
    """
    Accept either label format:
      A) {"idx2part": {"0": "3001", ...}, "num_classes": N}   (old format)
      B) {"3001": 0, "3002": 1, ...}                         (training export)

    Returns (idx2label, num_classes) where idx2label maps str(index) → label.
    """
    if "idx2part" in raw or "idx2color" in raw:
        # Format A — already in the right shape
        mapping = raw.get("idx2part") or raw.get("idx2color") or {}
        return mapping, raw.get("num_classes", len(mapping))

    # Format B — keys are labels, values are integer indices
    # Invert the mapping: index → label
    idx2label = {str(v): k for k, v in raw.items() if isinstance(v, int)}
    return idx2label, len(idx2label)


def _detect_model_type(model_path: str, model_dir: str, num_parts: int) -> tuple[int, bool]:
    """
    Auto-detect input size and whether the model is single-head (DINOv2) or
    dual-head (EfficientNet).

    Detection priority:
      1. export_info.json written by export_dinov2_onnx.py  → authoritative
      2. ML_MODEL_TYPE env/setting: "dinov2" or "efficientnet"
      3. ONNX output dimension: if == num_parts → single-head (DINOv2)

    Returns (input_size, is_single_head).
    """
    from app.core.config import settings

    # 1. Check export_info.json
    info_path = os.path.join(model_dir, "export_info.json")
    info = _load_json(info_path)
    if info:
        model_type = info.get("model", "").lower()
        if "dinov2" in model_type or info.get("heads") == ["parts"]:
            size = info.get("input_size", 518)
            logger.info("export_info.json: DINOv2 detected, input_size=%d", size)
            return size, True
        elif "efficientnet" in model_type:
            size = info.get("input_size", 224)
            logger.info("export_info.json: EfficientNet detected, input_size=%d", size)
            return size, False

    # 2. Explicit setting
    model_type_setting = getattr(settings, "ML_MODEL_TYPE", "").lower()
    if model_type_setting == "dinov2":
        logger.info("ML_MODEL_TYPE=dinov2 → single-head, input=518")
        return 518, True
    if model_type_setting == "efficientnet":
        logger.info("ML_MODEL_TYPE=efficientnet → dual-head, input=224")
        return 224, False

    # 3. Infer from ONNX output shape
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        out_shape = sess.get_outputs()[0].shape
        if len(out_shape) >= 2 and isinstance(out_shape[1], int):
            out_dim = out_shape[1]
            if out_dim == num_parts:
                logger.info("ONNX output dim=%d == num_parts → DINOv2 single-head, input=518", out_dim)
                return 518, True
            else:
                logger.info("ONNX output dim=%d > num_parts=%d → EfficientNet dual-head, input=224",
                            out_dim, num_parts)
                return 224, False
    except Exception:
        pass

    # 4. Safe fallback
    logger.warning("Could not detect model type — defaulting to EfficientNet (input=224, dual-head)")
    return 224, False


def load_model() -> bool:
    """Load the ONNX model and label decoders.  Returns True on success."""
    global _session, _part_labels, _color_labels, _num_parts, _input_size, _is_single_head

    from app.core.config import settings

    model_path = getattr(settings, "ML_MODEL_PATH", "")
    if not model_path or not os.path.exists(model_path):
        logger.info("ML model not found at '%s' — ML inference disabled", model_path)
        return False

    # Derive label paths from the same directory as the model
    model_dir = str(Path(model_path).parent)
    part_labels_path  = os.path.join(model_dir, "part_labels.json")
    color_labels_path = os.path.join(model_dir, "color_labels.json")

    raw_parts = _load_json(part_labels_path)
    if not raw_parts:
        logger.warning("part_labels.json missing — ML inference disabled")
        return False

    # Color labels are optional for DINOv2 (single-head model has no color output)
    raw_colors = _load_json(color_labels_path)

    try:
        import onnxruntime as ort

        # Grace Blackwell / DGX Spark: prefer CUDA EP, fall back to CPU
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        session = ort.InferenceSession(model_path, providers=providers)

        idx2part, num_parts = _normalize_labels(raw_parts)

        # Auto-detect model family (DINOv2 vs EfficientNet)
        detected_input_size, detected_single_head = _detect_model_type(
            model_path, model_dir, num_parts
        )

        _session        = session
        _part_labels    = {"idx2part": idx2part, "num_classes": num_parts}
        _num_parts      = num_parts
        _input_size     = detected_input_size
        _is_single_head = detected_single_head

        if raw_colors:
            idx2color, num_colors = _normalize_labels(raw_colors)
            _color_labels = {"idx2color": idx2color, "num_classes": num_colors}
        else:
            _color_labels = None
            num_colors = 0

        # Try to load part names cache
        _try_load_part_names(model_dir)

        ep = session.get_providers()[0]
        logger.info(
            "ML model loaded: %s  parts=%d  colors=%d  input=%dx%d  single_head=%s  EP=%s",
            Path(model_path).name, _num_parts, num_colors,
            _input_size, _input_size, _is_single_head, ep,
        )
        return True

    except Exception as e:
        logger.error("Failed to load ONNX model: %s", e)
        return False


def _preprocess(image_bytes: bytes) -> np.ndarray:
    """
    Decode bytes → normalised CHW float32 tensor [1, 3, H, H].

    Input size is 518 for DINOv2 (ViT-B/14) and 224 for EfficientNet.
    Uses BICUBIC resampling as in training for DINOv2; LANCZOS otherwise.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    size = _input_size
    resample = Image.Resampling.BICUBIC if size == 518 else Image.Resampling.LANCZOS
    img = img.resize((size, size), resample)
    arr = np.array(img, dtype=np.float32) / 255.0   # HWC [0,1]
    arr = arr.transpose(2, 0, 1)                     # CHW
    arr = (arr - _MEAN) / _STD                       # ImageNet norm
    return np.expand_dims(arr, 0)                    # [1, 3, H, H]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


async def predict(image_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Run inference and return top-3 predictions with part + color info.

    Works with both model families:
    - DINOv2 (single-head): returns parts only; color fields are None.
    - EfficientNet (dual-head): returns parts + color from concatenated output.
    """
    if _session is None:
        if not load_model():
            return []

    if _session is None:
        return []

    try:
        tensor = _preprocess(image_bytes)
        input_name  = _session.get_inputs()[0].name
        output_name = _session.get_outputs()[0].name

        raw = _session.run([output_name], {input_name: tensor})[0][0]

        # ── Part logits ──────────────────────────────────────────────────────
        part_logits = raw[:_num_parts]
        part_probs  = _softmax(part_logits)
        top3_idx    = part_probs.argsort()[::-1][:3]
        idx2part    = _part_labels.get("idx2part", {})

        # ── Color logits (EfficientNet dual-head only) ────────────────────────
        color_info: Dict = {"id": None, "name": None, "hex": None}
        if not _is_single_head and _color_labels is not None:
            color_logits = raw[_num_parts:]
            color_probs  = _softmax(color_logits)
            color_idx    = int(color_probs.argmax())
            idx2color    = _color_labels.get("idx2color", {})
            raw_color    = idx2color.get(str(color_idx), {})
            if isinstance(raw_color, dict):
                color_info = raw_color
            elif raw_color:
                color_info = {"name": str(raw_color), "id": None, "hex": None}

        results = []
        for pidx in top3_idx:
            part_num   = idx2part.get(str(pidx), f"unknown_{pidx}")
            confidence = float(part_probs[pidx])
            results.append({
                "part_num":   part_num,
                "part_name":  _part_names.get(part_num, ""),
                "confidence": confidence,
                "color_id":   color_info.get("id"),
                "color_name": color_info.get("name"),
                "color_hex":  color_info.get("hex"),
            })

        logger.debug(
            "ML prediction (%s): part=%s (%.1f%%)  color=%s",
            "dinov2" if _is_single_head else "efficientnet",
            results[0]["part_num"], results[0]["confidence"] * 100,
            results[0].get("color_name") or "n/a",
        )
        return results

    except Exception as e:
        logger.error("ML inference error: %s", e)
        return []
