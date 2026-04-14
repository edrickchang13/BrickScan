"""
ModelManager: Lazy-loading singleton for the three new BrickScan ML models.

Model files live at  brickscan/backend/models/:
  contrastive_encoder.onnx  → encode_image()     → np.float32[128]  (L2-normalised)
  distilled_student.onnx    → classify_image()   → List[dict]       (top-5 predictions)
  yolo_lego.onnx            → detect_pieces()    → List[BoundingBox]
  class_labels.json         → index → part_num   (used by student + k-NN)

All methods return None / [] gracefully when model files are absent, so the
existing EfficientNet / Brickognize / Gemini cascade keeps working during
development before the Spark training completes.

Input conventions (must match training scripts):
  contrastive_encoder : 224x224  BICUBIC, ImageNet normalisation (DINOv2 backbone, trained --image-size 224)
  distilled_student   : 224x224  LANCZOS, ImageNet normalisation (MobileNetV3-Small)
  yolo_lego           : 640x640  LETTERBOX, scale 0-1 (YOLOv8 convention)
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# backend/app/ml/model_manager.py  →  parent x3  =  backend/
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = _BACKEND_DIR / "models"

# ImageNet mean/std (CHW, float32)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

YOLO_INPUT_SIZE     = 640
YOLO_CONF_THRESHOLD = 0.30
YOLO_IOU_THRESHOLD  = 0.45


@dataclass
class BoundingBox:
    """Normalised [0-1] bounding box returned by detect_pieces()."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    crop_bytes: bytes = field(default=b"", repr=False)


class ModelManager:
    """
    Singleton owning the three new ML ONNX sessions.

    Usage:
        mm = ModelManager.get()
        emb   = mm.encode_image(img_bytes)      # np.array[128] or None
        preds = mm.classify_image(img_bytes)    # List[dict] or []
        boxes = mm.detect_pieces(img_bytes)     # List[BoundingBox] or []
    """

    _instance: Optional["ModelManager"] = None

    def __init__(self) -> None:
        self._encoder_session: Optional[Any] = None
        self._student_session: Optional[Any] = None
        self._yolo_session:    Optional[Any] = None
        self._class_labels:    Optional[Dict[str, str]] = None
        self._loaded: bool = False

    @classmethod
    def get(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── lazy init ─────────────────────────────────────────────────────────────
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        self._load_all()

    def _ort_providers(self) -> List[str]:
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" in ort.get_available_providers():
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            pass
        return ["CPUExecutionProvider"]

    def _load_all(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            logger.warning("onnxruntime not installed — new ML models disabled")
            return

        providers = self._ort_providers()

        def _try_load(name: str) -> Optional[Any]:
            path = MODELS_DIR / name
            if not path.exists():
                logger.info("%s not found — will be enabled after Spark training", name)
                return None
            try:
                sess = ort.InferenceSession(str(path), providers=providers)
                logger.info("Loaded %s (%s)", name, sess.get_providers()[0])
                return sess
            except Exception as e:
                logger.error("Failed to load %s: %s", name, e)
                return None

        self._encoder_session = _try_load("contrastive_encoder.onnx")
        self._student_session = _try_load("distilled_student.onnx")
        self._yolo_session    = _try_load("yolo_lego.onnx")

        labels_path = MODELS_DIR / "class_labels.json"
        if labels_path.exists():
            try:
                with open(labels_path) as f:
                    raw: Dict = json.load(f)
                if "idx2part" in raw:
                    self._class_labels = raw["idx2part"]
                elif raw and isinstance(next(iter(raw.values())), int):
                    # Inverted format: {part_num: index}
                    self._class_labels = {str(v): k for k, v in raw.items()}
                else:
                    self._class_labels = {str(k): str(v) for k, v in raw.items()}
                logger.info("Class labels: %d classes", len(self._class_labels))
            except Exception as e:
                logger.error("Failed to load class_labels.json: %s", e)

    # ── public properties ──────────────────────────────────────────────────────
    @property
    def encoder_available(self) -> bool:
        self._ensure_loaded(); return self._encoder_session is not None

    @property
    def student_available(self) -> bool:
        self._ensure_loaded(); return self._student_session is not None

    @property
    def yolo_available(self) -> bool:
        self._ensure_loaded(); return self._yolo_session is not None

    # ── public methods ─────────────────────────────────────────────────────────
    def encode_image(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Contrastive encoder → L2-normalised float32[128]. Returns None if unavailable.
        Note: trained with --image-size 224 (DINOv2 ViT-B/14 patch14 supports any multiple of 14).
        """
        self._ensure_loaded()
        if self._encoder_session is None:
            return None
        try:
            tensor = _preprocess(image_bytes, size=224)
            in_name  = self._encoder_session.get_inputs()[0].name
            out_name = self._encoder_session.get_outputs()[0].name
            emb = self._encoder_session.run([out_name], {in_name: tensor})[0][0].astype(np.float32)
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 1e-8 else emb
        except Exception as e:
            logger.error("encode_image: %s", e)
            return None

    def classify_image(self, image_bytes: bytes, top_k: int = 5) -> List[Dict[str, Any]]:
        """Distilled student → top-k predictions. Returns [] if unavailable."""
        self._ensure_loaded()
        if self._student_session is None:
            return []
        try:
            tensor = _preprocess(image_bytes, size=224)
            in_name  = self._student_session.get_inputs()[0].name
            out_name = self._student_session.get_outputs()[0].name
            logits = self._student_session.run([out_name], {in_name: tensor})[0][0]
            probs  = _softmax(logits)
            top_idx = probs.argsort()[::-1][:top_k]
            results = []
            for idx in top_idx:
                part_num = (
                    self._class_labels.get(str(idx), f"unknown_{idx}")
                    if self._class_labels else f"class_{idx}"
                )
                results.append({
                    "part_num":   part_num,
                    "part_name":  "",
                    "confidence": float(probs[idx]),
                    "color_id":   None,
                    "color_name": None,
                    "color_hex":  None,
                    "source":     "distilled_model",
                })
            return results
        except Exception as e:
            logger.error("classify_image: %s", e)
            return []

    def detect_pieces(self, image_bytes: bytes) -> List[BoundingBox]:
        """YOLOv8 detector → list of normalised bounding boxes with cropped bytes."""
        self._ensure_loaded()
        if self._yolo_session is None:
            return []
        try:
            from PIL import Image as PILImage
            img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
            orig_w, orig_h = img.size

            tensor, (pad_x, pad_y, scale) = _letterbox(img, YOLO_INPUT_SIZE)
            in_name  = self._yolo_session.get_inputs()[0].name
            out_name = self._yolo_session.get_outputs()[0].name
            raw = self._yolo_session.run([out_name], {in_name: tensor})[0]

            # Normalise to [N, 5+cls]: handle both [1, 5, N] and [1, N, 5]
            arr = raw[0] if raw.ndim == 3 else raw
            if arr.shape[0] < arr.shape[1]:
                arr = arr.T          # → [N, 5+cls]

            boxes: List[BoundingBox] = []
            seen: List[np.ndarray]   = []

            for row in arr:
                conf = float(row[4])
                if conf < YOLO_CONF_THRESHOLD:
                    continue
                cx, cy, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                cx0 = (cx - pad_x) / scale
                cy0 = (cy - pad_y) / scale
                bw0 = bw / scale
                bh0 = bh / scale
                x1 = max(0.0, (cx0 - bw0 / 2) / orig_w)
                y1 = max(0.0, (cy0 - bh0 / 2) / orig_h)
                x2 = min(1.0, (cx0 + bw0 / 2) / orig_w)
                y2 = min(1.0, (cy0 + bh0 / 2) / orig_h)
                if x2 <= x1 or y2 <= y1:
                    continue
                b = np.array([x1, y1, x2, y2])
                if any(_iou(b, s) > YOLO_IOU_THRESHOLD for s in seen):
                    continue
                seen.append(b)
                crop = img.crop((int(x1 * orig_w), int(y1 * orig_h),
                                 int(x2 * orig_w), int(y2 * orig_h)))
                buf = io.BytesIO()
                crop.save(buf, format="JPEG", quality=85)
                boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2,
                                         confidence=conf, crop_bytes=buf.getvalue()))

            boxes.sort(key=lambda b: b.confidence, reverse=True)
            return boxes
        except Exception as e:
            logger.error("detect_pieces: %s", e)
            return []


# ── helpers ────────────────────────────────────────────────────────────────────

def _preprocess(image_bytes: bytes, size: int = 224) -> np.ndarray:
    from PIL import Image as PILImage
    img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    resample = PILImage.Resampling.BICUBIC if size >= 224 else PILImage.Resampling.LANCZOS
    img = img.resize((size, size), resample)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = (arr - _MEAN) / _STD
    return np.expand_dims(arr, 0)


def _letterbox(img: Any, target: int):
    from PIL import Image as PILImage
    w, h = img.size
    scale = target / max(w, h)
    nw, nh = int(w * scale), int(h * scale)
    resized = img.resize((nw, nh), PILImage.Resampling.BILINEAR)
    canvas = PILImage.new("RGB", (target, target), (114, 114, 114))
    px, py = (target - nw) // 2, (target - nh) // 2
    canvas.paste(resized, (px, py))
    arr = np.array(canvas, dtype=np.float32) / 255.0
    return np.expand_dims(arr.transpose(2, 0, 1), 0), (float(px), float(py), scale)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max()); return e / e.sum()


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
    xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0
