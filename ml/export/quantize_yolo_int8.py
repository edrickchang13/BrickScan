"""Int8-quantize the YOLOv8-L detection ONNX for on-device inference.

Stage 1a of continuous-scan Phase 5. Produces yolo_lego.int8.onnx (~40 MB
target) from the 167 MB fp32 export, using static quantization with a
calibration set sampled from yolo_dataset/val (synthetic multi-piece scenes).

Usage:
    backend/venv/bin/python ml/export/quantize_yolo_int8.py \
        --input  backend/models/yolo_lego.onnx \
        --output backend/models/yolo_lego.int8.onnx \
        --calib-dir /Users/edrickchang/Documents/Claude/Projects/Lego/yolo_dataset/yolo_dataset/images/val \
        --num-calib 128
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process


YOLO_INPUT = 640


def letterbox(img: Image.Image, size: int = YOLO_INPUT) -> np.ndarray:
    """Letterbox an RGB PIL image to size×size, matching Ultralytics preprocess.

    Returns a float32 NCHW tensor in [0, 1].
    """
    img = img.convert("RGB")
    w, h = img.size
    scale = min(size / w, size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0  # HWC
    arr = np.transpose(arr, (2, 0, 1))[None, ...]  # 1,3,H,W
    return np.ascontiguousarray(arr)


class YoloCalibReader(CalibrationDataReader):
    def __init__(self, image_paths: list[Path], input_name: str):
        self.iter = iter(image_paths)
        self.input_name = input_name
        self.total = len(image_paths)
        self.done = 0

    def get_next(self):
        try:
            p = next(self.iter)
        except StopIteration:
            return None
        self.done += 1
        if self.done % 16 == 0 or self.done == self.total:
            print(f"  calib {self.done}/{self.total}", flush=True)
        img = Image.open(p)
        tensor = letterbox(img)
        return {self.input_name: tensor}

    def rewind(self):
        raise RuntimeError("rewind not supported; recreate the reader")


def pick_calib_images(calib_dir: Path, n: int, seed: int = 42) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    pool = [p for p in calib_dir.iterdir() if p.suffix.lower() in exts]
    if not pool:
        raise SystemExit(f"No images found in {calib_dir}")
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]


def get_input_name(onnx_path: Path) -> str:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return sess.get_inputs()[0].name


def find_detect_head_nodes(onnx_path: Path, head_prefix: str = "/model.22/") -> list[str]:
    """Collect all node names under the Ultralytics Detect head.

    Over-quantizing the final classification conv collapses all class scores
    to zero. Excluding the whole head module is a safer recipe than trying
    to guess the single culprit op; the head is a small fraction of total
    flops so leaving it fp32 barely affects the size/latency win.
    """
    import onnx

    m = onnx.load(str(onnx_path))
    names = [n.name for n in m.graph.node if n.name.startswith(head_prefix)]
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--num-calib", type=int, default=128)
    ap.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip shape-inference preprocess (use if already run)",
    )
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input ONNX not found: {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Shape-inference preprocess — required for clean static quant on YOLO.
    preprocessed = args.input.with_suffix(".preproc.onnx")
    if args.skip_preprocess and preprocessed.exists():
        print(f"[1/3] Reusing existing preprocessed ONNX: {preprocessed}")
    else:
        print(f"[1/3] Preprocessing (shape-inference) → {preprocessed}")
        t0 = time.time()
        # Ultralytics YOLOv8 exports sometimes trip symbolic shape inference
        # (dynamic axes on Detect head). Skip it — onnx shape inference + graph
        # optimization are enough for clean static quant.
        quant_pre_process(
            input_model_path=str(args.input),
            output_model_path=str(preprocessed),
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=True,
        )
        print(f"    preprocess done in {time.time() - t0:.1f}s")

    # Calibration set.
    print(f"[2/3] Sampling {args.num_calib} calibration images from {args.calib_dir}")
    calib_imgs = pick_calib_images(args.calib_dir, args.num_calib)
    input_name = get_input_name(preprocessed)
    print(f"    model input name: {input_name!r}")
    reader = YoloCalibReader(calib_imgs, input_name)

    # Static int8 quantization.
    print(f"[3/3] Quantizing → {args.output}")
    t0 = time.time()
    # Exclude the Ultralytics Detect head (/model.22/*) — quantizing it
    # collapses the class-score sigmoid to uniform zero. Head is a small
    # fraction of flops; leaving it fp32 barely changes size or latency.
    head_nodes = find_detect_head_nodes(preprocessed)
    print(f"    excluding {len(head_nodes)} Detect-head nodes from quantization")

    # YOLOv8 uses SiLU — asymmetric activations around zero. Symmetric
    # activation quant clips the negative lobe; asymmetric QInt8 (default)
    # is the right recipe. Symmetric per-channel weights are standard.
    quantize_static(
        model_input=str(preprocessed),
        model_output=str(args.output),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        nodes_to_exclude=head_nodes,
        extra_options={"WeightSymmetric": True},
    )
    dt = time.time() - t0
    size_mb = args.output.stat().st_size / (1024 * 1024)
    orig_mb = args.input.stat().st_size / (1024 * 1024)
    print(f"\nDone in {dt:.1f}s")
    print(f"  original: {orig_mb:.1f} MB")
    print(f"  int8   : {size_mb:.1f} MB ({size_mb / orig_mb * 100:.1f}% of original)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
