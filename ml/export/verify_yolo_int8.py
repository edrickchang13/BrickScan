"""Sanity-check the int8-quantized YOLOv8-L against the fp32 baseline.

Runs both models on a sampled set of val images and reports:
  - mean per-image detection-count delta (fp32 vs int8)
  - mean IoU of matched top-K boxes
  - mean score delta of matched boxes
  - wall-clock inference latency (CPU, single-threaded)

This is a smoke test — not a full mAP eval (would need the Roboflow Hex:Lego
val split, which isn't on this box). If IoU > 0.85 for matched boxes and
count delta < 15% we're in "acceptable" territory for on-device use.

Usage:
    backend/venv/bin/python ml/export/verify_yolo_int8.py \
        --fp32 backend/models/yolo_lego.onnx \
        --int8 backend/models/yolo_lego.int8.onnx \
        --calib-dir /Users/edrickchang/Documents/Claude/Projects/Lego/yolo_dataset/yolo_dataset/images/val \
        --num-images 32
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from quantize_yolo_int8 import letterbox


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """Greedy NMS. boxes: (N,4) xyxy, scores: (N,). Returns sorted indices kept."""
    if len(boxes) == 0:
        return []
    order = np.argsort(-scores)
    keep: list[int] = []
    suppressed = np.zeros(len(boxes), dtype=bool)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    for i_ in order:
        i = int(i_)
        if suppressed[i]:
            continue
        keep.append(i)
        xx1 = np.maximum(x1[i], x1)
        yy1 = np.maximum(y1[i], y1)
        xx2 = np.minimum(x2[i], x2)
        yy2 = np.minimum(y2[i], y2)
        iw = np.maximum(0.0, xx2 - xx1)
        ih = np.maximum(0.0, yy2 - yy1)
        inter = iw * ih
        union = areas[i] + areas - inter
        iou = np.where(union > 0, inter / union, 0.0)
        suppressed |= iou >= iou_thresh
        suppressed[i] = False  # keep self
    return keep


def decode_yolo(out: np.ndarray, score_thresh: float = 0.25, iou_thresh: float = 0.45):
    """Ultralytics YOLOv8 head: (1, 4+C, N) with cxcywh + raw class logits.

    Returns (boxes_xyxy, scores, class_idx) post-NMS in letterboxed pixel coords.
    """
    arr = out[0]  # (4+C, N)
    c_plus_4 = arr.shape[0]
    num_classes = c_plus_4 - 4
    boxes_cxcywh = arr[:4].T  # (N, 4)
    class_probs = arr[4:].T  # (N, C) — Ultralytics already applies sigmoid in newer exports
    if class_probs.max() > 1.0 or class_probs.min() < 0.0:
        class_probs = sigmoid(class_probs)
    scores = class_probs.max(axis=1)
    classes = class_probs.argmax(axis=1)
    mask = scores >= score_thresh
    boxes_cxcywh = boxes_cxcywh[mask]
    scores = scores[mask]
    classes = classes[mask]
    if len(boxes_cxcywh) == 0:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)
    # cxcywh → xyxy
    cx, cy, w, h = boxes_cxcywh.T
    boxes_xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    keep = nms(boxes_xyxy, scores, iou_thresh)
    return boxes_xyxy[keep], scores[keep], classes[keep]


def pairwise_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a.T
    bx1, by1, bx2, by2 = b.T
    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def match(a_boxes, b_boxes, iou_thresh: float = 0.5):
    if len(a_boxes) == 0 or len(b_boxes) == 0:
        return []
    iou_mat = pairwise_iou(a_boxes, b_boxes)
    pairs: list[tuple[int, int, float]] = []
    used_b = set()
    for i in np.argsort(-iou_mat.max(axis=1)):
        j = int(np.argmax(iou_mat[i]))
        if j in used_b:
            continue
        if iou_mat[i, j] < iou_thresh:
            continue
        pairs.append((int(i), j, float(iou_mat[i, j])))
        used_b.add(j)
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", type=Path, required=True)
    ap.add_argument("--int8", type=Path, required=True)
    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--num-images", type=int, default=32)
    args = ap.parse_args()

    rng = random.Random(1337)
    imgs = sorted(args.calib_dir.glob("*.jpg"))
    rng.shuffle(imgs)
    imgs = imgs[: args.num_images]

    sess_fp32 = ort.InferenceSession(str(args.fp32), providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(args.int8), providers=["CPUExecutionProvider"])
    input_name = sess_fp32.get_inputs()[0].name

    total_fp32_boxes = 0
    total_int8_boxes = 0
    matched_ious: list[float] = []
    score_deltas: list[float] = []
    class_agree = 0
    class_total = 0
    fp32_lats: list[float] = []
    int8_lats: list[float] = []

    for i, p in enumerate(imgs):
        tensor = letterbox(Image.open(p))

        t0 = time.perf_counter()
        out32 = sess_fp32.run(None, {input_name: tensor})[0]
        fp32_lats.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        out8 = sess_int8.run(None, {input_name: tensor})[0]
        int8_lats.append((time.perf_counter() - t0) * 1000)

        b32, s32, c32 = decode_yolo(out32)
        b8, s8, c8 = decode_yolo(out8)
        total_fp32_boxes += len(b32)
        total_int8_boxes += len(b8)

        pairs = match(b32, b8)
        for i32, i8, iou in pairs:
            matched_ious.append(iou)
            score_deltas.append(abs(s32[i32] - s8[i8]))
            class_total += 1
            if c32[i32] == c8[i8]:
                class_agree += 1

        if (i + 1) % 8 == 0:
            print(f"  {i + 1}/{len(imgs)}", flush=True)

    print("\n=== fp32 vs int8 YOLOv8-L on", len(imgs), "images ===")
    print(f"fp32 boxes total: {total_fp32_boxes}   int8 boxes total: {total_int8_boxes}")
    if total_fp32_boxes:
        pct = (total_int8_boxes - total_fp32_boxes) / total_fp32_boxes * 100
        print(f"count delta      : {pct:+.1f}% (int8 vs fp32)")
    if matched_ious:
        print(f"matched pairs    : {len(matched_ious)} / {total_fp32_boxes}")
        print(f"mean IoU@matched : {np.mean(matched_ious):.3f}")
        print(f"mean |Δscore|    : {np.mean(score_deltas):.3f}")
        print(f"class agreement  : {class_agree}/{class_total} ({class_agree / class_total * 100:.1f}%)")
    print(f"fp32 latency     : {np.mean(fp32_lats):.1f} ms  (median {np.median(fp32_lats):.1f})")
    print(f"int8 latency     : {np.mean(int8_lats):.1f} ms  (median {np.median(int8_lats):.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
