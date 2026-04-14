#!/usr/bin/env python3
"""
Benchmark DINOv2 ONNX vs EfficientNet-B4 ONNX for BrickScan.

Compares top-1 / top-5 accuracy and inference latency across three data sources:
  1. Rebrickable catalog images  (downloaded on-the-fly via CDN)
  2. Local sample images         (backend/data_pipeline/sample_images/images/)
  3. Synthetic test set          (Rebrickable part images, no training overlap)

Usage:
    python benchmark_models.py \
        --dinov2  /app/models/dinov2/dinov2_lego.onnx \
        --effnet  /app/models/efficientnet/lego_classifier.onnx \
        --dinov2-labels  /app/models/dinov2/part_labels.json \
        --effnet-labels  /app/models/efficientnet/part_labels.json \
        [--sample-images  brickscan/backend/data_pipeline/sample_images/images] \
        [--download-n     50]    # how many Rebrickable images to download for bench
        [--output         benchmark_results.json]

Output: a JSON report + a terminal table.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ImageNet stats — shared between both models
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

# Known common LEGO parts from Rebrickable for synthetic test set
REBRICKABLE_TEST_PARTS = [
    "3001", "3002", "3003", "3004", "3005", "3006", "3007", "3008", "3009", "3010",
    "3020", "3021", "3022", "3023", "3024", "3031", "3032", "3033", "3034", "3035",
    "3037", "3038", "3039", "3040", "3045", "3046", "3048", "3062b", "3068b", "3069b",
    "3070b", "3298", "3460", "3623", "3666", "3710", "3711", "3747a", "3795", "3832",
    "4073", "4150", "4162", "4286", "4460b", "4477", "4488", "4589", "6091", "6141",
]

REBRICKABLE_CDN = "https://cdn.rebrickable.com/media/parts/elements/{part_num}.jpg"
REBRICKABLE_FALLBACK = "https://cdn.rebrickable.com/media/parts/ldraw/{part_num}.png"


# ──────────────────────────────────────────────────────────────────────────────
# Model wrapper
# ──────────────────────────────────────────────────────────────────────────────

class ONNXModel:
    def __init__(self, model_path: str, labels_path: str, input_size: int):
        try:
            import onnxruntime as ort
        except ImportError:
            raise SystemExit("Run: pip install onnxruntime")

        self.input_size = input_size
        self.model_path = Path(model_path)

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.ep = self.session.get_providers()[0]

        labels = json.loads(Path(labels_path).read_text())
        raw = labels.get("idx2part", labels)
        self.idx2part = {int(k): v for k, v in raw.items()}
        self.part2idx = {v: k for k, v in self.idx2part.items()}
        self.num_classes = labels.get("num_classes", len(self.idx2part))

        # Detect if dual-head (EfficientNet) — output dim > num_classes
        out_shape = self.session.get_outputs()[0].shape
        self.is_dual_head = (
            len(out_shape) >= 2
            and isinstance(out_shape[1], int)
            and out_shape[1] > self.num_classes
        )
        log.info(
            "Loaded %s  classes=%d  input=%dx%d  dual_head=%s  EP=%s",
            self.model_path.name, self.num_classes, input_size, input_size,
            self.is_dual_head, self.ep,
        )

    def preprocess(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((self.input_size, self.input_size), Image.Resampling.BICUBIC)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)   # HWC → CHW
        arr = (arr - _MEAN) / _STD
        return np.expand_dims(arr, 0).astype(np.float32)

    def predict_topk(self, image_path: str, k: int = 5) -> Tuple[List[str], List[float], float]:
        """Returns (top_k_part_nums, top_k_probs, latency_ms)."""
        tensor = self.preprocess(image_path)
        t0 = time.perf_counter()
        raw = self.session.run([self.output_name], {self.input_name: tensor})[0][0]
        latency = (time.perf_counter() - t0) * 1000

        # Dual-head: first num_classes logits are parts
        part_logits = raw[:self.num_classes]
        probs = _softmax(part_logits)
        top_idx = probs.argsort()[::-1][:k]

        return (
            [self.idx2part.get(i, f"unk_{i}") for i in top_idx],
            [float(probs[i]) for i in top_idx],
            latency,
        )


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ──────────────────────────────────────────────────────────────────────────────
# Test-set acquisition
# ──────────────────────────────────────────────────────────────────────────────

def _download_rebrickable_images(part_nums: List[str], dest_dir: Path) -> List[Dict]:
    """Download catalog images from Rebrickable CDN. Returns list of {part_num, path}."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for part_num in part_nums:
        dest = dest_dir / f"{part_num}.jpg"
        if dest.exists():
            items.append({"part_num": part_num, "path": str(dest)})
            continue
        for url_tpl in [REBRICKABLE_CDN, REBRICKABLE_FALLBACK]:
            url = url_tpl.format(part_num=part_num)
            try:
                urllib.request.urlretrieve(url, str(dest))
                items.append({"part_num": part_num, "path": str(dest)})
                log.debug("Downloaded %s", part_num)
                break
            except Exception:
                continue
        else:
            log.debug("Could not download part %s — skipping", part_num)
    log.info("Downloaded %d/%d Rebrickable catalog images", len(items), len(part_nums))
    return items


def _load_local_samples(sample_dir: Path) -> List[Dict]:
    """Scan local sample images dir. Part number inferred from filename stem."""
    if not sample_dir.exists():
        return []
    items = []
    for p in sorted(sample_dir.iterdir()):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            items.append({"part_num": p.stem.split(".")[0], "path": str(p)})
    log.info("Found %d local sample images", len(items))
    return items


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def _evaluate(model: ONNXModel, items: List[Dict]) -> Dict:
    """
    Run model against a list of {part_num, path} dicts.
    Returns accuracy metrics and per-image details.
    """
    top1_hits = 0
    top5_hits = 0
    latencies = []
    details = []
    skipped = 0

    for item in items:
        gt_part = item["part_num"]
        path = item["path"]

        # Skip if part num not in model's vocab
        if gt_part not in model.part2idx:
            skipped += 1
            continue

        try:
            preds, probs, lat = model.predict_topk(path, k=5)
        except Exception as e:
            log.debug("Inference error on %s: %s", path, e)
            skipped += 1
            continue

        latencies.append(lat)
        is_top1 = (preds[0] == gt_part)
        is_top5 = (gt_part in preds)

        if is_top1:
            top1_hits += 1
        if is_top5:
            top5_hits += 1

        details.append({
            "part_num": gt_part,
            "top1_pred": preds[0],
            "top1_conf": round(probs[0], 4),
            "top1_correct": is_top1,
            "top5_correct": is_top5,
            "latency_ms": round(lat, 2),
        })

    n = len(details)
    if n == 0:
        return {"top1_acc": None, "top5_acc": None, "n": 0, "skipped": skipped, "details": []}

    return {
        "top1_acc":       round(top1_hits / n, 4),
        "top5_acc":       round(top5_hits / n, 4),
        "n":              n,
        "skipped":        skipped,
        "lat_p50_ms":     round(float(np.percentile(latencies, 50)), 2),
        "lat_p95_ms":     round(float(np.percentile(latencies, 95)), 2),
        "lat_mean_ms":    round(float(np.mean(latencies)), 2),
        "details":        details,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────

def _print_table(results: Dict) -> None:
    COLS = ["Dataset", "Model", "N", "Top-1 Acc", "Top-5 Acc", "Lat p50ms", "Lat p95ms"]
    rows = []
    for dataset_name, dataset_res in results["datasets"].items():
        for model_name, m in dataset_res.items():
            if m["n"] == 0:
                continue
            rows.append([
                dataset_name,
                model_name,
                str(m["n"]),
                f"{m['top1_acc']*100:.1f}%" if m["top1_acc"] is not None else "N/A",
                f"{m['top5_acc']*100:.1f}%" if m["top5_acc"] is not None else "N/A",
                f"{m['lat_p50_ms']:.1f}",
                f"{m['lat_p95_ms']:.1f}",
            ])

    widths = [max(len(r[i]) for r in ([COLS] + rows)) + 2 for i in range(len(COLS))]
    sep = "+" + "+".join("-" * w for w in widths) + "+"
    fmt = "|" + "|".join(f" {{:<{w-1}}}" for w in widths) + "|"

    print("\n" + sep)
    print(fmt.format(*COLS))
    print(sep)
    for row in rows:
        print(fmt.format(*row))
    print(sep)

    # Winner summary
    print("\n── Summary ──────────────────────────────────────────────")
    for dataset_name, dataset_res in results["datasets"].items():
        d = {k: v for k, v in dataset_res.items() if v["n"] > 0 and v["top1_acc"] is not None}
        if len(d) < 2:
            continue
        best = max(d.items(), key=lambda x: x[1]["top1_acc"])
        delta_pct = (best[1]["top1_acc"] - min(v["top1_acc"] for v in d.values())) * 100
        print(f"  {dataset_name}: winner={best[0]}  margin=+{delta_pct:.1f}pp top-1")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dinov2",          required=True, help="DINOv2 ONNX model path")
    parser.add_argument("--effnet",          required=True, help="EfficientNet ONNX model path")
    parser.add_argument("--dinov2-labels",   required=True, help="DINOv2 part_labels.json")
    parser.add_argument("--effnet-labels",   required=True, help="EfficientNet part_labels.json")
    parser.add_argument("--sample-images",   default=None,  help="Local sample images directory")
    parser.add_argument("--download-n",      type=int, default=50,
                        help="Number of Rebrickable images to download (max 50)")
    parser.add_argument("--cache-dir",       default="/tmp/brickscan_bench_images",
                        help="Cache dir for downloaded test images")
    parser.add_argument("--output",          default="benchmark_results.json")
    args = parser.parse_args()

    # ── Load models ──────────────────────────────────────────────────────────
    log.info("Loading DINOv2 model …")
    dinov2 = ONNXModel(args.dinov2, args.dinov2_labels, input_size=518)

    log.info("Loading EfficientNet model …")
    effnet = ONNXModel(args.effnet, args.effnet_labels, input_size=224)

    models = {"dinov2": dinov2, "efficientnet": effnet}

    # ── Acquire test data ─────────────────────────────────────────────────────
    n = min(args.download_n, len(REBRICKABLE_TEST_PARTS))
    cache = Path(args.cache_dir)

    log.info("Preparing Rebrickable catalog test set (%d parts) …", n)
    rebrickable_items = _download_rebrickable_images(REBRICKABLE_TEST_PARTS[:n], cache / "rebrickable")

    local_items = []
    if args.sample_images:
        log.info("Loading local sample images …")
        local_items = _load_local_samples(Path(args.sample_images))

    # Combined set (deduplicated by part_num)
    seen = {i["part_num"] for i in rebrickable_items}
    combined_items = list(rebrickable_items) + [i for i in local_items if i["part_num"] not in seen]
    log.info("Combined test set: %d images", len(combined_items))

    all_datasets = {
        "rebrickable_catalog": rebrickable_items,
        "local_samples":       local_items,
        "combined":            combined_items,
    }

    # ── Run evaluation ────────────────────────────────────────────────────────
    results: Dict[str, Any] = {"datasets": {}}

    for dataset_name, items in all_datasets.items():
        if not items:
            log.info("Skipping %s — no images", dataset_name)
            continue
        log.info("Evaluating on '%s' (%d images) …", dataset_name, len(items))
        results["datasets"][dataset_name] = {}

        for model_name, model in models.items():
            log.info("  → %s", model_name)
            res = _evaluate(model, items)
            results["datasets"][dataset_name][model_name] = res
            if res["n"] > 0:
                log.info(
                    "     top1=%.1f%%  top5=%.1f%%  lat_p50=%.1fms  (n=%d  skipped=%d)",
                    res["top1_acc"] * 100, res["top5_acc"] * 100,
                    res["lat_p50_ms"], res["n"], res["skipped"],
                )

    # ── Print + save ──────────────────────────────────────────────────────────
    _print_table(results)

    output = Path(args.output)
    output.write_text(json.dumps(results, indent=2))
    log.info("Full results written → %s", output)


if __name__ == "__main__":
    main()
