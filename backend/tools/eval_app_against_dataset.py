#!/usr/bin/env python3
"""
Eval the running BrickScan backend against a labeled LEGO dataset.

What this does:
  1. Walk a labeled image tree (one of the supported layouts below).
  2. POST each image to the backend's /api/local-inventory/scan endpoint.
  3. Compare the top-1, top-3, top-5 predictions to ground truth.
  4. Print per-class accuracy + a confusion matrix of the top-N misses.
  5. Optionally save a JSON report.

Use this to answer:
  • Is the model accurate on YOUR domain (the dataset you point it at)?
  • Are colour predictions correct alongside shape?
  • Where does the model fail systematically?

Supported layouts (auto-detected):
  A) ImageFolder by part_num:
       <root>/<part_num>/<image>.jpg
     (e.g. Nature 2023 real-photo dataset, our labeled_by_part dir)

  B) ImageFolder by semantic name (e.g. pacogarciam3 sorting):
       <root>/<semantic>/<image>.jpg
     where <semantic> is a name like "Brick_2x4". Pass --semantic-map
     to provide a {semantic_name: part_num} JSON.

  C) Roboflow YOLO format (e.g. Hex:Lego, lego-s6zjh):
       <root>/data.yaml
       <root>/{train,valid,test}/images/*.jpg
       <root>/{train,valid,test}/labels/*.txt
     The script extracts the largest bbox, crops it, and uses the YOLO class
     name (parsed via shape regex) as ground truth.

Usage examples:
    # Hex:Lego (Roboflow YOLO format)
    python eval_app_against_dataset.py \\
        --layout yolo \\
        --root /path/to/hex-lego_v3 \\
        --backend http://localhost:8000 \\
        --max-per-class 20

    # Nature 2023 (part_num folders)
    python eval_app_against_dataset.py \\
        --layout part_num \\
        --root /path/to/nature_2023_real \\
        --max-per-class 30

    # pacog Sorting (semantic folders)
    python eval_app_against_dataset.py \\
        --layout semantic \\
        --root "/path/to/lego-brick-sorting-image-recognition/cropped images" \\
        --semantic-map ./pacog_to_partnum.json
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

logger = logging.getLogger("eval_app")
SCAN_PATH = "/api/local-inventory/scan"


# ─── Hex:Lego label parser (shared with build_phase3_dataset.py) ──────────────

def hex_label_to_part(label: str) -> Optional[str]:
    """Parse a Hex:Lego YOLO label like '1x1_red' or 'plate_2x4_blue' to a
    canonical Rebrickable part_num. Returns None for unknown shapes."""
    m = re.match(r"^(plate_)?(\d+)x(\d+)", label.lower())
    if not m: return None
    is_plate = m.group(1) is not None
    a, b = int(m.group(2)), int(m.group(3))
    if a > b: a, b = b, a
    key = f"{'plate_' if is_plate else ''}{a}x{b}"
    return {
        "1x1": "3005", "1x2": "3004", "1x3": "3622", "1x4": "3010",
        "2x2": "3003", "2x3": "3002", "2x4": "3001",
        "plate_1x1": "3024", "plate_1x2": "3023", "plate_1x3": "3623",
        "plate_1x4": "3710", "plate_2x2": "3022", "plate_2x3": "3021",
        "plate_2x4": "3020",
    }.get(key)


def hex_label_to_color(label: str) -> Optional[str]:
    """Strip shape, return the colour token."""
    parts = label.lower().split("_")
    return parts[-1] if len(parts) >= 2 else None


# ─── Sample collectors per layout ────────────────────────────────────────────

def collect_part_num(root: Path, max_per_class: int) -> List[Tuple[Path, str, Optional[str]]]:
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir(): continue
        pn = d.name
        imgs = [p for p in d.glob("**/*")
                if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.is_file()]
        random.shuffle(imgs)
        for img in imgs[:max_per_class]:
            out.append((img, pn, None))    # (path, gt_part_num, gt_color_name)
    return out


def collect_semantic(root: Path, sem_map: Dict[str, str], max_per_class: int) -> List[Tuple[Path, str, Optional[str]]]:
    out = []
    for sem, pn in sem_map.items():
        for d in root.glob(f"**/{sem}"):
            if not d.is_dir(): continue
            imgs = [p for p in d.glob("**/*")
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.is_file()]
            random.shuffle(imgs)
            for img in imgs[:max_per_class]:
                out.append((img, pn, None))
            break
    return out


def collect_yolo(root: Path, max_per_class: int, crop_dir: Path) -> List[Tuple[Path, str, Optional[str]]]:
    """Walk a YOLO-formatted dataset, crop the largest bbox per image."""
    try:
        import yaml
        from PIL import Image
    except ImportError:
        logger.error("yaml + PIL required for --layout yolo — pip install pyyaml Pillow")
        return []

    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        logger.error("Missing %s", yaml_path)
        return []
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]

    crop_dir.mkdir(parents=True, exist_ok=True)
    out = []
    counts: Counter = Counter()

    for split in ("train", "valid", "test"):
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        if not img_dir.exists(): continue
        for img_path in sorted(img_dir.glob("*.*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lbl = lbl_dir / (img_path.stem + ".txt")
            if not lbl.exists(): continue
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            W, H = img.size
            # Pick the largest bbox (most prominent brick)
            best: Optional[tuple] = None
            with open(lbl) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5: continue
                    c = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                    area = w * h
                    if best is None or area > best[5]:
                        best = (c, cx, cy, w, h, area)
            if best is None: continue
            c, cx, cy, w, h, _ = best
            if c >= len(names): continue
            label = names[c] if isinstance(names[c], str) else str(names[c])
            pn = hex_label_to_part(label)
            if pn is None: continue
            color = hex_label_to_color(label)
            if counts[pn] >= max_per_class: continue
            counts[pn] += 1

            x1 = max(0, int((cx - w/2) * W))
            y1 = max(0, int((cy - h/2) * H))
            x2 = min(W, int((cx + w/2) * W))
            y2 = min(H, int((cy + h/2) * H))
            if x2 - x1 < 30 or y2 - y1 < 30: continue
            crop = img.crop((x1, y1, x2, y2))
            out_path = crop_dir / f"{label}_{img_path.stem}.jpg"
            crop.save(out_path, quality=92)
            out.append((out_path, pn, color))
    return out


# ─── Backend client ──────────────────────────────────────────────────────────

def post_scan(image_path: Path, backend: str, timeout: int = 60) -> Optional[Dict[str, Any]]:
    """POST a single image to /api/local-inventory/scan. Returns the parsed
    response dict, or None on failure. Uses urllib so we don't add a httpx
    dep just for one tool."""
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
    except Exception as e:
        logger.warning("Could not read %s: %s", image_path, e)
        return None
    body = json.dumps({"image_base64": b64}).encode("utf-8")
    req = Request(
        urljoin(backend, SCAN_PATH),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
        logger.warning("Scan failed for %s: %s", image_path.name, e)
        return None


# ─── Scoring ────────────────────────────────────────────────────────────────

def score(samples: List[Tuple[Path, str, Optional[str]]],
          backend: str,
          concurrency: int = 4) -> Dict[str, Any]:
    """Run the eval and return aggregate metrics."""
    n = len(samples)
    top1 = top3 = top5 = color_match = 0
    misses_by_pn: Dict[str, Counter] = defaultdict(Counter)
    n_with_color_gt = 0
    t0 = time.time()
    detail: List[Dict[str, Any]] = []
    completed = 0
    lock_count = [0]

    def task(args):
        path, gt_pn, gt_color = args
        resp = post_scan(path, backend)
        return path, gt_pn, gt_color, resp

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for path, gt_pn, gt_color, resp in pool.map(task, samples):
            completed += 1
            if not resp:
                detail.append({
                    "image": str(path), "gt_part": gt_pn, "gt_color": gt_color,
                    "predictions": [], "error": "no_response",
                })
                continue
            preds = resp.get("predictions") or []
            top_parts = [p.get("part_num") for p in preds[:5]]
            top_colors = [(p.get("color_name") or "").lower() for p in preds[:5]]

            if top_parts and top_parts[0] == gt_pn: top1 += 1
            if gt_pn in top_parts[:3]: top3 += 1
            if gt_pn in top_parts[:5]: top5 += 1

            if gt_color:
                n_with_color_gt += 1
                # ground truth like "red"; prediction like "Red" — case-insensitive
                # substring check is the loosest sane comparison
                if top_colors and gt_color in top_colors[0]:
                    color_match += 1

            if top_parts and top_parts[0] != gt_pn:
                misses_by_pn[gt_pn][top_parts[0]] += 1

            detail.append({
                "image": str(path),
                "gt_part": gt_pn,
                "gt_color": gt_color,
                "top1_part": top_parts[0] if top_parts else None,
                "top1_color": top_colors[0] if top_colors else None,
                "top1_conf": preds[0].get("confidence") if preds else None,
                "top1_source": preds[0].get("source") if preds else None,
                "top5_parts": top_parts,
            })

            if completed % 25 == 0 or completed == n:
                rate = completed / max(time.time() - t0, 1e-6)
                logger.info(
                    "  [%d/%d] top1=%.1f%% top3=%.1f%% top5=%.1f%% (%.1f img/s)",
                    completed, n,
                    100*top1/completed, 100*top3/completed,
                    100*top5/completed, rate,
                )

    dt = time.time() - t0
    return {
        "total": n,
        "top1": top1 / n if n else 0,
        "top3": top3 / n if n else 0,
        "top5": top5 / n if n else 0,
        "color_top1": color_match / n_with_color_gt if n_with_color_gt else None,
        "n_with_color_gt": n_with_color_gt,
        "wall_s": dt,
        "rate_img_per_s": n / dt if dt > 0 else 0,
        "top_misses": {k: v.most_common(5) for k, v in misses_by_pn.items()},
        "detail": detail,
    }


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--layout", required=True, choices=("part_num", "semantic", "yolo"))
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--backend", default="http://localhost:8000")
    ap.add_argument("--max-per-class", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=4,
                    help="Parallel scan requests (cap: backend handles ~4-8 concurrent).")
    ap.add_argument("--semantic-map", type=Path,
                    help="JSON file {semantic_name: part_num} for --layout semantic")
    ap.add_argument("--crop-dir", type=Path, default=Path("/tmp/eval_app_crops"),
                    help="Where to cache YOLO bbox crops")
    ap.add_argument("--out", type=Path, default=Path("./eval_app_report.json"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    random.seed(args.seed)

    logger.info("Backend: %s", args.backend)
    # Fail fast if backend isn't reachable
    try:
        with urlopen(urljoin(args.backend, "/health"), timeout=5) as r:
            assert json.loads(r.read())["status"] == "ok"
    except Exception as e:
        logger.error("Backend unreachable at %s: %s", args.backend, e)
        return 2

    if args.layout == "part_num":
        samples = collect_part_num(args.root, args.max_per_class)
    elif args.layout == "semantic":
        if not args.semantic_map:
            ap.error("--semantic-map is required for --layout semantic")
        sem_map = json.loads(args.semantic_map.read_text())
        samples = collect_semantic(args.root, sem_map, args.max_per_class)
    else:  # yolo
        samples = collect_yolo(args.root, args.max_per_class, args.crop_dir)

    if not samples:
        logger.error("No samples collected — check --root and --layout")
        return 1
    logger.info("Collected %d samples across %d unique part_nums",
                len(samples), len({s[1] for s in samples}))

    metrics = score(samples, args.backend, concurrency=args.concurrency)

    print()
    print("=" * 60)
    print(f"  total          : {metrics['total']}")
    print(f"  top-1 accuracy : {metrics['top1']*100:5.2f}%")
    print(f"  top-3 accuracy : {metrics['top3']*100:5.2f}%")
    print(f"  top-5 accuracy : {metrics['top5']*100:5.2f}%")
    if metrics["color_top1"] is not None:
        print(f"  color top-1    : {metrics['color_top1']*100:5.2f}%  "
              f"(n={metrics['n_with_color_gt']})")
    print(f"  wall time      : {metrics['wall_s']:.1f}s "
          f"({metrics['rate_img_per_s']:.1f} img/s)")
    print("=" * 60)

    if metrics["top_misses"]:
        print("\n  top miss confusion (gt → predicted) — most common 10 classes:")
        sorted_misses = sorted(metrics["top_misses"].items(),
                                key=lambda kv: -sum(c for _, c in kv[1]))[:10]
        for gt, m in sorted_misses:
            wrong = ", ".join(f"{p}×{c}" for p, c in m)
            print(f"    {gt:<10} → {wrong}")

    args.out.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nFull report: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
