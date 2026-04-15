#!/usr/bin/env python3
"""
eval_against_feedback.py — measure the live cascade's accuracy against
user-confirmed ground truth captured by the feedback flywheel.

Flow:
  1. Fetch GET /api/local-inventory/feedback/eval-set.json
  2. For each row, read the stored image from `image_path`, re-encode as
     base64, POST to /api/local-inventory/scan
  3. Compare the returned top-1 and top-3 predictions against correct_part_num
  4. Report overall accuracy, accuracy-by-source (which model won slot 1),
     and frequent confusion pairs

Results are written to ml/data/eval_results/<timestamp>_<label>.json so
runs can be compared across env-var configurations (A/B testing the
SCAN_GROUNDED_GEMINI / SCAN_COLLAPSE_VARIANTS / SCAN_COLOR_RERANK flags).

Usage:
  ./backend/venv/bin/python3 ml/scripts/eval_against_feedback.py \
      --base-url http://localhost:8000 \
      --config-label baseline \
      --limit 200
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: ./backend/venv/bin/pip install requests")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval")


def _normalise(p: str) -> str:
    """Normalise a part_num for equality comparison (lowercase, strip leading zeros)."""
    return (p or "").strip().lower().lstrip("0") or "0"


def fetch_eval_set(base_url: str, limit: int) -> List[Dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/api/local-inventory/feedback/eval-set.json?limit={limit}"
    log.info("Fetching eval set: %s", url)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    rows = r.json()
    log.info("Got %d eval rows", len(rows))
    return rows


def scan_one(base_url: str, image_path: Path) -> Optional[Dict[str, Any]]:
    """POST the image to /api/local-inventory/scan and return the response."""
    try:
        img_bytes = image_path.read_bytes()
    except Exception as e:
        log.warning("Can't read %s: %s", image_path, e)
        return None
    b64 = base64.b64encode(img_bytes).decode("ascii")
    try:
        r = requests.post(
            f"{base_url.rstrip('/')}/api/local-inventory/scan",
            json={"image_base64": b64},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        log.warning("Scan failed for %s: %s", image_path.name, e)
        return None


def evaluate(rows: List[Dict[str, Any]], base_url: str) -> Dict[str, Any]:
    total = 0
    top1_correct = 0
    top3_correct = 0
    by_source_total: Counter = Counter()
    by_source_top1: Counter = Counter()
    confusions: Counter = Counter()

    for i, row in enumerate(rows):
        truth = _normalise(row.get("correct_part_num", ""))
        img_path_str = row.get("image_path")
        if not img_path_str or not truth:
            continue
        img_path = Path(img_path_str)
        if not img_path.exists():
            log.debug("Skipping missing image: %s", img_path)
            continue

        resp = scan_one(base_url, img_path)
        if not resp or not resp.get("predictions"):
            continue

        preds = resp["predictions"]
        top1 = _normalise(preds[0].get("part_num", ""))
        top3_parts = {_normalise(p.get("part_num", "")) for p in preds[:3]}
        src = preds[0].get("source", "unknown")

        total += 1
        by_source_total[src] += 1
        if top1 == truth:
            top1_correct += 1
            by_source_top1[src] += 1
        if truth in top3_parts:
            top3_correct += 1
        if top1 != truth:
            confusions[(top1, truth)] += 1

        if (i + 1) % 10 == 0:
            log.info("Progress: %d / %d — top1 %.1f%%", i + 1, len(rows),
                     100.0 * top1_correct / max(total, 1))

    results = {
        "total_evaluated": total,
        "top1_accuracy": round(top1_correct / total, 4) if total else 0.0,
        "top3_accuracy": round(top3_correct / total, 4) if total else 0.0,
        "by_source": {
            src: {
                "total": by_source_total[src],
                "top1_correct": by_source_top1[src],
                "top1_accuracy": round(by_source_top1[src] / by_source_total[src], 4)
                                  if by_source_total[src] else 0.0,
            }
            for src in by_source_total
        },
        "top_confusions": [
            {"predicted": p, "correct": c, "count": n}
            for (p, c), n in confusions.most_common(20)
            if n >= 3
        ],
    }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="Backend URL (default http://localhost:8000)")
    parser.add_argument("--limit", type=int, default=200,
                        help="Max rows to evaluate (default 200)")
    parser.add_argument("--config-label", default="baseline",
                        help="Tag for this run — used in the output filename so runs compare cleanly")
    parser.add_argument("--output-dir", type=Path, default=Path("ml/data/eval_results"),
                        help="Where to write the results JSON (default ml/data/eval_results)")
    args = parser.parse_args()

    rows = fetch_eval_set(args.base_url, args.limit)
    if not rows:
        log.error("Eval set is empty — nothing to evaluate. Use the feedback flywheel to generate ground-truth first.")
        return 2

    log.info("Evaluating %d rows against %s", len(rows), args.base_url)
    results = evaluate(rows, args.base_url)
    results["config_label"] = args.config_label
    results["base_url"] = args.base_url
    results["generated_at"] = datetime.now(timezone.utc).isoformat()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = args.output_dir / f"{stamp}_{args.config_label}.json"
    out_path.write_text(json.dumps(results, indent=2))

    log.info("=" * 60)
    log.info("Config: %s", args.config_label)
    log.info("Rows:   %d", results["total_evaluated"])
    log.info("Top-1:  %.1f%%", 100.0 * results["top1_accuracy"])
    log.info("Top-3:  %.1f%%", 100.0 * results["top3_accuracy"])
    for src, stats in results["by_source"].items():
        log.info("  %-25s n=%-4d top1=%.1f%%", src, stats["total"], 100.0 * stats["top1_accuracy"])
    if results["top_confusions"]:
        log.info("Top confusion pairs (>=3):")
        for c in results["top_confusions"][:10]:
            log.info("  %s -> %s (x%d)", c["predicted"], c["correct"], c["count"])
    log.info("=" * 60)
    log.info("Saved: %s", out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
