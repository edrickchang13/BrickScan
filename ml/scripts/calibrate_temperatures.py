#!/usr/bin/env python3
"""
Temperature calibration — fit optimal per-source softmax temperatures
so the cascade's reported confidences actually match how often the model
is correct (expected calibration error ↓).

Ladvien's insight ("validation != production") applied here: different
sources in the cascade (Brickognize, Gemini, local-k-NN, merged) all
have different over/under-confidence biases. A single temperature per
source, fit on the feedback eval set, brings them into alignment.

Mechanism:
  For each source S, look at all feedback rows where the top-1 prediction
  came from S. Group their confidence scores by whether the prediction
  was correct (top_correct/partially_correct) or wrong. Fit a scalar
  temperature T_S that minimises NLL of the Bernoulli "top-1 correct?"
  conditional on confidence.

  Calibrated confidence = (raw_conf ** (1/T)) — T>1 softens overconfident
  sources, T<1 sharpens underconfident ones.

Output: ml/data/calibration_temperatures.json
  {
    "brickognize":          1.42,
    "gemini":               0.89,
    "brickognize+gemini":   1.21,
    "contrastive_knn":      1.05,
    "default":              1.00
  }

The backend hybrid_recognition.py optionally reads this file and applies
the temperature per source before the final merge (via a SCAN_USE_CALIBRATION
env flag, disabled by default to preserve baseline behaviour).

Usage:
  ./backend/venv/bin/python3 ml/scripts/calibrate_temperatures.py \\
      --base-url http://localhost:8000 --limit 500
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("calibrate")


def fetch_eval_set(base_url: str, limit: int) -> List[Dict[str, Any]]:
    import requests
    url = f"{base_url.rstrip('/')}/api/local-inventory/feedback/eval-set.json?limit={limit}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def collect_by_source(rows: List[Dict[str, Any]]
                      ) -> Dict[str, List[Tuple[float, bool]]]:
    """
    Extract (confidence, was_correct) pairs per source from the eval rows.

    A row "was correct" if feedback_type indicates the user confirmed the
    top-1 pick was right (top_correct/partially_correct).
    """
    per_src: Dict[str, List[Tuple[float, bool]]] = defaultdict(list)
    for r in rows:
        src = (r.get("source") or "unknown").strip().lower() or "unknown"
        conf = float(r.get("confidence") or 0.0)
        ft = r.get("feedback_type")
        was_correct = ft in ("top_correct", "partially_correct")
        per_src[src].append((conf, was_correct))
    return per_src


def fit_temperature(samples: List[Tuple[float, bool]]) -> float:
    """
    Grid-search the scalar temperature T in [0.25, 4.0] that minimises
    the negative log-likelihood of the "top-1 correct?" Bernoulli given
    the calibrated confidence p' = p^(1/T).

    Not gradient-based (no torch here, keep it lean) — the loss landscape
    is smooth and unimodal, so a fine grid is more than enough.
    """
    if not samples:
        return 1.0
    # Coarse grid first, then refine around the minimum
    def nll(temperature: float) -> float:
        t = max(temperature, 1e-3)
        loss = 0.0
        for (p, y) in samples:
            p = min(max(p, 1e-6), 1 - 1e-6)
            cal = max(min(p ** (1.0 / t), 1 - 1e-6), 1e-6)
            loss -= (math.log(cal) if y else math.log(1 - cal))
        return loss / len(samples)

    # Coarse pass
    best = (None, float("inf"))
    for step in [x * 0.05 for x in range(5, 81)]:   # 0.25 to 4.0, step 0.05
        l = nll(step)
        if l < best[1]:
            best = (step, l)
    # Refine around best
    center = best[0]
    for step in [center + d * 0.01 for d in range(-10, 11)]:
        if step <= 0:
            continue
        l = nll(step)
        if l < best[1]:
            best = (step, l)
    return round(best[0], 3)


def expected_calibration_error(samples: List[Tuple[float, bool]],
                               temperature: float = 1.0,
                               bins: int = 10) -> float:
    """
    ECE: weighted mean of |avg_conf - accuracy| across equal-width conf bins.
    0 = perfectly calibrated. 0.1+ = notably miscalibrated.
    """
    if not samples:
        return 0.0
    t = max(temperature, 1e-3)
    bin_edges = [i / bins for i in range(bins + 1)]
    bucket_confs: List[List[float]] = [[] for _ in range(bins)]
    bucket_correct: List[List[int]] = [[] for _ in range(bins)]
    for (p, y) in samples:
        cal = min(max(p ** (1.0 / t), 0.0), 1.0)
        bi = min(int(cal * bins), bins - 1)
        bucket_confs[bi].append(cal)
        bucket_correct[bi].append(1 if y else 0)
    ece = 0.0
    n = len(samples)
    for bi in range(bins):
        if not bucket_confs[bi]:
            continue
        mean_conf = sum(bucket_confs[bi]) / len(bucket_confs[bi])
        acc = sum(bucket_correct[bi]) / len(bucket_correct[bi])
        ece += (len(bucket_confs[bi]) / n) * abs(mean_conf - acc)
    return ece


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--limit",    type=int, default=500)
    parser.add_argument("--output",   type=Path,
                        default=Path("ml/data/calibration_temperatures.json"))
    parser.add_argument("--min-samples-per-source", type=int, default=10,
                        help="Skip sources with fewer rows (default 10)")
    args = parser.parse_args()

    log.info("Fetching eval set from %s (limit %d)", args.base_url, args.limit)
    rows = fetch_eval_set(args.base_url, args.limit)
    log.info("Got %d rows", len(rows))

    per_src = collect_by_source(rows)
    if not per_src:
        log.warning("No rows → nothing to calibrate. Use the feedback flywheel first.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"default": 1.0}, indent=2))
        return 2

    result: Dict[str, float] = {}
    summary: List[Dict[str, Any]] = []
    for src, samples in sorted(per_src.items(), key=lambda kv: -len(kv[1])):
        if len(samples) < args.min_samples_per_source:
            log.info("Skipping %s — only %d samples", src, len(samples))
            continue
        ece_before = expected_calibration_error(samples, temperature=1.0)
        temp = fit_temperature(samples)
        ece_after = expected_calibration_error(samples, temperature=temp)
        result[src] = temp
        summary.append({
            "source":       src,
            "n":            len(samples),
            "temperature":  temp,
            "ece_before":   round(ece_before, 4),
            "ece_after":    round(ece_after, 4),
            "acc":          round(sum(1 for _, y in samples if y) / len(samples), 3),
        })
        log.info(
            "%-25s n=%-4d T=%.3f  ECE %.3f → %.3f (acc=%.1f%%)",
            src, len(samples), temp, ece_before, ece_after,
            100.0 * sum(1 for _, y in samples if y) / len(samples),
        )

    result.setdefault("default", 1.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    log.info("Saved: %s", args.output)

    sidecar = args.output.with_suffix(".summary.json")
    sidecar.write_text(json.dumps(summary, indent=2))
    log.info("Summary: %s", sidecar)

    return 0


if __name__ == "__main__":
    sys.exit(main())
