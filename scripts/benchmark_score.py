#!/usr/bin/env python3
"""
Score a head-to-head Brickify/Brickit/BrickScan benchmark run.

Inputs (relative to repo root unless --assets-dir overrides):
    docs/benchmark_assets/ground_truth.csv
    docs/benchmark_assets/results_brickscan.csv
    docs/benchmark_assets/results_brickit.csv
    docs/benchmark_assets/results_brickify.csv

Outputs:
    docs/benchmark_assets/results_summary.csv   — flat per-app metrics
    docs/benchmark_assets/results_summary.md    — markdown table for the marketing site

Usage:
    python3 scripts/benchmark_score.py
    python3 scripts/benchmark_score.py --assets-dir /path/to/elsewhere
    python3 scripts/benchmark_score.py --apps brickscan brickit brickify

CSV schemas — see docs/COMPETITIVE_BENCHMARK.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ASSETS = REPO_ROOT / "docs" / "benchmark_assets"


# ─── Data structures ───────────────────────────────────────────────────────

@dataclass
class GroundTruthRow:
    photo_id: str
    brick_index: int
    part_num: str
    part_name: str
    color_id: Optional[int]
    color_name: str
    is_worn: bool = False


@dataclass
class ResultRow:
    photo_id: str
    brick_index: int
    predicted_part_num: str        # "" or "MISS" when no detection
    predicted_color_name: str
    latency_ms: int


@dataclass
class AppMetrics:
    app: str
    total_samples: int = 0
    matched_samples: int = 0          # gt rows the app produced a prediction for (not MISS)
    extra_predictions: int = 0
    crashed_photos: int = 0
    top1_part_correct: int = 0
    top1_color_correct: int = 0
    top1_joint_correct: int = 0
    latencies_ms: List[int] = field(default_factory=list)
    confusions: List[Tuple[str, str]] = field(default_factory=list)
    per_condition: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: {
        "n": 0, "part": 0, "color": 0, "joint": 0, "miss": 0,
    }))

    @property
    def top1_part_accuracy(self) -> float:
        return self.top1_part_correct / self.total_samples if self.total_samples else 0.0

    @property
    def top1_color_accuracy(self) -> float:
        return self.top1_color_correct / self.total_samples if self.total_samples else 0.0

    @property
    def top1_joint_accuracy(self) -> float:
        return self.top1_joint_correct / self.total_samples if self.total_samples else 0.0

    @property
    def recall(self) -> float:
        return self.matched_samples / self.total_samples if self.total_samples else 0.0

    @property
    def precision(self) -> float:
        emitted = self.matched_samples + self.extra_predictions
        return self.top1_part_correct / emitted if emitted else 0.0

    @property
    def mean_latency_ms(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def median_latency_ms(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lat = sorted(self.latencies_ms)
        idx = max(0, int(0.95 * (len(sorted_lat) - 1)))
        return float(sorted_lat[idx])


# ─── Normalisation ─────────────────────────────────────────────────────────

# Some apps return "Bright Red" where Rebrickable says "Red" — collapse
# common aliases so we don't punish style differences.
COLOR_ALIASES = {
    "bright red": "red",
    "bright blue": "blue",
    "trans clear": "trans-clear",
    "transparent clear": "trans-clear",
    "white-glow": "glow in dark white",
    "lt gray": "light gray",
    "lt grey": "light gray",
    "light grey": "light gray",
    "dk gray": "dark gray",
    "dk grey": "dark gray",
    "dark grey": "dark gray",
}


def normalize_color(name: str) -> str:
    n = (name or "").strip().lower()
    return COLOR_ALIASES.get(n, n)


_VARIANT_RE = re.compile(r"(pr\d+[a-z]?|px\d+[a-z]?|[a-z])$", re.IGNORECASE)


def normalize_part(part_num: str) -> str:
    """Strip mold/variant suffixes (3001a → 3001, 3001pr0042 → 3001)
    so different apps' suffixing styles don't appear as disagreements."""
    p = (part_num or "").strip().lower()
    if not p:
        return p
    # Strip print/mold suffixes iteratively
    while True:
        m = _VARIANT_RE.search(p)
        if not m or m.group(0) == p:
            break
        # Don't strip if doing so leaves an empty stem
        candidate = p[: m.start()]
        if not candidate:
            break
        p = candidate
    return p


# ─── IO ────────────────────────────────────────────────────────────────────

def read_ground_truth(path: Path) -> Dict[Tuple[str, int], GroundTruthRow]:
    out: Dict[Tuple[str, int], GroundTruthRow] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["photo_id"].strip(), int(row["brick_index"]))
            out[key] = GroundTruthRow(
                photo_id=key[0],
                brick_index=key[1],
                part_num=row["part_num"].strip(),
                part_name=row.get("part_name", "").strip(),
                color_id=_int_or_none(row.get("color_id")),
                color_name=row.get("color_name", "").strip(),
                is_worn=row.get("is_worn", "").strip().lower() in ("true", "1", "yes"),
            )
    return out


def read_results(path: Path) -> List[ResultRow]:
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.append(ResultRow(
                photo_id=row["photo_id"].strip(),
                brick_index=int(row["brick_index"]),
                predicted_part_num=row.get("predicted_part_num", "").strip(),
                predicted_color_name=row.get("predicted_color_name", "").strip(),
                latency_ms=int(row.get("latency_ms", "0") or 0),
            ))
    return out


def _int_or_none(s):
    if s is None or s == "":
        return None
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


# ─── Scoring ───────────────────────────────────────────────────────────────

def score(app: str,
          ground_truth: Dict[Tuple[str, int], GroundTruthRow],
          results: List[ResultRow]) -> AppMetrics:
    metrics = AppMetrics(app=app)
    metrics.total_samples = len(ground_truth)

    seen_keys = set()
    for r in results:
        key = (r.photo_id, r.brick_index)
        if key not in ground_truth:
            metrics.extra_predictions += 1
            continue
        seen_keys.add(key)
        gt = ground_truth[key]
        cond = r.photo_id
        cond_bucket = metrics.per_condition[cond]
        cond_bucket["n"] += 1
        if r.predicted_part_num.upper() == "MISS" or not r.predicted_part_num:
            cond_bucket["miss"] += 1
            continue
        metrics.matched_samples += 1
        metrics.latencies_ms.append(r.latency_ms)

        part_match = normalize_part(r.predicted_part_num) == normalize_part(gt.part_num)
        color_match = normalize_color(r.predicted_color_name) == normalize_color(gt.color_name)
        if part_match:
            metrics.top1_part_correct += 1
            cond_bucket["part"] += 1
        else:
            metrics.confusions.append((gt.part_num, r.predicted_part_num))
        if color_match:
            metrics.top1_color_correct += 1
            cond_bucket["color"] += 1
        if part_match and color_match:
            metrics.top1_joint_correct += 1
            cond_bucket["joint"] += 1

    # Bricks the app didn't even attempt
    for key in ground_truth:
        if key not in seen_keys:
            cond = key[0]
            metrics.per_condition[cond]["miss"] += 1
            metrics.per_condition[cond]["n"] += 1
    return metrics


# ─── Output ────────────────────────────────────────────────────────────────

def write_summary_csv(path: Path, metrics: List[AppMetrics]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "app", "n", "top1_part", "top1_color", "top1_joint",
            "recall", "precision", "mean_ms", "median_ms", "p95_ms",
        ])
        for m in metrics:
            w.writerow([
                m.app, m.total_samples,
                f"{m.top1_part_accuracy:.3f}",
                f"{m.top1_color_accuracy:.3f}",
                f"{m.top1_joint_accuracy:.3f}",
                f"{m.recall:.3f}",
                f"{m.precision:.3f}",
                f"{m.mean_latency_ms:.0f}",
                f"{m.median_latency_ms:.0f}",
                f"{m.p95_latency_ms:.0f}",
            ])


def write_summary_md(path: Path, metrics: List[AppMetrics]) -> None:
    lines = ["# Benchmark — head-to-head results\n"]
    lines.append("| App | n | Top-1 part | Top-1 colour | Joint (part+colour) | Recall | Precision | Mean ms | Median ms | p95 ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in metrics:
        lines.append(
            f"| **{m.app}** | {m.total_samples} | "
            f"{m.top1_part_accuracy*100:.1f}% | "
            f"{m.top1_color_accuracy*100:.1f}% | "
            f"{m.top1_joint_accuracy*100:.1f}% | "
            f"{m.recall*100:.1f}% | "
            f"{m.precision*100:.1f}% | "
            f"{m.mean_latency_ms:.0f} | "
            f"{m.median_latency_ms:.0f} | "
            f"{m.p95_latency_ms:.0f} |"
        )
    lines.append("\n## Top confusions per app\n")
    for m in metrics:
        if not m.confusions:
            continue
        lines.append(f"### {m.app}")
        cnt = Counter(m.confusions).most_common(10)
        lines.append("| Ground truth → predicted | count |")
        lines.append("|---|---:|")
        for (gt, pred), n in cnt:
            lines.append(f"| `{gt}` → `{pred}` | {n} |")
        lines.append("")
    lines.append("\n## Per-condition breakdown\n")
    for m in metrics:
        if not m.per_condition:
            continue
        lines.append(f"### {m.app}")
        lines.append("| Condition | n | Top-1 part | Top-1 colour | Miss rate |")
        lines.append("|---|---:|---:|---:|---:|")
        for cond, b in sorted(m.per_condition.items()):
            n = b["n"] or 1
            lines.append(
                f"| {cond} | {b['n']} | "
                f"{b['part']/n*100:.1f}% | "
                f"{b['color']/n*100:.1f}% | "
                f"{b['miss']/n*100:.1f}% |"
            )
        lines.append("")
    path.write_text("\n".join(lines))


# ─── Entry point ───────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS)
    ap.add_argument("--apps", nargs="+", default=["brickscan", "brickit", "brickify"])
    args = ap.parse_args()

    gt_path = args.assets_dir / "ground_truth.csv"
    if not gt_path.exists():
        print(f"ERROR: ground_truth.csv missing at {gt_path}", file=sys.stderr)
        print(
            "Create it per docs/COMPETITIVE_BENCHMARK.md before running this script.",
            file=sys.stderr,
        )
        return 1

    gt = read_ground_truth(gt_path)
    print(f"Loaded {len(gt)} ground-truth samples from {gt_path}")

    metrics: List[AppMetrics] = []
    for app in args.apps:
        results_path = args.assets_dir / f"results_{app}.csv"
        if not results_path.exists():
            print(f"WARN: missing {results_path} — skipping {app}", file=sys.stderr)
            continue
        results = read_results(results_path)
        m = score(app, gt, results)
        metrics.append(m)
        print(
            f"  {app:>10}: n={m.total_samples}  "
            f"part={m.top1_part_accuracy*100:5.1f}%  "
            f"colour={m.top1_color_accuracy*100:5.1f}%  "
            f"joint={m.top1_joint_accuracy*100:5.1f}%  "
            f"miss={(1 - m.recall)*100:5.1f}%  "
            f"mean_ms={m.mean_latency_ms:.0f}"
        )

    if not metrics:
        print("No app result files found — nothing to score.", file=sys.stderr)
        return 1

    csv_out = args.assets_dir / "results_summary.csv"
    md_out = args.assets_dir / "results_summary.md"
    write_summary_csv(csv_out, metrics)
    write_summary_md(md_out, metrics)
    print(f"\nWrote {csv_out}")
    print(f"Wrote {md_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
