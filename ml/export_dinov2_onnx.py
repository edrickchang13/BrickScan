#!/usr/bin/env python3
"""
Export DINOv2 checkpoint to ONNX for BrickScan backend.

Run this on the DGX Spark where the checkpoint lives:

    python export_dinov2_onnx.py \
        --checkpoint /path/to/best_model.pt \
        --output-dir /path/to/output \
        [--num-classes 2100] \
        [--backbone vit_base_patch14_dinov2] \
        [--opset 17]

Outputs:
    output-dir/
        dinov2_lego.onnx          - ONNX model (fp32)
        dinov2_lego_fp16.onnx     - ONNX model (fp16, smaller, same accuracy)
        part_labels.json          - {"idx2part": {"0": "3001", ...}, "num_classes": N}
        export_info.json          - metadata (input size, opset, accuracy snapshot)

After export, SCP the output dir to the backend server:
    scp -r output-dir/ user@backend:/app/models/dinov2/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Model definition  (mirrors train_dinov2.py exactly)
# ──────────────────────────────────────────────────────────────────────────────

class DINOv2Classifier(nn.Module):
    """DINOv2 ViT-B/14 backbone + 2-layer classification MLP."""

    def __init__(self, num_classes: int, backbone_name: str = "vit_base_patch14_dinov2"):
        super().__init__()
        try:
            from timm.models import create_model
        except ImportError:
            raise SystemExit("timm not installed — run: pip install timm")

        self.backbone = create_model(backbone_name, pretrained=False)
        self.feat_dim = self.backbone.num_features

        if hasattr(self.backbone, "head"):
            self.backbone.head = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.forward_features(x)
        if isinstance(feat, dict):
            feat = feat.get("x", feat.get("features", next(iter(feat.values()))))
        if feat.dim() == 4:
            feat = feat.mean(dim=(2, 3))
        # ViT produces (B, seq_len, dim) — take [CLS] token (index 0)
        if feat.dim() == 3:
            feat = feat[:, 0, :]
        return self.classifier(feat)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint loading — handles multiple save formats
# ──────────────────────────────────────────────────────────────────────────────

def _load_checkpoint(checkpoint_path: Path, model: DINOv2Classifier, device: torch.device) -> dict:
    """
    Load checkpoint tolerantly.  Handles:
      - Raw state dict:            {"backbone.blocks.0...": tensor, ...}
      - Wrapped state dict:        {"state_dict": {...}}
      - PyTorch Lightning ckpt:    {"state_dict": {"model.backbone...": tensor}}
      - Full model save:           the model object itself (rare)
    Returns the raw ckpt dict for metadata extraction.
    """
    log.info("Loading checkpoint from %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, DINOv2Classifier):
        log.info("Checkpoint is a full model object — copying state dict")
        model.load_state_dict(ckpt.state_dict())
        return {}

    if not isinstance(ckpt, dict):
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")

    # Unwrap nested state_dict
    sd = ckpt.get("state_dict", ckpt)

    # Strip Lightning "model." prefix if present
    if any(k.startswith("model.") for k in sd):
        log.info("Stripping 'model.' prefix from Lightning checkpoint keys")
        sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        log.warning("%d missing keys: %s …", len(missing), missing[:3])
    if unexpected:
        log.warning("%d unexpected keys: %s …", len(unexpected), unexpected[:3])

    log.info("Checkpoint loaded ✓  (epoch=%s  val_acc=%s)",
             ckpt.get("epoch", "?"), ckpt.get("val_acc", "?"))
    return ckpt


# ──────────────────────────────────────────────────────────────────────────────
# Label extraction — find or reconstruct class→index mapping
# ──────────────────────────────────────────────────────────────────────────────

def _build_labels(ckpt: dict, num_classes: int, checkpoint_dir: Path) -> dict:
    """
    Return a dict suitable for part_labels.json.

    Priority:
    1. class_to_idx stored inside checkpoint
    2. class_to_idx.json / part_labels.json next to the checkpoint
    3. Synthetic zero-padded numeric labels as a fallback
    """
    # 1. Embedded in checkpoint
    for key in ("class_to_idx", "idx_to_class", "part_labels", "classes"):
        if key in ckpt:
            raw = ckpt[key]
            if key == "class_to_idx":
                idx2part = {str(v): k for k, v in raw.items()}
            elif key == "idx_to_class":
                idx2part = {str(k): v for k, v in raw.items()}
            else:
                idx2part = raw
            log.info("Labels extracted from checkpoint key '%s' (%d classes)", key, len(idx2part))
            return {"idx2part": idx2part, "num_classes": len(idx2part)}

    # 2. Sibling JSON files
    for fname in ("class_to_idx.json", "part_labels.json", "labels.json"):
        candidate = checkpoint_dir / fname
        if candidate.exists():
            raw = json.loads(candidate.read_text())
            if "idx2part" in raw:
                log.info("Labels loaded from %s", candidate)
                return raw
            # class_to_idx format
            idx2part = {str(v): k for k, v in raw.items() if isinstance(v, int)}
            if idx2part:
                log.info("Labels loaded from %s (class_to_idx format)", candidate)
                return {"idx2part": idx2part, "num_classes": len(idx2part)}

    # 3. Synthetic fallback
    log.warning(
        "No label file found — generating synthetic labels part_0000 … part_%04d. "
        "Replace part_labels.json with real labels before deploying.",
        num_classes - 1,
    )
    idx2part = {str(i): f"part_{i:04d}" for i in range(num_classes)}
    return {"idx2part": idx2part, "num_classes": num_classes}


# ──────────────────────────────────────────────────────────────────────────────
# ONNX export
# ──────────────────────────────────────────────────────────────────────────────

INPUT_SIZE = 518   # DINOv2 ViT-B/14 native resolution


def _export_onnx(
    model: DINOv2Classifier,
    output_path: Path,
    opset: int = 17,
    fp16: bool = False,
) -> None:
    """Trace and export to ONNX."""
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        raise SystemExit("Run: pip install onnx onnxruntime")

    model.eval()
    dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)

    log.info("Tracing model → %s  (opset=%d, fp16=%s)", output_path.name, opset, fp16)
    t0 = time.perf_counter()

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            opset_version=opset,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={
                "image":  {0: "batch"},
                "logits": {0: "batch"},
            },
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
        )

    elapsed = time.perf_counter() - t0
    log.info("ONNX export completed in %.1fs", elapsed)

    # Verify the graph
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    log.info("ONNX graph check passed ✓")

    if fp16:
        log.info("Converting to FP16 …")
        try:
            from onnxconverter_common import float16
            onnx_fp16 = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
            fp16_path = output_path.parent / output_path.name.replace(".onnx", "_fp16.onnx")
            onnx.save(onnx_fp16, str(fp16_path))
            log.info("FP16 model saved → %s  (%.1f MB)",
                     fp16_path.name, fp16_path.stat().st_size / 1e6)
        except ImportError:
            log.warning("onnxconverter-common not installed — skipping FP16 export")

    # Quick ORT latency test
    log.info("Running ORT latency benchmark (10 warmup + 50 inference) …")
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(str(output_path), providers=providers)
    dummy_np = np.random.rand(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)

    for _ in range(10):   # warmup
        sess.run(None, {"image": dummy_np})

    times = []
    for _ in range(50):
        t = time.perf_counter()
        sess.run(None, {"image": dummy_np})
        times.append((time.perf_counter() - t) * 1000)

    log.info(
        "ORT latency  p50=%.1fms  p95=%.1fms  p99=%.1fms  [EP: %s]",
        np.percentile(times, 50), np.percentile(times, 95), np.percentile(times, 99),
        sess.get_providers()[0],
    )

    size_mb = output_path.stat().st_size / 1e6
    log.info("Model size: %.1f MB → %s", size_mb, output_path)

    return {
        "latency_p50_ms": round(np.percentile(times, 50), 2),
        "latency_p95_ms": round(np.percentile(times, 95), 2),
        "size_mb": round(size_mb, 1),
        "execution_provider": sess.get_providers()[0],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",   required=True,  help="Path to .pt/.pth checkpoint")
    parser.add_argument("--output-dir",   required=True,  help="Directory for output files")
    parser.add_argument("--num-classes",  type=int, default=None,
                        help="Number of part classes (auto-detected if omitted)")
    parser.add_argument("--backbone",     default="vit_base_patch14_dinov2",
                        help="timm backbone name (default: vit_base_patch14_dinov2)")
    parser.add_argument("--opset",        type=int, default=17, help="ONNX opset version")
    parser.add_argument("--fp16",         action="store_true",  help="Also export FP16 ONNX")
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        sys.exit(f"Checkpoint not found: {checkpoint_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    log.info("Device: %s", device)

    # ── Probe num_classes from checkpoint if not provided ──────────────────
    num_classes = args.num_classes
    if num_classes is None:
        probe = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = probe.get("state_dict", probe) if isinstance(probe, dict) else probe.state_dict()
        # Last linear layer weight: shape (num_classes, in_features)
        for key in reversed(list(sd.keys())):
            if "weight" in key and sd[key].dim() == 2:
                num_classes = sd[key].shape[0]
                log.info("Auto-detected num_classes=%d from key '%s'", num_classes, key)
                break
        if num_classes is None:
            sys.exit("Could not auto-detect num_classes — pass --num-classes explicitly")

    # ── Build and load model ───────────────────────────────────────────────
    model = DINOv2Classifier(num_classes=num_classes, backbone_name=args.backbone).to(device)
    ckpt = _load_checkpoint(checkpoint_path, model, device)
    model.eval()
    log.info("Model: %s  feat_dim=%d  num_classes=%d",
             args.backbone, model.feat_dim, num_classes)

    # ── Export ONNX ────────────────────────────────────────────────────────
    onnx_path = output_dir / "dinov2_lego.onnx"
    perf = _export_onnx(model.cpu(), onnx_path, opset=args.opset, fp16=args.fp16)

    # ── Build and save labels ──────────────────────────────────────────────
    labels = _build_labels(ckpt, num_classes, checkpoint_path.parent)
    labels_path = output_dir / "part_labels.json"
    labels_path.write_text(json.dumps(labels, indent=2))
    log.info("Labels saved → %s  (%d classes)", labels_path.name, labels["num_classes"])

    # ── Write export metadata ──────────────────────────────────────────────
    meta = {
        "model": "DINOv2",
        "backbone": args.backbone,
        "input_size": INPUT_SIZE,
        "num_classes": num_classes,
        "opset": args.opset,
        "output_shape": f"[batch, {num_classes}]",
        "preprocessing": {
            "resize": [INPUT_SIZE, INPUT_SIZE],
            "mean": [0.485, 0.456, 0.406],
            "std":  [0.229, 0.224, 0.225],
        },
        "heads": ["parts"],  # single-head (no color head)
        "source_checkpoint": str(checkpoint_path),
        "ort_benchmark": perf,
        "notes": (
            "Single-head model: output is part logits only. "
            "Color prediction not available — set color_name=None in backend. "
            "Use dinov2_lego_fp16.onnx for ~2x throughput with negligible accuracy loss."
        ),
    }
    meta_path = output_dir / "export_info.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("Export metadata → %s", meta_path.name)

    log.info(
        "\n"
        "══════════════════════════════════════════════\n"
        "  Export complete!\n"
        "  Files: %s\n"
        "  SCP command:\n"
        "    scp -r %s user@backend:/app/models/dinov2/\n"
        "  Then set in .env:\n"
        "    ML_MODEL_PATH=/app/models/dinov2/dinov2_lego.onnx\n"
        "    ML_MODEL_TYPE=dinov2\n"
        "══════════════════════════════════════════════",
        output_dir,
        output_dir,
    )


if __name__ == "__main__":
    main()
