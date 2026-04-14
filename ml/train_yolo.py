#!/usr/bin/env python3
"""
train_yolo.py — YOLOv8n fine-tuning for multi-piece LEGO detection (Workstream C).

Replaces the 2×2 grid quadrant hack in the backend with real per-piece bounding boxes.
Single class: "lego_piece".  Exports yolo_lego.onnx for backend inference.

Usage (DGX Spark):
  python train_yolo.py \
    --data   ~/brickscan/yolo_dataset/lego.yaml \
    --output ~/brickscan/models/yolo/ \
    --epochs 100

Requirements:
  pip install ultralytics
"""

import os
import sys
import argparse
import logging
import shutil
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─── Arg parsing ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',        required=True,
                   help='Path to lego.yaml dataset config')
    p.add_argument('--output',      default='~/brickscan/models/yolo/',
                   help='Output directory for weights and ONNX')
    p.add_argument('--model',       default='yolov8n.pt',
                   help='Base YOLOv8 model (n/s/m/l/x, default n=nano)')
    p.add_argument('--epochs',      type=int, default=100)
    p.add_argument('--batch-size',  type=int, default=32)
    p.add_argument('--image-size',  type=int, default=640)
    p.add_argument('--patience',    type=int, default=20,
                   help='Early stopping patience')
    p.add_argument('--device',      default='',
                   help='Device: 0, 0,1, cpu (default: auto)')
    p.add_argument('--workers',     type=int, default=8)
    p.add_argument('--no-train',    action='store_true',
                   help='Skip training, go straight to export')
    p.add_argument('--resume',      action='store_true',
                   help='Resume from last checkpoint')
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
        import ultralytics
        logger.info(f"ultralytics {ultralytics.__version__}")
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data).expanduser()
    if not data_path.exists():
        logger.error(f"Dataset config not found: {data_path}")
        logger.error("Generate it first with: python generate_multipiece_scenes.py ...")
        sys.exit(1)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name  = f'lego_yolo_{timestamp}'

    # ── Train ────────────────────────────────────────────────────────────────
    if not args.no_train:
        logger.info("=" * 60)
        logger.info("  YOLO LEGO Piece Detector Training")
        logger.info("=" * 60)
        logger.info(f"  Model:      {args.model}")
        logger.info(f"  Dataset:    {data_path}")
        logger.info(f"  Epochs:     {args.epochs}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Image size: {args.image_size}")
        logger.info(f"  Output:     {output_dir}")
        logger.info("")

        model = YOLO(args.model)

        train_kwargs = dict(
            data         = str(data_path),
            epochs       = args.epochs,
            batch        = args.batch_size,
            imgsz        = args.image_size,
            patience     = args.patience,
            project      = str(output_dir),
            name         = run_name,
            workers      = args.workers,
            save         = True,
            save_period  = 10,
            exist_ok     = True,
            plots        = True,
            # Augmentation (good defaults for synthetic→real domain gap)
            hsv_h        = 0.015,
            hsv_s        = 0.7,
            hsv_v        = 0.4,
            degrees      = 180.0,   # LEGO pieces can be any orientation
            translate    = 0.1,
            scale        = 0.5,
            flipud       = 0.5,
            fliplr       = 0.5,
            mosaic       = 1.0,
            mixup        = 0.1,
        )

        if args.device:
            train_kwargs['device'] = args.device

        if args.resume:
            # Find latest checkpoint
            ckpt_candidates = sorted(output_dir.glob('**/last.pt'))
            if ckpt_candidates:
                train_kwargs['resume'] = str(ckpt_candidates[-1])
                logger.info(f"Resuming from {train_kwargs['resume']}")

        results = model.train(**train_kwargs)
        logger.info(f"Training complete. Results: {results}")

        # Best weights path
        best_weights = output_dir / run_name / 'weights' / 'best.pt'
    else:
        # Find most recent best.pt
        candidates = sorted(output_dir.glob('**/best.pt'))
        if not candidates:
            logger.error("No best.pt found. Run training first.")
            sys.exit(1)
        best_weights = candidates[-1]
        logger.info(f"Using existing weights: {best_weights}")

    # ── Validate ─────────────────────────────────────────────────────────────
    logger.info("\nRunning validation...")
    val_model = YOLO(str(best_weights))
    val_results = val_model.val(data=str(data_path), imgsz=args.image_size)
    logger.info(f"Validation mAP50: {val_results.box.map50:.4f}")
    logger.info(f"Validation mAP50-95: {val_results.box.map:.4f}")

    # ── Export to ONNX ────────────────────────────────────────────────────────
    logger.info("\nExporting to ONNX...")
    export_model = YOLO(str(best_weights))
    onnx_path = export_model.export(
        format     = 'onnx',
        imgsz      = args.image_size,
        opset      = 12,          # CoreML-compatible opset
        simplify   = True,
        dynamic    = False,       # fixed shape for mobile deployment
        half       = False,
    )
    logger.info(f"ONNX exported to: {onnx_path}")

    # Copy ONNX to output root
    dest_onnx = output_dir / 'yolo_lego.onnx'
    shutil.copy2(str(onnx_path), str(dest_onnx))
    logger.info(f"Copied to: {dest_onnx}")

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  DELIVERABLES")
    logger.info("=" * 60)
    logger.info(f"  Weights:  {best_weights}")
    logger.info(f"  ONNX:     {dest_onnx}")
    logger.info(f"  Next: scp spark:{dest_onnx} ~/Documents/Claude/Projects/Lego/brickscan/backend/models/yolo_lego.onnx")


if __name__ == '__main__':
    main()
