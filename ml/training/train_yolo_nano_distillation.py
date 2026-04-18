"""YOLOv8-L → YOLOv8-n distillation on DGX Spark.

Stage 1b of continuous-scan Phase 5 — if int8 quantization (Stage 1a, 58 MB)
isn't small enough to hit the doc's ≤10 MB install-delta target, distil the
large teacher down to a nano student (~6 MB).

NOT a ready-to-run job — Ultralytics YOLO's Python API doesn't expose a
distillation hook by default, so this script monkey-patches `BaseTrainer.
loss_fn` to inject a KD term (soft class targets + bbox cwh regression from
the teacher). Kick the tires on a small subset first.

Usage on Spark:

    ssh spark
    cd ~/brickscan/ml/training
    python train_yolo_nano_distillation.py \
        --teacher ~/brickscan/ml/weights/yolov8l_lego.pt \
        --data   ~/brickscan/ml/data/roboflow_hex_lego.yaml \
        --epochs 80 \
        --kd-alpha 0.6

Do NOT run before the teacher checkpoint is on the box — verify its path
first. Expected runtime on a single B100: ~6h for 80 epochs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss


class DistillationLoss(v8DetectionLoss):
    """v8DetectionLoss + KD term from the teacher's logits.

    We add:
      L = L_ultralytics + alpha * KL(softmax(student_cls) || softmax(teacher_cls/T))

    The bbox branch is NOT distilled separately — the teacher's label-set
    ground-truth already implicitly shapes it. Emphasising bbox matching on
    top of the classification KL tends to destabilise nano convergence.
    """
    def __init__(self, model, teacher, alpha: float = 0.6, temperature: float = 4.0):
        super().__init__(model)
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.alpha = alpha
        self.T = temperature

    def __call__(self, preds, batch):
        base_loss, loss_items = super().__call__(preds, batch)
        if self.alpha <= 0:
            return base_loss, loss_items
        imgs = batch["img"]
        with torch.no_grad():
            t_out = self.teacher(imgs)
        kd = _kd_cls_kl(preds, t_out, self.T)
        return base_loss + self.alpha * kd, loss_items


def _kd_cls_kl(student_out, teacher_out, T: float) -> torch.Tensor:
    """KL divergence over class logits at each anchor.

    Both preds are (B, 4+C, N). We softmax along C with temperature T
    and compute KL(student || teacher) averaged over batch*anchors.
    """
    def split(o):
        if isinstance(o, (list, tuple)):
            o = o[0]
        return o[:, 4:, :], o[:, :4, :]  # cls, box
    s_cls, _ = split(student_out)
    t_cls, _ = split(teacher_out)
    s_lp = F.log_softmax(s_cls / T, dim=1)
    t_p = F.softmax(t_cls / T, dim=1)
    return F.kl_div(s_lp, t_p, reduction="batchmean") * (T * T)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=Path, required=True, help="YOLOv8-L .pt")
    ap.add_argument("--data", type=Path, required=True, help="Ultralytics data yaml")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--kd-alpha", type=float, default=0.6)
    ap.add_argument("--kd-temperature", type=float, default=4.0)
    ap.add_argument("--project", type=str, default="runs/kd")
    ap.add_argument("--name", type=str, default="yolov8n_distill_from_l")
    args = ap.parse_args()

    if not args.teacher.exists():
        raise SystemExit(f"Teacher checkpoint not found: {args.teacher}")
    if not args.data.exists():
        raise SystemExit(f"Data yaml not found: {args.data}")

    teacher = YOLO(str(args.teacher)).model
    student = YOLO("yolov8n.pt")

    orig_init_loss = student.model._init_criterion if hasattr(student.model, "_init_criterion") else None

    def patched_loss(self):
        self.criterion = DistillationLoss(
            self, teacher, alpha=args.kd_alpha, temperature=args.kd_temperature,
        )
        return self.criterion

    # Ultralytics 8.x calls `self.criterion = self.model.init_criterion()`
    # inside BaseTrainer. Patch the student model class so the trainer picks
    # up our distillation loss on instantiation.
    student.model.init_criterion = patched_loss.__get__(student.model, type(student.model))

    student.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        exist_ok=True,
        device=0,
    )

    # Best weights are at runs/kd/yolov8n_distill_from_l/weights/best.pt
    # Export ONNX for Stage 1a-style quant pass, then copy to backend/models/.
    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    if best_pt.exists():
        YOLO(str(best_pt)).export(format="onnx", imgsz=args.imgsz, opset=14, simplify=True)
        print(f"\nDistilled student exported. Next steps:")
        print(f"  - scp {best_pt.with_suffix('.onnx')} <mac>:brickscan/backend/models/yolo_lego_n.onnx")
        print(f"  - run ml/export/quantize_yolo_int8.py on it (target ~5–8 MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
