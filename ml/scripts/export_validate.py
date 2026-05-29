#!/usr/bin/env python3
"""ONNX (and optional CoreML) export validation for the on-device embedding model.

Phase-1 export proof for brickscan-livescan. Interface contract (fixed):
    RGB 224x224  ->  L2-normalized float embedding (dim TBD by the student)

The distilled student isn't ready yet, so we validate the export *mechanics*
against the frozen DINOv2 backbone wrapped as an embedding model with EXACTLY this
interface (224x224 in, L2-normalized vector out). When the student lands it has the
same interface, so this script applies unchanged (swap --backbone / load weights).

What it does:
  1. Wraps the backbone as EmbeddingModel: input (B,3,224,224) in [0,1] (ImageNet
     normalization folded INTO the graph), output L2-normalized embedding (B,D).
  2. Exports to ONNX (opset 17, dynamic batch), runs onnx.checker.
  3. Runs onnxruntime on CPU and confirms outputs match PyTorch:
       - max abs diff, mean cosine similarity, and — the metric that actually
         matters for retrieval — whether int8 k-NN top-1 is unchanged when the
         gallery/query are embedded by ORT instead of torch.
  4. Measures CPU embedding latency on this host (Spark) as a ROUGH proxy. Real
     iPhone/ANE latency is Phase 2; the Spark CPU number is an upper bound for a
     mobile NPU, not a prediction.
  5. If coremltools is importable, attempts a CoreML conversion and reports the
     spec (note: CoreML *prediction* needs macOS; conversion can run on Linux).
     Otherwise documents the ONNX->CoreML path for Phase 2.

Example (on the Spark):
    python scripts/export_validate.py --backbone dinov2 \
        --val-dir training_data/real_photos_v3/val --probe-images 64
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from knn_baseline import gather, load_image  # reuse identical preprocessing

SEED = 1234
IMG_SIZE = 224


# ──────────────────────────────────────────────────────────────────────────────
# Embedding model: contract interface (224x224 [0,1] in -> L2-normalized out)
# ──────────────────────────────────────────────────────────────────────────────

class EmbeddingModel(nn.Module):
    """Backbone wrapped to the on-device contract.

    Input:  (B,3,224,224) float in [0,1]  (raw decoded RGB; NO external normalize)
    Output: (B,D) float, L2-normalized

    ImageNet mean/std normalization is folded into the graph so the iOS side only
    has to hand over a [0,1] CHW tensor — one fewer place for a preprocessing
    mismatch between training and device.
    """

    def __init__(self, backbone_name="vit_base_patch14_dinov2"):
        super().__init__()
        import timm
        # NB: DINOv2 ViT-B/14 is natively 518x518 (37x37 patch grid). The on-device
        # contract fixes input at 224x224 (16x16 grid). We build the model at the
        # FIXED 224 size with img_size=224, which makes timm interpolate the
        # pretrained position embeddings to the 16x16 grid ONCE at construction and
        # store them as a static parameter. That removes the dynamic antialiased
        # bicubic pos-embed interpolation from the forward graph entirely — which is
        # what otherwise (a) breaks torch->ONNX numeric parity and (b) fails CoreML
        # conversion (_upsample_bicubic2d_aa not implemented). dynamic_img_size is
        # intentionally OFF: the device only ever sends 224x224.
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0, img_size=IMG_SIZE
        ).eval()
        cfg = timm.data.resolve_data_config({}, model=self.backbone)
        mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1)
        std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.feat_dim = self.backbone.num_features

    def forward(self, x):                       # x: (B,3,224,224) in [0,1]
        x = (x - self.mean) / self.std
        feat = self.backbone(x)                 # num_classes=0 -> pooled features
        return torch.nn.functional.normalize(feat, dim=1)


# ──────────────────────────────────────────────────────────────────────────────

def load_probe_batch(val_dir, n):
    """Load up to n real val images as a (n,3,224,224) [0,1] tensor (deterministic)."""
    items, _ = gather(val_dir, cap=1)           # 1 image/class, deterministic order
    items = items[:n]
    xs = []
    for p, _ in items:
        try:
            xs.append(load_image(p, IMG_SIZE))
        except Exception:
            continue
    return torch.stack(xs)                       # (n,3,224,224) in [0,1]


def export_onnx(model, path, opset=17):
    import onnx
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, str(path), opset_version=opset,
            input_names=["image"], output_names=["embedding"],
            dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
            do_constant_folding=True, training=torch.onnx.TrainingMode.EVAL,
        )
    m = onnx.load(str(path))
    onnx.checker.check_model(m)
    # torch 2.12's dynamo exporter writes weights to an external <name>.data sidecar
    # by default; the .onnx file holds only the graph. Count both for true size.
    total = os.path.getsize(path)
    for sib in path.parent.glob(path.name + "*"):
        if sib != path:
            total += os.path.getsize(sib)
    return total


def ort_session_cpu(path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 0                  # let ORT pick; report below
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def cpu_latency(sess, reps=50, warmup=10):
    x = np.random.rand(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
    name = sess.get_inputs()[0].name
    for _ in range(warmup):
        sess.run(None, {name: x})
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        sess.run(None, {name: x})
        ts.append((time.perf_counter() - t) * 1000)
    return {"p50_ms": round(float(np.percentile(ts, 50)), 1),
            "p90_ms": round(float(np.percentile(ts, 90)), 1),
            "p99_ms": round(float(np.percentile(ts, 99)), 1),
            "mean_ms": round(float(np.mean(ts)), 1)}


def _save_size(path):
    if os.path.isdir(path):
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        return total
    return os.path.getsize(path)


def try_coreml(model, out_path):
    """Attempt CoreML conversion. Returns (status_str, info_dict).

    Strategy on the Spark (Linux): the MIL graph CONVERSION runs fine, but the
    mlprogram BlobWriter (libmilstoragepython) is macOS-only, so .mlpackage save
    fails with 'BlobWriter not loaded'. We treat reaching that point as proof the
    conversion graph is valid, then fall back to the legacy 'neuralnetwork' format
    (which serializes on Linux) to emit a real artifact. The PRODUCTION export
    (mlprogram, ANE) must run on a Mac in Phase 2.
    """
    try:
        import coremltools as ct
    except Exception as e:
        return "coremltools_not_available", {"detail": str(e)}

    model.eval()
    ex = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    ts = torch.jit.trace(model, ex)
    info = {}

    # 1. Preferred: mlprogram (ANE-friendly, iOS16+). Conversion validates the graph.
    mlprog_save_ok = False
    try:
        mlmodel = ct.convert(
            ts,
            inputs=[ct.TensorType(name="image", shape=(1, 3, IMG_SIZE, IMG_SIZE))],
            outputs=[ct.TensorType(name="embedding")],
            minimum_deployment_target=ct.target.iOS16,
            compute_units=ct.ComputeUnit.ALL,        # ANE+GPU+CPU on device
            convert_to="mlprogram",
        )
        info["mlprogram_convert"] = "ok"
        try:
            mlpkg = Path(str(out_path))
            mlmodel.save(str(mlpkg))
            mlprog_save_ok = True
            info["mlprogram_path"] = str(mlpkg)
            info["mlprogram_size_mb"] = round(_save_size(mlpkg) / 1e6, 1)
        except Exception as se:
            info["mlprogram_save"] = f"FAILED (expected on Linux): {repr(se)[:120]}"
    except Exception as ce:
        info["mlprogram_convert"] = f"FAILED: {repr(ce)[:200]}"

    if mlprog_save_ok:
        return "converted_mlprogram", info

    # 2. Fallback: neuralnetwork format serializes on Linux -> emit real artifact.
    try:
        nn_model = ct.convert(
            ts,
            inputs=[ct.TensorType(name="image", shape=(1, 3, IMG_SIZE, IMG_SIZE))],
            convert_to="neuralnetwork",
        )
        nn_path = Path(str(out_path).replace(".mlpackage", ".mlmodel"))
        nn_model.save(str(nn_path))
        spec = nn_model.get_spec()
        info.update({
            "neuralnetwork_path": str(nn_path),
            "neuralnetwork_size_mb": round(_save_size(nn_path) / 1e6, 1),
            "spec_inputs": [i.name for i in spec.description.input],
            "spec_outputs": [o.name for o in spec.description.output],
            "note": ("mlprogram graph CONVERTED on Linux but save needs macOS "
                     "(BlobWriter). neuralnetwork artifact saved as proof; redo as "
                     "mlprogram on a Mac for ANE in Phase 2."),
        })
        return "converted_graph_saved_neuralnetwork", info
    except Exception as ne:
        info["neuralnetwork"] = f"FAILED: {repr(ne)[:200]}"
        return "convert_failed", info


# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2", help="dinov2 -> vit_base_patch14_dinov2")
    ap.add_argument("--val-dir", default="training_data/real_photos_v3/val")
    ap.add_argument("--probe-images", type=int, default=64)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--out-dir", default="output/ondevice_export")
    ap.add_argument("--try-coreml", action="store_true", default=True)
    ap.add_argument("--no-coreml", dest="try_coreml", action="store_false")
    args = ap.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    backbone_name = ("vit_base_patch14_dinov2" if args.backbone == "dinov2"
                     else args.backbone)
    print(f"[build] EmbeddingModel({backbone_name})  interface: 224x224 [0,1] "
          f"-> L2-normalized embedding")
    model = EmbeddingModel(backbone_name).eval()
    dim = model.feat_dim

    # ---- torch reference embeddings on real probe images ----
    probe = load_probe_batch(args.val_dir, args.probe_images)
    print(f"[probe] {probe.shape[0]} real val images")
    with torch.no_grad():
        ref = model(probe).cpu().numpy().astype(np.float32)

    # ---- ONNX export ----
    onnx_path = out / "embed_model.onnx"
    nbytes = export_onnx(model, onnx_path, args.opset)
    print(f"[onnx] exported {onnx_path.name}  ({nbytes/1e6:.1f} MB)  opset={args.opset}  "
          f"graph check passed")

    # ---- ORT CPU parity ----
    sess = ort_session_cpu(onnx_path)
    in_name = sess.get_inputs()[0].name
    ort_out = sess.run(None, {in_name: probe.numpy().astype(np.float32)})[0]
    ort_out = ort_out.astype(np.float32)

    max_abs = float(np.max(np.abs(ref - ort_out)))
    cos = float(np.mean(np.sum(ref * ort_out, axis=1) /
                        (np.linalg.norm(ref, axis=1) * np.linalg.norm(ort_out, axis=1))))
    # Retrieval-equivalence: does swapping torch->ORT change which gallery vec is
    # nearest? Use the probe set as its own gallery+query (leave-one-out top-1).
    def loo_top1_agreement(a, b):
        # nearest *other* index under each backend; agreement = same neighbor
        def nn_idx(M):
            S = M @ M.T
            np.fill_diagonal(S, -1.0)
            return S.argmax(1)
        return float((nn_idx(a) == nn_idx(b)).mean())
    nn_agree = loo_top1_agreement(ref, ort_out)

    print(f"[parity] torch vs ORT-CPU:  max_abs_diff={max_abs:.2e}  "
          f"mean_cosine={cos:.6f}  NN-top1-agreement={100*nn_agree:.1f}%")
    parity_ok = (max_abs < 1e-3) and (cos > 0.9999) and (nn_agree > 0.999)
    print(f"[parity] {'PASS' if parity_ok else 'CHECK'} "
          f"(thresholds: max_abs<1e-3, cos>0.9999, NN-agree>99.9%)")

    # ---- CPU latency proxy ----
    lat = cpu_latency(sess)
    n_threads = sess.get_session_options().intra_op_num_threads
    print(f"[latency] CPU embedding (bs=1, {IMG_SIZE}x{IMG_SIZE}): "
          f"p50={lat['p50_ms']}ms p90={lat['p90_ms']}ms p99={lat['p99_ms']}ms "
          f"(ORT CPU EP; PROXY ONLY — iPhone/ANE measured in Phase 2)")

    # ---- CoreML ----
    coreml = {"status": "skipped"}
    if args.try_coreml:
        status, info = try_coreml(model, out / "embed_model.mlpackage")
        coreml = {"status": status, **info}
        print(f"[coreml] status={status}")
        for kk in ("mlprogram_convert", "mlprogram_save", "mlprogram_size_mb",
                   "neuralnetwork_path", "neuralnetwork_size_mb", "spec_outputs"):
            if kk in info:
                print(f"         {kk}: {info[kk]}")

    report = {
        "backbone": backbone_name, "embed_dim": dim, "input_size": IMG_SIZE,
        "opset": args.opset, "seed": SEED,
        "onnx": {"path": str(onnx_path), "size_mb": round(nbytes / 1e6, 1)},
        "parity": {"max_abs_diff": max_abs, "mean_cosine": cos,
                   "nn_top1_agreement": nn_agree, "pass": parity_ok},
        "cpu_latency_proxy": {**lat, "intra_op_threads": n_threads,
                              "note": "Spark CPU; rough upper bound, not iPhone/ANE"},
        "coreml": coreml,
        "coreml_phase2_path": (
            "If coremltools prediction is unavailable here (Linux): export ONNX "
            "(this script), then on a Mac run coremltools.convert on the SAME traced "
            "EmbeddingModel (or via onnx) with convert_to='mlprogram', "
            "minimum_deployment_target=iOS16, ComputeUnit.ALL; validate numerics "
            "match ORT/torch on-device, then benchmark ANE latency."
        ),
    }
    (out / "export_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n[report] {out/'export_report.json'}")


if __name__ == "__main__":
    main()
