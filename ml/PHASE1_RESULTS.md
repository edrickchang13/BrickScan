# BrickScan Live-Scan — Phase 1 Results (on-device recognition core)

Goal: take the Phase 0 retrieval spine and make it run **on the phone** — distill
the heavy DINOv2 teacher into a small ANE-friendly student, prove on-device int8
vector search, and package the color model as a shippable artifact.

All work on the shared DGX Spark (GB10). Scripts in `ml/scripts/`; artifacts in
`ml/models/`. Reproduce via `ssh spark && cd ~/brickscan/ml && . venv/bin/activate`.

## 1. Distilled on-device student (`scripts/student_index_eval.py`, FastViT-SA24) — ✅

Distilled the frozen DINOv2 ViT-B/14 teacher into a small, ANE-friendly **FastViT-SA24**
student (cosine + 0.5·MSE embedding regression over 86,845 cached teacher embeddings,
4 epochs). Authoritative retrieval eval on the 439-class held-out set (gallery 30/class,
student-embedded, int8 global-scale index):

| Metric | Teacher (DINOv2 ViT-B) | **Student (FastViT-SA24)** |
|---|---|---|
| params | ~86 M | **21.3 M** |
| embedding dim | 768 | 768 |
| single-frame top-1 | 89% | **90.1%** |
| fused top-1 (N=4, conf-wt) | 95.4% | **95.6%** |

**The #1 Phase 1 risk — that distilling to a tiny on-device model would tank accuracy —
did not materialize: the 21 M-param student matches (even edges) the heavyweight teacher
at both single-frame and fused.** Exported to ONNX (86 MB fp32, gitignored; int8
quantization is a follow-up to shrink the app bundle) and wired into the app via
`liveScanEngine` with the `gallery_index.json` (12.5k int8 exemplars). _Note: onnxruntime
ran on CPU on the Spark (no CUDA EP there) — a tooling detail, not a model issue; on-device
the student targets the Apple Neural Engine._ Look-alike basic bricks (3001–3005) remain
the residual, improving with fusion but still the hard cases (consistent with Phase 0).

## 2. On-device int8 vector search (`scripts/ondevice_index.py` — ondevice-eng) — ✅

Build the gallery embedding index, quantize to int8, and confirm retrieval survives
quantization + fits on a phone. (Validated on DINOv2 embeddings as the model stand-in;
applies to any same-interface model, incl. the Phase 1 student.)

**int8 is effectively lossless for retrieval** (gallery 16/class, query 8/class):

| Index | top-1 | recall@3 | recall@5 | agree w/ float |
|---|---|---|---|---|
| float32 | 85.2% | 92.2% | 94.5% | — |
| int8 (symmetric) | 85.4% | 92.0% | 94.4% | 96.2% |
| usearch int8 (HNSW) | 85.1% | 92.0% | 94.4% | 96.4% |

**Index size + speed** — 439 classes: 21.3 MB float32 → **5.3 MB int8** (6.4 MB
USearch HNSW). Confirmed projections at 4 exemplars/part: **~37 MB @ 10k parts,
~279 MB @ 76k parts** (int8; vs 123 MB / 934 MB at fp32) — phone-viable, with
**sub-millisecond search** (0.15 ms @10k, 0.28 ms @76k; HNSW is logarithmic).

**Export** — ONNX clean (opset-17; torch↔ORT parity max-diff 1.4e-6, 100% NN
agreement; output L2-norm exactly 1.0; weights externalize to a `.onnx.data`
sidecar). CoreML: the MIL graph converts, but `mlprogram` serialization + ANE
prediction are **macOS-only** — production CoreML/ANE export runs on a Mac (recipe in
`scripts/ONDEVICE_NOTES.md`). CPU latency proxy ~105–193 ms on the Spark (rough; real
ANE latency is Phase 2, and on the small student, not this DINOv2 stand-in).

**Two guardrails (each costs real accuracy if missed):** (a) int8 must use a
**single global scale** (1/127), never per-vector — per-vector collapses cosine
ranking (measured 85%→64%); (b) bake DINOv2 pos-embeds at a **fixed 224 grid** (not
`dynamic_img_size`) or ONNX parity drops to ~97% and coremltools breaks.

## 3. Portable color model (`scripts/color_model.py`, `models/color_v1/`) — ✅

The Phase 0 color pipeline (84.2% top-1) packaged as a **self-contained 702 KB
artifact** (`color_model.npz`): LDA transform + 15,918 real exemplars projected into
LDA space + extraction params + color metadata. Pure-numpy inference (sklearn-free,
verified bit-for-bit), loads only the artifact — no training data at inference.

- Reproduced **from the artifact alone** on `color_v1/val`: **84.2% top-1 / 90.8% top-3.**
- A per-color-prototype variant shrinks to 8.4 KB but drops to 71% — ship the 702 KB
  full-gallery version (trivially small for on-device).

## Takeaway (pending §1)

On-device vector search and color are both proven shippable. The open question — and
the Phase 1 success criterion — is the **student's accuracy gap**: if the distilled
FastViT student stays close to the teacher (and multi-frame fusion recovers the rest),
the full live pipeline runs on-device. Updating §1 the moment training completes.
