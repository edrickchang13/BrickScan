# On-device retrieval — Phase 1 mechanics (notes for Phase 2 iOS work)

Scripts: `scripts/ondevice_index.py` (int8 k-NN proof), `scripts/export_validate.py`
(ONNX/CoreML export + parity + CPU latency). Validated against frozen DINOv2 as a
stand-in for the distilled student (same interface: RGB 224x224 -> L2-normalized
embedding -> cosine k-NN). Seeded (1234), reproducible. Run on the Spark:
`cd ~/brickscan/ml && . venv/bin/activate && python scripts/<name>.py`.

## Non-obvious findings that will bite Phase 2

1. **int8 quantization must use a SINGLE GLOBAL scale (1/127), not per-vector.**
   Embeddings are L2-normalized, so every component is already in [-1,1] and a
   global scale is near-optimal AND preserves cosine ranking. Per-vector scaling
   gives each gallery vector its own scale, which breaks cross-gallery ranking and
   tanked top-1 from ~85% to ~64% in testing. USearch's `ScalarKind.I8` does the
   right (global-style) thing. If the iOS side hand-rolls quantization, use one
   scale for the whole index.

2. **Quantize the GALLERY to int8; keep the QUERY in float.** The query embedding
   comes fresh from the model at inference. Asymmetric (float query · int8 gallery)
   was the best int8 variant (+0.13pp vs float) and avoids a round-trip. USearch
   handles this internally when you pass a float query to an I8 index.

3. **Bake position embeddings at the fixed 224 grid; do NOT use `dynamic_img_size`.**
   DINOv2 ViT-B/14 is natively 518x518. Building with `img_size=224` makes timm
   interpolate pos-embeds to the 16x16 grid ONCE at construction (static weights).
   With `dynamic_img_size=True` the forward graph contains a dynamic antialiased
   bicubic interp (`_upsample_bicubic2d_aa`) that (a) drops torch->ONNX parity to
   ~96.9% NN agreement and (b) is unsupported by coremltools. Baking it fixed gave
   perfect parity (max_abs 1.45e-6, cosine 1.000000, 100% NN agreement). The
   distilled student should likewise have a static input size of 224.

4. **Fold ImageNet mean/std normalization INTO the exported graph.** The model
   takes a [0,1] CHW tensor; normalization is a graph op. One fewer train/device
   preprocessing mismatch. Verified: ORT output L2 norm == 1.0 exactly.

5. **torch 2.12 dynamo ONNX export externalizes weights** to a `<name>.onnx.data`
   sidecar — the `.onnx` file is just the graph (~1MB). Ship BOTH files. True size
   = graph + sidecar (344 MB for the DINOv2 stand-in; the student will be far
   smaller).

## CoreML status (Spark is Linux)

coremltools 9.0 is installed and the MIL **conversion** of the full model runs
end-to-end, but the `mlprogram` BlobWriter (`libmilstoragepython`) and the
predictor (`libcoremlpython`) are **macOS-only** — a trivial model fails the same
way, so it's purely environmental, not model-specific. We emit a `neuralnetwork`
artifact (`embed_model.mlmodel`) on Linux as proof the graph serializes.

### Phase 2 CoreML path (run on a Mac)
```python
import coremltools as ct, torch
m = EmbeddingModel("vit_base_patch14_dinov2").eval()   # img_size=224, see #3
ts = torch.jit.trace(m, torch.zeros(1,3,224,224))
ml = ct.convert(ts,
    inputs=[ct.TensorType(name="image", shape=(1,3,224,224))],
    outputs=[ct.TensorType(name="embedding")],
    minimum_deployment_target=ct.target.iOS16,
    compute_units=ct.ComputeUnit.ALL,      # ANE + GPU + CPU
    convert_to="mlprogram")
ml.save("embed_model.mlpackage")
```
Then on-device: validate embeddings match ORT/torch (cosine > 0.999), benchmark ANE
latency, and consider `ct.optimize.coreml` palettization/int8 weight compression to
shrink the model further. USearch ships a Swift Package + ObjC headers, so the
int8 index in #1-#2 ports directly; the `.usearch` file built here loads as-is.
