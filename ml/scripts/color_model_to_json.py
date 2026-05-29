#!/usr/bin/env python3
"""Convert the portable color model (color_model.npz) into a JS-loadable asset.

Phase 2 ships the LDA-kNN color matcher on-device (mobile/src/ml/colorClassifier.ts).
The TS side cannot read a numpy .npz, so this packs the artifact into a single
JSON file that Metro bundles directly (require('.../color_model.json')).

What's in the npz (see scripts/color_model.py:build_artifact) and how we carry it:

  feat_mu[12], feat_sd[12]        -> plain JSON arrays (z-score)
  lda_xbar[12], lda_scalings[12,12] -> plain JSON arrays (LDA transform;
                                       projection is (z - xbar) @ scalings)
  gallery_proj[N,12] float32      -> int16-quantized base64 blob + a scale.
                                     N=15918, so float32 would be ~764 KB of
                                     JSON-as-numbers; an int16 blob is ~382 KB
                                     raw and the max quant error (~2e-4 in LDA
                                     space) changes the top-1 / top-3 ranking on
                                     <0.3% of queries (verified). The TS loader
                                     dequantizes once at load: q_f = q_i16 / scale.
  gallery_y[N]                     -> the per-exemplar color id, encoded as
                                     uint16 indices into `classes` (a sorted,
                                     deduped id list) + the same base64 scheme,
                                     so the gallery labels add ~32 KB not ~120 KB.
  knn_k, extraction params         -> JSON scalars (the feature pipeline config)
  color_ids / names / hex          -> JSON arrays (id -> name/hex for the result)

The blobs are little-endian and base64'd; colorClassifier.ts decodes them with
the same atob path nativePixelBridge.ts uses, then reinterprets the bytes as a
typed array. Keep this in lockstep with colorClassifier.ts's loader.

  python scripts/color_model_to_json.py \
      --artifact models/color_v1/color_model.npz \
      --out ../mobile/assets/models/color_model.json
"""
import argparse
import base64
import json
import os

import numpy as np

# Asset schema version. Bump (and bump the TS COLOR_ASSET_VERSION check) on any
# breaking change to the JSON layout below.
ASSET_VERSION = 1


def _b64_int16(arr_f32, abs_max=None):
    """Symmetric int16-quantize a float array -> (base64 LE bytes, scale).

    Dequantize on the JS side as value = int16 / scale.
    """
    if abs_max is None:
        abs_max = float(np.abs(arr_f32).max())
    abs_max = max(abs_max, 1e-12)
    scale = 32767.0 / abs_max
    q = np.round(np.asarray(arr_f32, np.float64) * scale)
    q = np.clip(q, -32768, 32767).astype("<i2")  # little-endian int16
    return base64.b64encode(q.tobytes()).decode("ascii"), scale


def _b64_uint16(arr_int):
    """Encode small non-negative integers (class indices) as LE uint16 base64."""
    q = np.asarray(arr_int, dtype="<u2")
    return base64.b64encode(q.tobytes()).decode("ascii")


def convert(artifact_path, out_path):
    z = np.load(artifact_path, allow_pickle=True)

    if int(z["version"]) != 1:
        raise SystemExit(f"unexpected artifact version {int(z['version'])}")

    gallery_proj = z["gallery_proj"].astype(np.float32)        # [N,12]
    gallery_y = np.array([str(c) for c in z["gallery_y"]])     # [N]
    n, dim = gallery_proj.shape

    # Stable, sorted class list — MUST match ColorClassifier._classes ordering in
    # color_model.py (sorted(set(gallery_y))), which is the kNN vote accumulator
    # order. The TS kNN reproduces that exact order.
    classes = sorted(set(gallery_y.tolist()))
    cls_idx = {c: i for i, c in enumerate(classes)}
    gallery_y_idx = np.array([cls_idx[c] for c in gallery_y], dtype=np.int64)

    proj_b64, proj_scale = _b64_int16(gallery_proj)
    y_b64 = _b64_uint16(gallery_y_idx)

    color_ids = [str(c) for c in z["color_ids"]]
    color_names = [str(s) for s in z["color_names"]]
    color_hex = [str(s) for s in z["color_hex"]]

    asset = {
        "assetVersion": ASSET_VERSION,
        "artifactVersion": int(z["version"]),
        "method": "lda-knn",
        # ---- z-score ----
        "featMu": [float(x) for x in z["feat_mu"]],
        "featSd": [float(x) for x in z["feat_sd"]],
        # ---- LDA transform: P = (z - xbar) @ scalings ----
        "ldaXbar": [float(x) for x in z["lda_xbar"]],
        # row-major [12][12]
        "ldaScalings": [[float(v) for v in row] for row in z["lda_scalings"]],
        "featDim": int(dim),
        "ldaDim": int(z["lda_scalings"].shape[1]),
        # ---- gallery (LDA space), quantized ----
        "gallery": {
            "count": int(n),
            "dim": int(dim),
            "projScale": proj_scale,         # dequant: int16 / projScale
            "projI16B64": proj_b64,          # [N*dim] little-endian int16
            "yIdxU16B64": y_b64,             # [N]     little-endian uint16
        },
        # ---- matcher ----
        "knnK": int(z["knn_k"]),
        "knnWeights": str(z["knn_weights"]),
        # ---- class order for the kNN vote accumulator ----
        "classes": classes,
        # ---- extraction params (the feature pipeline) ----
        "extraction": {
            "imgSize": int(z["img_size"]),
            "wb": bool(z["wb"]),
            "wbP": int(z["wb_p"]),
            "borderFrac": float(z["border_frac"]),
            "fgThresh": float(z["fg_thresh"]),
            "hiV": float(z["hi_v"]),
            "hiS": float(z["hi_s"]),
            "loV": float(z["lo_v"]),
        },
        # ---- color metadata (id -> name/hex) ----
        "colorIds": color_ids,
        "colorNames": color_names,
        "colorHex": color_hex,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    # Compact separators: this is a bundled asset, not meant to be hand-edited.
    with open(out_path, "w") as f:
        json.dump(asset, f, separators=(",", ":"))

    size_kb = os.path.getsize(out_path) / 1024.0
    print(f"  wrote {out_path}  ({size_kb:.1f} KB)")
    print(f"  gallery={n}x{dim}  classes={len(classes)}  colors={len(color_ids)}  "
          f"k={int(z['knn_k'])}  projScale={proj_scale:.4f}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--artifact", default="models/color_v1/color_model.npz")
    ap.add_argument("--out", default="../mobile/assets/models/color_model.json")
    args = ap.parse_args()
    convert(args.artifact, args.out)


if __name__ == "__main__":
    main()
