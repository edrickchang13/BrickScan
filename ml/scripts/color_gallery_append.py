#!/usr/bin/env python3
"""Append a confirmed color exemplar to the color gallery — FLYWHEEL, no retrain.

The color model (scripts/color_model.py, models/color_v1/color_model.npz) is a
frozen feature extractor + a BAKED LDA transform + a distance-weighted k-NN over
a gallery of train exemplars already projected into LDA space (gallery_proj[N,C],
gallery_y[N]). Recognising a new/under-represented color works exactly like the
part spine: append more exemplars to the gallery. NO sklearn refit, NO retraining
— the z-score (feat_mu/feat_sd) and LDA (lda_xbar/lda_scalings) stay frozen; we
only project the new crop through them and stack one row onto the gallery.

Why this is correct without refit
----------------------------------
`ColorClassifier.predict` ranks colors by distance-weighted k-NN in the LDA
space. The LDA basis is fixed; adding a gallery point in that fixed basis just
gives the matcher a new neighbour to vote with. The new crop is projected with
the SAME pipeline the artifact uses at inference:
    feature_from_rgb01 (12-d)  ->  (z - feat_mu)/feat_sd  ->  (z - lda_xbar) @ lda_scalings
so the appended row lives in the identical coordinate system as the seed gallery.
(This reuses ColorClassifier.feature + .project verbatim, so it stays bit-for-bit
consistent with eval.)

A new color id that wasn't in the original gallery is supported: its label is
added to gallery_y and (optionally) to color_ids/names/hex via --colors-csv, so
predict() can return it immediately.

Usage (on the Spark; reproducible):

    cd ~/brickscan/ml && . venv/bin/activate
    # Append one confirmed crop for Rebrickable color 5 (Red) to the artifact:
    python scripts/color_gallery_append.py append \
        --artifact   models/color_v1/color_model.npz \
        --image      training_data/color_v1/val/5/<some>.jpg \
        --color-id   5

    # Append in bulk from a dir tree <root>/<color_id>/*.jpg (e.g. confirmed
    # feedback crops), then re-eval straight from the artifact:
    python scripts/color_gallery_append.py append-dir \
        --artifact models/color_v1/color_model.npz \
        --root /path/to/confirmed_color_crops
    python scripts/color_model.py eval \
        --artifact models/color_v1/color_model.npz \
        --val-dir  training_data/color_v1/val
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Reuse the EXACT extractor, projection and metadata loader the artifact uses,
# so an appended exemplar is indistinguishable from a seed one.
from color_model import (
    ARTIFACT_VERSION,
    ColorClassifier,
    IMG_EXT,
    load_color_meta,
    load_img,
)


def _project_crop(clf: ColorClassifier, image_path: str) -> Optional[np.ndarray]:
    """image path -> 1 row in LDA space, or None if no brick body segmented."""
    img = load_img(image_path, clf.p["img_size"])
    if img is None:
        return None
    feat = clf.feature(img)               # 12-d, same as eval
    if feat is None:
        return None
    return np.asarray(clf.project(feat), dtype=np.float32).reshape(1, -1)


def _load_artifact_arrays(artifact_path: str) -> dict:
    """Load the full npz into a writable dict (so we can grow the gallery)."""
    z = np.load(artifact_path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _save_artifact_arrays(artifact_path: str, arrays: dict, *, backup: bool) -> None:
    """Atomically rewrite the npz (+ optional .bak) and refresh the sidecar count."""
    artifact_path = str(artifact_path)
    if backup and os.path.exists(artifact_path):
        bak = artifact_path + ".bak"
        if not os.path.exists(bak):                 # keep the FIRST (pristine) backup
            Path(bak).write_bytes(Path(artifact_path).read_bytes())
    # np.savez_compressed appends ".npz" to a path without that suffix, so write
    # to a ".npz"-suffixed temp explicitly and atomically replace the target.
    tmp = artifact_path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, artifact_path)

    # Touch the human-readable sidecar's gallery_size if present (non-fatal).
    sidecar = os.path.splitext(artifact_path)[0] + ".meta.json"
    if os.path.exists(sidecar):
        try:
            meta = json.loads(Path(sidecar).read_text())
            meta["gallery_size"] = int(arrays["gallery_proj"].shape[0])
            meta["num_colors"] = int(len(set(arrays["gallery_y"].astype(str).tolist())))
            meta.setdefault("flywheel", {})
            meta["flywheel"]["last_append_ts"] = time.time()
            Path(sidecar).write_text(json.dumps(meta, indent=2))
        except Exception:
            pass


def append_rows(
    arrays: dict, new_proj: np.ndarray, new_labels: List[str],
    colors_csv: Optional[str] = None,
) -> Tuple[dict, int, List[str]]:
    """Stack projected rows + labels onto the gallery. Returns (arrays, n_added,
    newly_introduced_color_ids). Extends color_ids/names/hex for unseen labels."""
    gp = arrays["gallery_proj"].astype(np.float32)
    gy = arrays["gallery_y"].astype(str)
    if new_proj.shape[1] != gp.shape[1]:
        raise ValueError(f"LDA dim mismatch: gallery={gp.shape[1]} new={new_proj.shape[1]}")

    arrays["gallery_proj"] = np.vstack([gp, new_proj.astype(np.float32)])
    arrays["gallery_y"] = np.concatenate([gy, np.array(new_labels, dtype=gy.dtype)])

    # Register any brand-new color id so predict() can name it.
    known = set(arrays["color_ids"].astype(str).tolist())
    introduced = [c for c in dict.fromkeys(new_labels) if c not in known]
    if introduced:
        meta = load_color_meta(colors_csv) if colors_csv else {}
        ids = arrays["color_ids"].astype(str).tolist()
        names = list(arrays["color_names"].tolist())
        hexes = arrays["color_hex"].astype(str).tolist()
        for c in introduced:
            nm, hx = meta.get(c, (c, ""))
            ids.append(c); names.append(nm); hexes.append(hx)
        arrays["color_ids"] = np.array(ids, dtype="U8")
        arrays["color_names"] = np.array(names, dtype=object)
        arrays["color_hex"] = np.array(hexes, dtype="U6")
    return arrays, len(new_labels), introduced


# ──────────────────────────────────────────────────────────────────────────────
def _cmd_append(args):
    clf = ColorClassifier(args.artifact)         # for feature() + project() only
    proj = _project_crop(clf, args.image)
    if proj is None:
        print(f"[append] no brick body segmented from {args.image} — nothing added")
        return
    arrays = _load_artifact_arrays(args.artifact)
    before = arrays["gallery_proj"].shape[0]
    arrays, n, introduced = append_rows(
        arrays, proj, [str(args.color_id)], colors_csv=args.colors_csv)
    _save_artifact_arrays(args.artifact, arrays, backup=not args.no_backup)
    print(f"[append] +{n} exemplar for color {args.color_id} "
          f"(gallery {before} -> {arrays['gallery_proj'].shape[0]})"
          + (f"  NEW color id(s): {introduced}" if introduced else ""))

    # Immediate-effect proof: the just-added crop must now retrieve its own
    # color as top-1 (it's its own nearest neighbour at distance 0).
    pred = clf.predict(args.image, topk=1)        # uses the OLD in-memory gallery
    clf2 = ColorClassifier(args.artifact)         # reload -> sees the appended row
    pred2 = clf2.predict(args.image, topk=1)
    print(f"[append] top-1 before reload={pred[0]['id'] if pred else None} "
          f"after reload={pred2[0]['id'] if pred2 else None} "
          f"(want {args.color_id})")


def _cmd_append_dir(args):
    clf = ColorClassifier(args.artifact)
    root = Path(args.root)
    color_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    rows, labels, skipped = [], [], 0
    for d in color_dirs:
        cid = d.name
        files = sorted(f for f in os.listdir(d)
                       if os.path.splitext(f)[1].lower() in IMG_EXT)
        if args.per_color:
            files = files[: args.per_color]
        for fn in files:
            proj = _project_crop(clf, str(d / fn))
            if proj is None:
                skipped += 1
                continue
            rows.append(proj)
            labels.append(str(cid))
    if not rows:
        print(f"[append-dir] no segmentable crops under {root} — nothing added")
        return
    new_proj = np.vstack(rows)
    arrays = _load_artifact_arrays(args.artifact)
    before = arrays["gallery_proj"].shape[0]
    arrays, n, introduced = append_rows(
        arrays, new_proj, labels, colors_csv=args.colors_csv)
    _save_artifact_arrays(args.artifact, arrays, backup=not args.no_backup)
    print(f"[append-dir] +{n} exemplars across {len(set(labels))} colors "
          f"(skipped {skipped} unsegmentable); gallery {before} -> "
          f"{arrays['gallery_proj'].shape[0]}"
          + (f"  NEW color id(s): {introduced}" if introduced else ""))


def _cmd_inspect(args):
    z = np.load(args.artifact, allow_pickle=True)
    gy = z["gallery_y"].astype(str)
    uniq, counts = np.unique(gy, return_counts=True)
    print(f"artifact     : {args.artifact}  (v{int(z['version'])}, "
          f"want v{ARTIFACT_VERSION})")
    print(f"gallery_proj : {z['gallery_proj'].shape}")
    print(f"gallery_y    : {gy.shape[0]} exemplars over {len(uniq)} colors")
    order = np.argsort(-counts)
    print("per-color exemplar counts (top 15):")
    for i in order[:15]:
        print(f"  color {uniq[i]:>4}: {counts[i]}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("append", help="append ONE confirmed crop for a color id")
    a.add_argument("--artifact", required=True)
    a.add_argument("--image", required=True)
    a.add_argument("--color-id", required=True)
    a.add_argument("--colors-csv", default=None,
                   help="rebrickable colors.csv, to name a brand-new color id")
    a.add_argument("--no-backup", action="store_true",
                   help="skip writing a one-time .bak of the pristine artifact")
    a.set_defaults(func=_cmd_append)

    ad = sub.add_parser("append-dir", help="bulk-append from <root>/<color_id>/*.jpg")
    ad.add_argument("--artifact", required=True)
    ad.add_argument("--root", required=True)
    ad.add_argument("--per-color", type=int, default=0,
                    help="cap exemplars added per color (0 = all)")
    ad.add_argument("--colors-csv", default=None)
    ad.add_argument("--no-backup", action="store_true")
    ad.set_defaults(func=_cmd_append_dir)

    ins = sub.add_parser("inspect", help="show gallery size + per-color counts")
    ins.add_argument("--artifact", required=True)
    ins.set_defaults(func=_cmd_inspect)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
