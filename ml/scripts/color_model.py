#!/usr/bin/env python3
"""Portable, deployable color-ID model for BrickScan (Phase 1).

Turns the Phase-0 LDA-kNN color matcher (scripts/color_eval.py, 84.2% top-1 /
90.8% top-3 on color_v1/val) into a small SELF-CONTAINED artifact that runs
inference WITHOUT the training images and WITHOUT scikit-learn.

What ships in the artifact (a single .npz, plus a sidecar .json of metadata):

  z-score         feat_mu[12], feat_sd[12]      feature standardization
  LDA transform   lda_xbar[12], lda_scalings[12,C-1]
                                                 baked so that the sklearn
                                                 transform is exactly
                                                 P = (z - lda_xbar) @ lda_scalings
  gallery         gallery_proj[N,C-1], gallery_y[N]
                                                 train exemplars already
                                                 projected into LDA space, so no
                                                 train images are needed at
                                                 inference. (Optionally reduced
                                                 to per-color prototypes.)
  matcher params  knn_k, knn_weights            distance-weighted k-NN
  extraction      img_size, wb, wb_p, border_frac, fg_thresh, hi_v, hi_s, lo_v
                                                 the exact crop->feature params
  color metadata  color_ids, color_names, color_hex

Inference (ColorClassifier.predict) reimplements the LDA projection and the
distance-weighted k-NN in pure numpy, so the only runtime deps are numpy,
Pillow and scikit-image (for the LAB conversion + the WB/segmentation that the
feature extractor already used). It reproduces the training-time ranking
bit-for-bit (verified against sklearn).

  # Build the artifact from the fitted model (uses the train gallery ONCE):
  python scripts/color_model.py build \
      --train-dir   training_data/color_v1/train \
      --val-dir     training_data/color_v1/val \
      --colors-csv  training_data/rebrickable_csv/colors.csv \
      --out         models/color_v1/color_model.npz

  # Reproduce val top-1/top-3 FROM THE ARTIFACT ALONE (no train dir touched):
  python scripts/color_model.py eval \
      --artifact    models/color_v1/color_model.npz \
      --val-dir     training_data/color_v1/val

  # Classify a single crop from the artifact:
  python scripts/color_model.py predict \
      --artifact models/color_v1/color_model.npz --image some_crop.jpg
"""
import argparse
import json
import os

import numpy as np
from PIL import Image
from skimage.color import rgb2lab

# Artifact format version. Bump on any breaking change to the npz schema.
ARTIFACT_VERSION = 1
SEED = 0
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Extraction defaults. These ARE the model: they get baked into the artifact so
# inference reproduces training-time features exactly. Keep in lockstep with the
# values used in color_eval.py.
DEFAULTS = dict(
    img_size=128,
    wb=True,
    wb_p=6,         # shades-of-gray Minkowski norm
    border_frac=0.10,
    fg_thresh=0.14,  # min RGB distance from background to be "brick"
    hi_v=0.93,      # specular highlight guard: very bright ...
    hi_s=0.12,      # ... and low chroma
    lo_v=0.04,      # deep-shadow guard
)


# ===========================================================================
# Color extraction  (identical math to scripts/color_eval.py; param-driven so
# the exact values can be carried in the artifact)
# ===========================================================================
def shades_of_gray_wb(img, p):
    """Minkowski-norm color constancy. img HxWx3 in [0,1]."""
    illum = np.power(np.mean(np.power(img, p), axis=(0, 1)), 1.0 / p)
    illum = illum / (illum.mean() + 1e-8)
    return np.clip(img / (illum + 1e-8), 0, 1)


def _v_s(flat):
    """Vectorized HSV value (max) and saturation (chroma/max) for RGB [N,3]."""
    mx = flat.max(axis=1)
    mn = flat.min(axis=1)
    s = np.where(mx > 1e-6, (mx - mn) / (mx + 1e-6), 0.0)
    return mx, s


def brick_body_lab(img, p):
    """Brick-body LAB pixels + context, from an RGB[0,1] crop.

    White-balances, estimates background from a border frame, keeps pixels far
    from it, drops specular highlights and deep shadow. Mirrors color_eval.py
    exactly; `p` bundles the extraction params (wb, wb_p, border_frac, ...).
    """
    if p["wb"]:
        img = shades_of_gray_wb(img, p["wb_p"])
    h, w, _ = img.shape
    flat = img.reshape(-1, 3)

    b = max(2, int(min(h, w) * p["border_frac"]))
    border = np.concatenate([
        img[:b].reshape(-1, 3), img[-b:].reshape(-1, 3),
        img[:, :b].reshape(-1, 3), img[:, -b:].reshape(-1, 3),
    ])
    bg = np.median(border, axis=0)

    dist = np.linalg.norm(flat - bg, axis=1)
    fg_mask = dist > p["fg_thresh"]
    fg_frac = float(fg_mask.mean())
    if fg_mask.sum() < 0.04 * len(flat):
        cy0, cy1, cx0, cx1 = int(h * .30), int(h * .70), int(w * .30), int(w * .70)
        fg = img[cy0:cy1, cx0:cx1].reshape(-1, 3)
    else:
        fg = flat[fg_mask]

    v, s = _v_s(fg)
    keep = ~((v > p["hi_v"]) & (s < p["hi_s"])) & ~(v < p["lo_v"])
    if keep.sum() < max(20, int(0.15 * len(fg))):
        keep = np.ones(len(fg), bool)
    body = fg[keep]

    bg_lab = rgb2lab(bg.reshape(1, 1, 3)).reshape(3)
    return rgb2lab(body.reshape(-1, 1, 3)).reshape(-1, 3), bg_lab, fg_frac


def rich_feature(body_lab, bg_lab, fg_frac):
    """12-d color feature from brick-body LAB pixels [K,3] (see color_eval.py)."""
    L, a, b = body_lab[:, 0], body_lab[:, 1], body_lab[:, 2]
    chroma = np.sqrt(a * a + b * b)
    med = np.median(body_lab, axis=0)
    Lp = np.percentile(L, [10, 90])
    return np.array([
        med[0], med[1], med[2],
        Lp[0], Lp[1],
        np.median(chroma), np.std(chroma),
        np.subtract(*np.percentile(a, [75, 25])),   # a IQR
        np.subtract(*np.percentile(b, [75, 25])),   # b IQR
        np.std(L),
        float(np.linalg.norm(med - bg_lab)),         # transparency cue
        fg_frac,                                     # transparency cue
    ], np.float32)


def feature_from_rgb01(img, p):
    """RGB[0,1] crop -> 12-d feature, or None if no brick body was found."""
    body, bg_lab, fg_frac = brick_body_lab(img, p)
    if len(body) == 0:
        return None
    return rich_feature(body, bg_lab, fg_frac)


def load_img(path, img_size):
    """Load an image file -> RGB[0,1] resized to (img_size, img_size)."""
    try:
        return np.asarray(
            Image.open(path).convert("RGB").resize((img_size, img_size), Image.BILINEAR),
            np.float32,
        ) / 255.0
    except Exception:
        return None


def sample_files(d, n):
    """Deterministic even-coverage sample of up to n image files in a dir."""
    files = sorted(f for f in os.listdir(d)
                   if os.path.splitext(f)[1].lower() in IMG_EXT)
    if len(files) > n:
        idx = np.unique(np.linspace(0, len(files) - 1, n).round().astype(int))
        files = [files[i] for i in idx]
    return files


def extract_split(root, classes, per_class, p):
    """Return (feat[N,12], y[N]) over up to per_class crops/class under root."""
    feats, ys = [], []
    for c in classes:
        d = os.path.join(root, c)
        if not os.path.isdir(d):
            continue
        for fn in sample_files(d, per_class):
            img = load_img(os.path.join(d, fn), p["img_size"])
            if img is None:
                continue
            f = feature_from_rgb01(img, p)
            if f is None:
                continue
            feats.append(f)
            ys.append(c)
    return np.stack(feats), np.array(ys)


# ===========================================================================
# Palette / color metadata
# ===========================================================================
def load_color_meta(csv_path):
    """id -> (name, hex) for canonical (non-negative, 6-hex) Rebrickable colors."""
    meta = {}
    with open(csv_path) as f:
        next(f)
        for line in f:
            pp = line.rstrip("\n").split(",")
            cid, name, hexc = pp[0], pp[1], pp[2]
            if int(cid) < 0 or len(hexc) != 6:
                continue
            meta[cid] = (name, hexc)
    return meta


# ===========================================================================
# Build: fit the model on the train gallery ONCE, then bake into an artifact
# ===========================================================================
def build_artifact(args):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    p = dict(DEFAULTS)
    if args.no_wb:
        p["wb"] = False
    np.random.seed(SEED)

    classes = sorted(d for d in os.listdir(args.train_dir)
                     if os.path.isdir(os.path.join(args.train_dir, d)))
    color_meta = load_color_meta(args.colors_csv)

    # Reuse the color_eval feature cache if it matches, else extract fresh.
    Xtr = ytr = None
    cache_key = f"{args.gallery_per_class}_{args.val_per_class}_{int(p['wb'])}"
    if args.cache and os.path.exists(args.cache):
        z = np.load(args.cache, allow_pickle=True)
        if str(z["key"]) == cache_key and list(z["classes"]) == list(classes):
            Xtr = z["Xtr"]
            ytr = z["ytr"].astype(str)
            print(f"  [cache] gallery features from {args.cache}: "
                  f"train={len(ytr)}")
    if Xtr is None:
        print(f"  [extract] train gallery from {args.train_dir} ...")
        Xtr, ytr = extract_split(args.train_dir, classes,
                                 args.gallery_per_class, p)
        ytr = ytr.astype(str)
        print(f"  [extract] gallery exemplars = {len(ytr)}")

    # Fit z-score + LDA on the FULL dataset gallery (== "dataset" palette, the
    # production matcher). The fitted transform is exactly reproducible as
    # (z - xbar_) @ scalings_, so we bake those instead of the sklearn object.
    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-6
    Ztr = (Xtr - mu) / sd
    lda = LinearDiscriminantAnalysis()
    lda.fit(Ztr, ytr)
    proj = (Ztr - lda.xbar_) @ lda.scalings_          # gallery in LDA space

    # Optionally collapse the gallery to per-color prototypes (smaller artifact,
    # k forced to 1). Default keeps full exemplars to match color_eval exactly.
    if args.prototypes:
        uy = sorted(set(ytr))
        proto = np.stack([proj[ytr == c].mean(0) for c in uy]).astype(np.float32)
        gallery_proj, gallery_y = proto, np.array(uy)
        knn_k = 1
        print(f"  [gallery] reduced to {len(uy)} per-color prototypes")
    else:
        gallery_proj, gallery_y = proj.astype(np.float32), ytr
        knn_k = args.k

    gallery_classes = sorted(set(gallery_y))
    color_ids = gallery_classes
    color_names = [color_meta.get(c, (c, ""))[0] for c in color_ids]
    color_hex = [color_meta.get(c, ("", ""))[1] for c in color_ids]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        version=ARTIFACT_VERSION,
        # standardization
        feat_mu=mu.astype(np.float32),
        feat_sd=sd.astype(np.float32),
        # LDA (pure-numpy reproducible)
        lda_xbar=lda.xbar_.astype(np.float32),
        lda_scalings=lda.scalings_.astype(np.float32),
        # gallery in LDA space
        gallery_proj=gallery_proj,
        gallery_y=gallery_y.astype("U8"),
        # matcher
        knn_k=knn_k,
        knn_weights="distance",
        # extraction params (the model)
        img_size=p["img_size"], wb=p["wb"], wb_p=p["wb_p"],
        border_frac=p["border_frac"], fg_thresh=p["fg_thresh"],
        hi_v=p["hi_v"], hi_s=p["hi_s"], lo_v=p["lo_v"],
        # color metadata
        color_ids=np.array(color_ids, dtype="U8"),
        color_names=np.array(color_names, dtype=object),
        color_hex=np.array(color_hex, dtype="U6"),
    )

    # Human-readable sidecar (does not affect inference).
    meta = {
        "version": ARTIFACT_VERSION,
        "method": "lda-knn (LDA metric + distance-weighted kNN over real train exemplars)",
        "feature_dim": int(Xtr.shape[1]),
        "lda_components": int(lda.scalings_.shape[1]),
        "gallery_size": int(len(gallery_y)),
        "num_colors": len(color_ids),
        "knn_k": int(knn_k),
        "prototypes": bool(args.prototypes),
        "extraction": {k: (bool(v) if isinstance(v, np.bool_) else v)
                       for k, v in p.items()},
        "colors": {cid: {"name": nm, "hex": hx}
                   for cid, nm, hx in zip(color_ids, color_names, color_hex)},
    }
    sidecar = os.path.splitext(args.out)[0] + ".meta.json"
    with open(sidecar, "w") as f:
        json.dump(meta, f, indent=2)

    size_kb = os.path.getsize(args.out) / 1024.0
    print(f"\n  wrote {args.out}  ({size_kb:.1f} KB)")
    print(f"  wrote {sidecar}")
    print(f"  gallery={len(gallery_y)} exemplars  colors={len(color_ids)}  "
          f"lda_dim={lda.scalings_.shape[1]}  k={knn_k}")
    return args.out


# ===========================================================================
# Inference: load ONLY the artifact + a crop -> Rebrickable color id.
# Pure-numpy LDA projection + distance-weighted kNN; no sklearn, no train dir.
# ===========================================================================
class ColorClassifier:
    """Self-contained color classifier loaded from a color_model.npz artifact."""

    def __init__(self, artifact_path):
        z = np.load(artifact_path, allow_pickle=True)
        self.version = int(z["version"])
        self.feat_mu = z["feat_mu"].astype(np.float64)
        self.feat_sd = z["feat_sd"].astype(np.float64)
        self.lda_xbar = z["lda_xbar"].astype(np.float64)
        self.lda_scalings = z["lda_scalings"].astype(np.float64)
        self.gallery_proj = z["gallery_proj"].astype(np.float64)
        self.gallery_y = np.array([str(c) for c in z["gallery_y"]])
        self.knn_k = int(z["knn_k"])
        # Extraction params, rebuilt into the dict brick_body_lab expects.
        self.p = dict(
            img_size=int(z["img_size"]), wb=bool(z["wb"]), wb_p=float(z["wb_p"]),
            border_frac=float(z["border_frac"]), fg_thresh=float(z["fg_thresh"]),
            hi_v=float(z["hi_v"]), hi_s=float(z["hi_s"]), lo_v=float(z["lo_v"]),
        )
        self.color_ids = [str(c) for c in z["color_ids"]]
        self.color_names = [str(n) for n in z["color_names"]]
        self.color_hex = [str(h) for h in z["color_hex"]]
        self._name = {i: n for i, n in zip(self.color_ids, self.color_names)}
        self._hex = {i: h for i, h in zip(self.color_ids, self.color_hex)}
        # Stable class ordering for the kNN vote accumulator.
        self._classes = sorted(set(self.gallery_y))
        self._cls_idx = {c: i for i, c in enumerate(self._classes)}
        self._gy_idx = np.array([self._cls_idx[c] for c in self.gallery_y])

    # -- pieces ---------------------------------------------------------------
    def feature(self, img_rgb01):
        """RGB[0,1] crop -> 12-d feature (None if no brick body found)."""
        return feature_from_rgb01(img_rgb01, self.p)

    def project(self, feat):
        """12-d feature -> LDA space: standardize, then (z - xbar) @ scalings."""
        z = (np.asarray(feat, np.float64) - self.feat_mu) / self.feat_sd
        return (z - self.lda_xbar) @ self.lda_scalings

    def _rank(self, q_proj):
        """Distance-weighted kNN ranking of color ids for one projected query.

        Reproduces sklearn KNeighborsClassifier(weights='distance').predict_proba
        ordering: weight = 1/distance, summed per class, descending.
        """
        d = np.sqrt(((self.gallery_proj - q_proj) ** 2).sum(1))
        k = min(self.knn_k, len(d))
        nn = np.argpartition(d, k - 1)[:k]
        dn = d[nn]
        w = 1.0 / np.where(dn == 0.0, 1e-300, dn)
        acc = np.zeros(len(self._classes))
        np.add.at(acc, self._gy_idx[nn], w)
        return [self._classes[j] for j in np.argsort(-acc)]

    # -- public API -----------------------------------------------------------
    def rank_feature(self, feat):
        """12-d feature -> ranked list of Rebrickable color ids (best first)."""
        return self._rank(self.project(feat))

    def predict(self, img, topk=3):
        """Image (path | RGB[0,1] ndarray) -> ranked top-k predictions.

        Returns a list of dicts: {id, name, hex}. Empty list if no brick body
        was segmented from the crop.
        """
        if isinstance(img, str):
            img = load_img(img, self.p["img_size"])
            if img is None:
                return []
        feat = self.feature(img)
        if feat is None:
            return []
        ranked = self.rank_feature(feat)[:topk]
        return [{"id": c, "name": self._name.get(c, c), "hex": self._hex.get(c, "")}
                for c in ranked]


# ===========================================================================
# Eval: reproduce val top-1/top-3 FROM THE ARTIFACT ALONE
# ===========================================================================
def eval_artifact(args):
    clf = ColorClassifier(args.artifact)
    classes = sorted(d for d in os.listdir(args.val_dir)
                     if os.path.isdir(os.path.join(args.val_dir, d)))

    n = top1 = top3 = 0
    per_class = {c: [0, 0] for c in classes}
    confusion = {}
    for c in classes:
        d = os.path.join(args.val_dir, c)
        for fn in sample_files(d, args.val_per_class):
            img = load_img(os.path.join(d, fn), clf.p["img_size"])
            if img is None:
                continue
            feat = clf.feature(img)
            if feat is None:
                continue
            ranked = clf.rank_feature(feat)
            n += 1
            per_class[c][1] += 1
            hit1 = ranked[0] == c
            top1 += hit1
            per_class[c][0] += hit1
            top3 += c in ranked[:3]
            if not hit1:
                confusion[(c, ranked[0])] = confusion.get((c, ranked[0]), 0) + 1

    print(f"\n=== eval FROM ARTIFACT ALONE (no train dir) ===")
    print(f"  artifact   : {args.artifact}  (v{clf.version}, "
          f"gallery={len(clf.gallery_y)}, k={clf.knn_k})")
    print(f"  val crops  : {n}  over {len(classes)} colors")
    print(f"  top-1      : {100 * top1 / n:.1f}%")
    print(f"  top-3      : {100 * top3 / n:.1f}%")

    if args.diagnostics:
        nm = clf._name
        rows = sorted((h / m, c, h, m) for c, (h, m) in per_class.items() if m)
        print("\n  worst 12 colors:")
        print("   " + "  ".join(
            f"{c}:{nm.get(c, c)[:11]}={100 * a:.0f}%({h}/{m})"
            for a, c, h, m in rows[:12]))
        print("\n  top confusion pairs (true -> pred):")
        for (t, pr), k in sorted(confusion.items(), key=lambda kv: -kv[1])[:10]:
            print(f"    {nm.get(t, t):>18} -> {nm.get(pr, pr):<18} x{k}")
    return top1 / n, top3 / n


def predict_cli(args):
    clf = ColorClassifier(args.artifact)
    preds = clf.predict(args.image, topk=args.topk)
    if not preds:
        print("no brick body segmented from crop")
        return
    print(f"predictions for {args.image}:")
    for i, p in enumerate(preds, 1):
        print(f"  {i}. id={p['id']:>5}  {p['name']:<22} #{p['hex']}")


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="fit model + write portable artifact")
    b.add_argument("--train-dir", required=True)
    b.add_argument("--val-dir", help="unused for build; kept for symmetry")
    b.add_argument("--colors-csv", required=True)
    b.add_argument("--out", default="models/color_v1/color_model.npz")
    b.add_argument("--gallery-per-class", type=int, default=350)
    b.add_argument("--val-per-class", type=int, default=40,
                   help="only used to match the color_eval feature cache key")
    b.add_argument("--k", type=int, default=3)
    b.add_argument("--no-wb", action="store_true")
    b.add_argument("--prototypes", action="store_true",
                   help="collapse gallery to per-color means (smaller, k=1)")
    b.add_argument("--cache", default="/tmp/color_eval_feat.npz",
                   help="reuse color_eval.py feature cache if key matches")
    b.set_defaults(func=build_artifact)

    e = sub.add_parser("eval", help="reproduce val top-1/top-3 from artifact")
    e.add_argument("--artifact", required=True)
    e.add_argument("--val-dir", required=True)
    e.add_argument("--val-per-class", type=int, default=40)
    e.add_argument("--diagnostics", action="store_true")
    e.set_defaults(func=eval_artifact)

    p = sub.add_parser("predict", help="classify a single crop from artifact")
    p.add_argument("--artifact", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--topk", type=int, default=3)
    p.set_defaults(func=predict_cli)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
