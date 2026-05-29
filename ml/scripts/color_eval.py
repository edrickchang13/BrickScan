#!/usr/bin/env python3
"""Color-ID pipeline + eval for BrickScan (Phase 0).

Classifies a brick crop's color by RETRIEVAL against real photographed
exemplars instead of matching to Rebrickable's ideal hex. This is the same
move that took part-ID to ~89%: matching against the real camera/lighting
distribution survives the color shift that canonical-hex DeltaE does not, and
it disambiguates near-duplicate darks/grays (Black vs Pearl Titanium vs Dark
Bluish Gray) that the ideal-hex distance collapses.

Pipeline
--------
1. Extract a robust, achromatic-aware brick color from each crop:
     - shades-of-gray white balance,
     - segment the brick off the (bright, neutral) background,
     - drop specular highlights (blown-out, low-chroma) and deep shadow,
     - summarize the brick body with a rich LAB feature (central color +
       glossy-shade distribution + two transparency cues).
2. Build a gallery of these features from color_v1/TRAIN.
3. Match a val crop with one of:
     canon   : CIEDE2000 to canonical Rebrickable hex            (baseline)
     knn-lab : k-NN on median-LAB vs train exemplars, CIEDE2000 metric
     lda-knn : k-NN after an LDA metric learned on train         (default/best)

Reports top-1/top-3 per palette (full / common / dataset) for each method.

Measured on color_v1/val (60 colors): canon ~39% top-1 (dataset palette);
lda-knn ~84% top-1 / ~91% top-3. RebrickNet incumbent color is ~80%.

  python scripts/color_eval.py \
      --train-dir training_data/color_v1/train \
      --val-dir   training_data/color_v1/val \
      --colors-csv training_data/rebrickable_csv/colors.csv
"""
import argparse
import os

import numpy as np
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COMMON_MIN_PARTS = 3000  # "common" palette = colors used on >=COMMON_MIN_PARTS parts
SEED = 0


# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
def load_palette(csv_path):
    """Return (ids, names, lab[N,3], nparts[N]) for canonical (non-negative) colors."""
    ids, names, labs, nparts = [], [], [], []
    with open(csv_path) as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split(",")
            cid, name, hexc = p[0], p[1], p[2]
            if int(cid) < 0 or len(hexc) != 6:
                continue
            rgb = np.array([int(hexc[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0
            ids.append(cid)
            names.append(name)
            labs.append(rgb2lab(rgb.reshape(1, 1, 3)).reshape(3))
            nparts.append(int(p[4]) if len(p) > 4 and p[4].isdigit() else 0)
    return ids, names, np.stack(labs), np.array(nparts)


# --------------------------------------------------------------------------- #
# Color extraction
# --------------------------------------------------------------------------- #
def shades_of_gray_wb(img, p=6):
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


def brick_body_lab(img, wb=True):
    """Brick-body pixels in CIELAB plus context.

    Returns (body_lab[K,3], bg_lab[3], fg_frac). White-balances, estimates the
    background from a border frame, keeps pixels far from it, then drops
    specular highlights (very bright + low-chroma) and deep shadow. Falls back
    to a center crop when the brick fills the frame or matches the background
    (e.g. a white brick on a white surface). The highlight/shadow guard is
    skipped if it would remove too much, so a genuinely white or black brick
    keeps its body.
    """
    if wb:
        img = shades_of_gray_wb(img)
    h, w, _ = img.shape
    flat = img.reshape(-1, 3)

    b = max(2, int(min(h, w) * 0.10))
    border = np.concatenate([
        img[:b].reshape(-1, 3), img[-b:].reshape(-1, 3),
        img[:, :b].reshape(-1, 3), img[:, -b:].reshape(-1, 3),
    ])
    bg = np.median(border, axis=0)

    dist = np.linalg.norm(flat - bg, axis=1)
    fg_mask = dist > 0.14
    fg_frac = float(fg_mask.mean())
    if fg_mask.sum() < 0.04 * len(flat):
        cy0, cy1, cx0, cx1 = int(h * .30), int(h * .70), int(w * .30), int(w * .70)
        fg = img[cy0:cy1, cx0:cx1].reshape(-1, 3)
    else:
        fg = flat[fg_mask]

    v, s = _v_s(fg)
    keep = ~((v > 0.93) & (s < 0.12)) & ~(v < 0.04)
    if keep.sum() < max(20, int(0.15 * len(fg))):
        keep = np.ones(len(fg), bool)
    body = fg[keep]

    bg_lab = rgb2lab(bg.reshape(1, 1, 3)).reshape(3)
    return rgb2lab(body.reshape(-1, 1, 3)).reshape(-1, 3), bg_lab, fg_frac


def rich_feature(body_lab, bg_lab, fg_frac):
    """12-d color feature from brick-body LAB pixels [K,3].

    Central color (median L,a,b) + the glossy-shade distribution (L 10/90
    percentiles, L std, chroma median/std, a/b IQR) which separates
    near-duplicate achromatics, + two transparency cues (body-to-background LAB
    distance and the foreground fraction) which help the Trans-* colors whose
    bodies bleed the background through.
    """
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


def load_img(path):
    try:
        return np.asarray(
            Image.open(path).convert("RGB").resize((128, 128), Image.BILINEAR),
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


def extract_split(root, classes, per_class, wb):
    """Return (feat[N,12], med_lab[N,3], y[N]) over up to per_class crops/class."""
    feats, medlabs, ys = [], [], []
    for c in classes:
        d = os.path.join(root, c)
        if not os.path.isdir(d):
            continue
        for fn in sample_files(d, per_class):
            img = load_img(os.path.join(d, fn))
            if img is None:
                continue
            body, bg_lab, fg_frac = brick_body_lab(img, wb=wb)
            if len(body) == 0:
                continue
            feats.append(rich_feature(body, bg_lab, fg_frac))
            medlabs.append(np.median(body, axis=0))
            ys.append(c)
    return np.stack(feats), np.stack(medlabs), np.array(ys)


# --------------------------------------------------------------------------- #
# Matchers -> each returns a ranked list of color ids per val crop
# --------------------------------------------------------------------------- #
def rank_canon(val_lab, pal_lab, ids, cand_idx):
    """CIEDE2000 to canonical hex within the candidate palette (baseline)."""
    d = deltaE_ciede2000(val_lab[:, None, :], pal_lab[None, cand_idx, :])
    return [[ids[cand_idx[j]] for j in row] for row in np.argsort(d, axis=1)]


def rank_knn_lab(train_lab, ytr, val_lab, cand_ids, k):
    """Distance-weighted k-NN on median-LAB, CIEDE2000 metric, within candidates."""
    elig = np.isin(ytr, list(cand_ids))
    tl, yl = train_lab[elig], ytr[elig]
    out = []
    for q in val_lab:
        dd = deltaE_ciede2000(np.tile(q, (len(tl), 1)), tl)
        kk = min(k, len(dd))
        nn = np.argpartition(dd, kk - 1)[:kk]
        w = {}
        for lb, dist in zip(yl[nn], dd[nn]):
            w[lb] = w.get(lb, 0.0) + 1.0 / (dist + 1.0)
        out.append([lb for lb, _ in sorted(w.items(), key=lambda kv: -kv[1])])
    return out


def rank_lda_knn(Xtr, ytr, Xva, cand_ids, k):
    """k-NN after an LDA metric learned on train (z-scored rich feature).

    The candidate palette restricts both the LDA fit and the gallery, so a
    smaller palette is a strictly easier problem. Returns ranked ids per crop.
    """
    elig = np.isin(ytr, list(cand_ids))
    Xt, yt = Xtr[elig], ytr[elig]
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    Zt, Zv = (Xt - mu) / sd, (Xva - mu) / sd
    lda = LinearDiscriminantAnalysis()
    Pt = lda.fit_transform(Zt, yt)
    Pv = lda.transform(Zv)
    knn = KNeighborsClassifier(n_neighbors=min(k, len(Zt)), weights="distance")
    knn.fit(Pt, yt)
    proba = knn.predict_proba(Pv)
    return [[knn.classes_[j] for j in np.argsort(-row)] for row in proba]


def topk(rank_ids, y_true, k):
    return float(np.mean([yt in r[:k] for r, yt in zip(rank_ids, y_true)]))


# --------------------------------------------------------------------------- #
# Eval
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--colors-csv", required=True)
    ap.add_argument("--gallery-per-class", type=int, default=350,
                    help="train exemplars per class for the retrieval gallery")
    ap.add_argument("--val-per-class", type=int, default=40)
    ap.add_argument("--k", type=int, default=3, help="neighbors for k-NN / lda-knn")
    ap.add_argument("--knn-lab-k", type=int, default=7)
    ap.add_argument("--no-wb", action="store_true", help="disable white-balance")
    ap.add_argument("--cache", default="/tmp/color_eval_feat.npz",
                    help="npz cache of extracted features (delete to recompute)")
    ap.add_argument("--methods", default="canon,knn-lab,lda-knn")
    args = ap.parse_args()
    wb = not args.no_wb
    np.random.seed(SEED)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    ids, names, pal_lab, nparts = load_palette(args.colors_csv)
    id2i = {c: i for i, c in enumerate(ids)}
    classes = sorted(d for d in os.listdir(args.val_dir)
                     if os.path.isdir(os.path.join(args.val_dir, d)) and d in id2i)
    ds_idx = np.array([id2i[c] for c in classes])
    common_idx = np.where(nparts >= COMMON_MIN_PARTS)[0]
    palettes = {  # name -> (candidate canonical-indices, candidate id set)
        "full":    (np.arange(len(ids)), set(ids)),
        "common":  (common_idx, {ids[i] for i in common_idx}),
        "dataset": (ds_idx, set(classes)),
    }
    print(f"palette full={len(ids)} common={len(common_idx)} dataset={len(ds_idx)} "
          f"| val classes={len(classes)} | wb={wb} k={args.k} "
          f"gallery/cls={args.gallery_per_class}")

    # Extract (cached). Cache key covers everything that changes the features.
    key = f"{args.gallery_per_class}_{args.val_per_class}_{int(wb)}"
    cached = None
    if os.path.exists(args.cache):
        z = np.load(args.cache, allow_pickle=True)
        if str(z["key"]) == key and list(z["classes"]) == list(classes):
            Xtr, Ltr, ytr, Xva, Lva, yva = (z["Xtr"], z["Ltr"], z["ytr"],
                                            z["Xva"], z["Lva"], z["yva"])
            cached = True
            print(f"  [cache] loaded {args.cache} train={len(ytr)} val={len(yva)}")
    if cached is None:
        print("  [extract] train ...")
        Xtr, Ltr, ytr = extract_split(args.train_dir, classes,
                                      args.gallery_per_class, wb)
        print(f"  [extract] val ... (train exemplars={len(ytr)})")
        Xva, Lva, yva = extract_split(args.val_dir, classes, args.val_per_class, wb)
        np.savez(args.cache, Xtr=Xtr, Ltr=Ltr, ytr=ytr, Xva=Xva, Lva=Lva, yva=yva,
                 key=key, classes=np.array(classes))
        print(f"  [extract] cached {args.cache}. train={len(ytr)} val={len(yva)}")

    # Run each method on each palette.
    print("\n=== top-1 / top-3 by method x palette ===")
    ranks_for_diag = None
    for meth in methods:
        for m, (cand_idx, cand_ids) in palettes.items():
            if meth == "canon":
                rank = rank_canon(Lva, pal_lab, ids, cand_idx)
            elif meth == "knn-lab":
                rank = rank_knn_lab(Ltr, ytr, Lva, cand_ids, args.knn_lab_k)
            elif meth == "lda-knn":
                rank = rank_lda_knn(Xtr, ytr, Xva, cand_ids, args.k)
            else:
                raise SystemExit(f"unknown method {meth}")
            print(f"  {meth:>8} palette={m:>7}  "
                  f"top1={100 * topk(rank, yva, 1):5.1f}%  "
                  f"top3={100 * topk(rank, yva, 3):5.1f}%  (n={len(yva)})")
            if meth == "lda-knn" and m == "dataset":
                ranks_for_diag = rank

    # Diagnostics for the production matcher (lda-knn, dataset palette).
    if ranks_for_diag is not None:
        pc = {c: [0, 0] for c in classes}
        conf = {}
        for r, yt in zip(ranks_for_diag, yva):
            pc[yt][1] += 1
            pc[yt][0] += (r[0] == yt)
            if r[0] != yt:
                conf[(yt, r[0])] = conf.get((yt, r[0]), 0) + 1
        rows = sorted((h / n, c, h, n) for c, (h, n) in pc.items() if n)
        print("\n=== worst 12 classes (lda-knn, dataset) ===")
        print("  " + "  ".join(f"{c}:{names[id2i[c]][:11]}={100*a:.0f}%({h}/{n})"
                                for a, c, h, n in rows[:12]))
        print("\n=== top confusion pairs (true -> pred) ===")
        for (t, p), n in sorted(conf.items(), key=lambda kv: -kv[1])[:12]:
            print(f"  {names[id2i[t]]:>18} -> {names[id2i[p]]:<18} x{n}")


if __name__ == "__main__":
    main()
