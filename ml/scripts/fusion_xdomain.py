#!/usr/bin/env python3
"""CROSS-DOMAIN multi-frame FUSION eval for BrickScan live-scan (Phase 0).

Companion to fusion_eval.py. That script's fusion lift is IN-DOMAIN: query
and gallery are drawn from the same pool (real_photos_v3/val vs /train), so
a fused query and its gallery neighbours share capture conditions. This
script asks the harder question: does the fusion lift SURVIVE a domain shift?

real_photos_v3 images are symlinks; the true capture source is encoded in
the symlink TARGET path (the filenames are uniformly prefixed "pri_", so the
filename prefix carries no source signal). Resolvable sources:
  nature          -> training_data/nature_2023_real        (Nature-Studio real photos)
  kaggle_images   -> data/kaggle/lego-brick-images          (rendered brick images)
  kaggle_sorting  -> data/kaggle/lego-brick-sorting-...      (joosthazelzet sorting photos)
  cdn             -> data/rebrickable_cdn                    (catalog CDN renders)

We hold out ONE source as the QUERY domain and build the GALLERY from ALL
OTHER sources -- a genuine train/test domain gap. Classes are restricted to
those present in BOTH query and gallery (others are unscorable). For each
group size N we fuse N held-out-source views of a class (mean OR
confidence-weighted, identical to fusion_eval) and k-NN against the
per-image cross-domain gallery. N=1 is the single-frame cross-domain
baseline. Reports overall top-1 / recall@3 per N and basic bricks 3001-3005.

Embedding + fusion logic is imported from fusion_eval to stay in lockstep.
Deterministic (seeded). Example:
  python scripts/fusion_xdomain.py --backbone dinov2 \
      --root training_data/real_photos_v3/val \
      --held-out nature --group-sizes 1,2,4,8
"""
import argparse
import os

import numpy as np
import torch

from fusion_eval import (
    IMG_EXT,
    SEED,
    embed_items,
    evaluate,
    load_backbone,
)

# Map a resolved symlink target (or plain path) to a capture-source label.
# Order matters only in that each rule is a disjoint substring test.
SOURCE_RULES = [
    ("nature", "nature_2023_real"),
    ("kaggle_images", "/kaggle/lego-brick-images"),
    ("kaggle_sorting", "/kaggle/lego-brick-sorting"),
    ("cdn", "rebrickable_cdn"),
]


def source_of(path):
    """Capture source for an image symlink, from its TARGET path.

    Returns None for broken symlinks (missing target) and for targets that
    match no known source rule -- both are excluded from the eval.
    """
    if os.path.islink(path):
        if not os.path.exists(path):  # follows link; target must exist
            return None
        target = os.readlink(path)
    else:
        if not os.path.exists(path):
            return None
        target = path
    for label, needle in SOURCE_RULES:
        if needle in target:
            return label
    return None


def cap_per_class(items, cap, rng):
    """Subsample to <=cap items per class, deterministically.

    Mirrors fusion_eval.list_images' intent (bound per-class count) but
    operates after source partitioning so cross-domain depth is preserved.
    """
    if not cap:
        return items
    by_class = {}
    for it in items:
        by_class.setdefault(it[1], []).append(it)
    out = []
    for c in sorted(by_class):
        group = by_class[c]
        if len(group) > cap:
            idx = rng.permutation(len(group))[:cap]
            group = [group[i] for i in sorted(idx)]
        out.extend(group)
    return out


def gather_by_source(root, held_out):
    """Walk root once; split images into query (held_out src) vs gallery.

    Returns (query_items, gallery_items, stats) where *_items are lists of
    (path, class) and stats summarises source/class coverage. Both lists are
    restricted to classes present in BOTH partitions; unscorable query
    classes (no gallery entry) and gallery-only classes are dropped.
    """
    q_raw, g_raw = [], []
    src_counts = {}
    broken = 0
    for c in sorted(os.listdir(root)):
        cdir = os.path.join(root, c)
        if not os.path.isdir(cdir):
            continue
        for f in os.listdir(cdir):
            if os.path.splitext(f)[1].lower() not in IMG_EXT:
                continue
            p = os.path.join(cdir, f)
            s = source_of(p)
            if s is None:
                broken += 1
                continue
            src_counts[s] = src_counts.get(s, 0) + 1
            (q_raw if s == held_out else g_raw).append((p, c))

    q_classes = {c for _, c in q_raw}
    g_classes = {c for _, c in g_raw}
    shared = q_classes & g_classes
    query = sorted([it for it in q_raw if it[1] in shared])
    gallery = sorted([it for it in g_raw if it[1] in shared])
    stats = {
        "src_counts": src_counts,
        "broken": broken,
        "q_classes": len(q_classes),
        "g_classes": len(g_classes),
        "shared": len(shared),
        "q_only_dropped": len(q_classes - shared),
    }
    return query, gallery, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2", choices=["dinov2", "cradio"])
    ap.add_argument("--root", default="training_data/real_photos_v3/val",
                    help="dir of class subdirs holding source-prefixed symlinks")
    ap.add_argument("--held-out", default="nature",
                    help="source used as the cross-domain QUERY; gallery = all others. "
                         "'all' loops over every resolvable source.")
    ap.add_argument("--gallery-per-class", type=int, default=30)
    ap.add_argument("--query-per-class", type=int, default=40)
    ap.add_argument("--group-sizes", default="1,2,4,8")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_fn, size = load_backbone(args.backbone, device)
    Ns = [int(x) for x in args.group_sizes.split(",")]

    if args.held_out == "all":
        held_list = [lbl for lbl, _ in SOURCE_RULES]
    else:
        held_list = [args.held_out]

    for held in held_list:
        cap_rng = np.random.default_rng(SEED)
        query, gallery, st = gather_by_source(args.root, held)
        query = cap_per_class(query, args.query_per_class, cap_rng)
        gallery = cap_per_class(gallery, args.gallery_per_class, cap_rng)
        if not query or not gallery:
            print(f"\n### held-out={held}: no usable query/gallery "
                  f"(src_counts={st['src_counts']}) -- skipping")
            continue

        G, glabel = embed_items(gallery, embed_fn, size, device, args.batch_size)
        Q, qlabel = embed_items(query, embed_fn, size, device, args.batch_size)
        G = G.to(device)
        glab = np.array(glabel)
        Qd = Q.to(device)
        gallery_srcs = sorted(s for s in st["src_counts"] if s != held)

        print(f"\n### CROSS-DOMAIN  held-out(query)={held}  gallery={'+'.join(gallery_srcs)}")
        print(f"backbone={args.backbone} device={device} dim={G.shape[1]}")
        print(f"shared_classes={st['shared']} (query-source had {st['q_classes']}; "
              f"dropped {st['q_only_dropped']} not in gallery) "
              f"broken_links_skipped={st['broken']}")
        print(f"gallery_imgs={len(glabel)} query_imgs={len(qlabel)} "
              f"src_counts={st['src_counts']}")
        print(f"{'N':>3} {'method':>12} {'groups':>7} {'top1':>7} {'r@3':>7}   basic(3001-3005)")
        for N in Ns:
            for method in (["mean"] if N == 1 else ["mean", "confweighted"]):
                rng = np.random.default_rng(SEED)
                r = evaluate(Qd, qlabel, G, glab, N, method, rng)
                if r:
                    print(f"{N:>3} {method:>12} {r['n_groups']:>7} "
                          f"{100 * r['top1']:>6.1f}% {100 * r['r3']:>6.1f}%   {r['basic']}")


if __name__ == "__main__":
    main()
