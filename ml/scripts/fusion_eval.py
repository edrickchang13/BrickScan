#!/usr/bin/env python3
"""Multi-frame FUSION eval for BrickScan live-scan (Phase 0 crux test).

Tests the core thesis: does fusing N views of the SAME piece (as a live
multi-angle sweep would) lift retrieval accuracy over a single frame?

For each group size N, we group query images by class, fuse N views of a
class into one query embedding (mean OR confidence-weighted pooling), then
k-NN against the per-image gallery. N=1 == the single-frame baseline.
Reports overall top-1 per N and the look-alike basic bricks (3001-3005).

Deterministic (seeded). Example:
  python scripts/fusion_eval.py --backbone dinov2 \
      --gallery-dir training_data/real_photos_v3/train \
      --query-dir   training_data/real_photos_v3/val \
      --group-sizes 1,2,4,8,16
"""
import argparse, os
import numpy as np
import torch
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BASIC = ["3001", "3002", "3003", "3004", "3005"]
SEED = 0


def list_images(class_dir, cap):
    files = [class_dir + "/" + f for f in os.listdir(class_dir)
             if os.path.splitext(f)[1].lower() in IMG_EXT]
    files.sort()
    if cap and len(files) > cap:
        idx = np.linspace(0, len(files) - 1, cap).astype(int)
        files = [files[i] for i in idx]
    return files


def gather(root, cap, classes=None):
    cls = sorted([d for d in os.listdir(root) if os.path.isdir(root + "/" + d)])
    if classes is not None:
        cls = [c for c in cls if c in classes]
    items = []
    for c in cls:
        for p in list_images(root + "/" + c, cap):
            items.append((p, c))
    return items, cls


def load_backbone(name, device):
    if name == "dinov2":
        import timm
        model = timm.create_model("vit_base_patch14_dinov2.lvd142m",
                                  pretrained=True, num_classes=0,
                                  dynamic_img_size=True).eval().to(device)
        cfg = timm.data.resolve_data_config({}, model=model)
        mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(device)
        std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(device)
        return (lambda b: model((b - mean) / std)), 224
    if name == "cradio":
        from transformers import AutoModel
        model = AutoModel.from_pretrained("nvidia/C-RADIOv3-B",
                                          trust_remote_code=True).eval().to(device)

        def embed(b):
            out = model(b)
            return out[0] if isinstance(out, (tuple, list)) else out
        return embed, 224
    raise ValueError(name)


def load_image(path, size):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1)


@torch.no_grad()
def embed_items(items, embed_fn, size, device, bs):
    feats, labels, buf, blab = [], [], [], []

    def flush():
        if not buf:
            return
        f = torch.nn.functional.normalize(embed_fn(torch.stack(buf).to(device)).float(), dim=1)
        feats.append(f.cpu()); labels.extend(blab); buf.clear(); blab.clear()

    for p, lbl in items:
        try:
            t = load_image(p, size)
        except Exception:
            continue
        buf.append(t); blab.append(lbl)
        if len(buf) >= bs:
            flush()
    flush()
    return torch.cat(feats), labels


def build_groups(qlabel, N, rng):
    """Disjoint groups of N indices, all same class. Returns list of (label, idx_array)."""
    by_class = {}
    for i, l in enumerate(qlabel):
        by_class.setdefault(l, []).append(i)
    groups = []
    for c, idxs in by_class.items():
        idxs = np.array(idxs); rng.shuffle(idxs)
        n_full = len(idxs) // N
        for g in range(n_full):
            groups.append((c, idxs[g * N:(g + 1) * N]))
    return groups


def evaluate(Q, qlabel, G, glab, N, method, rng):
    groups = build_groups(qlabel, N, rng)
    if not groups:
        return None
    # per-view max similarity to gallery -> confidence weight
    if method == "confweighted" and N > 1:
        maxsim = (Q @ G.T).max(1).values  # (nq,)
    fused, flab = [], []
    for c, idx in groups:
        emb = Q[idx]
        if method == "confweighted" and N > 1:
            w = torch.softmax(maxsim[idx] * 20.0, dim=0).unsqueeze(1)
            v = (emb * w).sum(0)
        else:
            v = emb.mean(0)
        fused.append(torch.nn.functional.normalize(v, dim=0))
        flab.append(c)
    F = torch.stack(fused)
    flab = np.array(flab)
    sims = F @ G.T
    top5 = glab[sims.topk(min(5, sims.shape[1]), dim=1).indices.cpu().numpy()]
    top1 = float((top5[:, 0] == flab).mean())
    r3 = float(np.mean([flab[i] in top5[i, :3] for i in range(len(flab))]))
    per_basic = {}
    for b in BASIC:
        m = flab == b
        if m.sum():
            per_basic[b] = round(100 * float((top5[m, 0] == b).mean()), 1)
    return {"n_groups": len(groups), "top1": top1, "r3": r3, "basic": per_basic}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2", choices=["dinov2", "cradio"])
    ap.add_argument("--gallery-dir", required=True)
    ap.add_argument("--query-dir", required=True)
    ap.add_argument("--gallery-per-class", type=int, default=30)
    ap.add_argument("--query-per-class", type=int, default=40)
    ap.add_argument("--group-sizes", default="1,2,4,8,16")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_fn, size = load_backbone(args.backbone, device)

    gitems, gcls = gather(args.gallery_dir, args.gallery_per_class)
    qitems, _ = gather(args.query_dir, args.query_per_class, classes=set(gcls))
    G, glabel = embed_items(gitems, embed_fn, size, device, args.batch_size)
    Q, qlabel = embed_items(qitems, embed_fn, size, device, args.batch_size)
    G = G.to(device); glab = np.array(glabel)
    Qd = Q.to(device)
    print(f"backbone={args.backbone} gallery={len(glabel)}/{len(gcls)}cls "
          f"query={len(qlabel)} dim={G.shape[1]} device={device}")
    print(f"{'N':>3} {'method':>12} {'groups':>7} {'top1':>7} {'r@3':>7}   basic(3001-3005)")
    Ns = [int(x) for x in args.group_sizes.split(",")]
    for N in Ns:
        for method in (["mean"] if N == 1 else ["mean", "confweighted"]):
            rng = np.random.default_rng(SEED)
            r = evaluate(Qd, qlabel, G, glab, N, method, rng)
            if r:
                print(f"{N:>3} {method:>12} {r['n_groups']:>7} "
                      f"{100*r['top1']:>6.1f}% {100*r['r3']:>6.1f}%   {r['basic']}")


if __name__ == "__main__":
    main()
