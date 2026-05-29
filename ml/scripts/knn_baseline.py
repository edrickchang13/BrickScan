#!/usr/bin/env python3
"""Frozen-backbone k-NN baseline for BrickScan part recognition.

Embeds a gallery and a query set of LEGO-part photos with a FROZEN vision
backbone (no training), then measures retrieval accuracy: how often the
nearest neighbour / nearest centroid in embedding space carries the correct
part number. Directly comparable to the trained classifier's held-out
cross-domain top-1.

Backbones:
  dinov2  -> timm vit_base_patch14_dinov2 (ImageNet-normalised input)
  cradio  -> nvidia/C-RADIOv3-B via transformers (raw [0,1] input; has its
             own internal input conditioner)

Example:
  python scripts/knn_baseline.py --backbone dinov2 \
      --gallery-dir training_data/real_photos_v3/train \
      --query-dir   training_data/real_photos_v3/val_xdomain
"""
import argparse, os, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(class_dir, cap):
    files = [class_dir / f for f in os.listdir(class_dir)
             if Path(f).suffix.lower() in IMG_EXT]
    files.sort()
    if cap and len(files) > cap:
        idx = np.linspace(0, len(files) - 1, cap).astype(int)
        files = [files[i] for i in idx]
    return files


def gather(root, cap, classes=None):
    root = Path(root)
    cls = sorted([d.name for d in root.iterdir() if d.is_dir()])
    if classes is not None:
        cls = [c for c in cls if c in classes]
    items = []
    for c in cls:
        for p in list_images(root / c, cap):
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

        def embed(batch):
            return model((batch - mean) / std)
        return embed, 224, "DINOv2 vit_base_patch14"

    if name == "cradio":
        model = None
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained("nvidia/C-RADIOv3-B",
                                              trust_remote_code=True).eval().to(device)
            tag = "C-RADIOv3-B (transformers)"
        except Exception as e:
            print(f"  [cradio] transformers load failed ({e}); trying torch.hub...")
            model = torch.hub.load("NVlabs/RADIO", "radio_model",
                                   version="c-radio_v3-b", progress=True,
                                   skip_validation=True).eval().to(device)
            tag = "C-RADIOv3-B (torch.hub)"

        def embed(batch):  # C-RADIO expects [0,1]; normalises internally
            out = model(batch)
            if isinstance(out, (tuple, list)):
                return out[0]
            for attr in ("summary", "pooler_output", "last_hidden_state"):
                if hasattr(out, attr):
                    v = getattr(out, attr)
                    return v.mean(1) if v.dim() == 3 else v
            return out
        return embed, 224, tag

    raise ValueError(name)


def load_image(path, size):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


@torch.no_grad()
def embed_items(items, embed_fn, size, device, bs):
    feats, out_labels, buf, buf_lbl = [], [], [], []

    def flush():
        if not buf:
            return
        x = torch.stack(buf).to(device)
        f = embed_fn(x).float()
        f = torch.nn.functional.normalize(f, dim=1)
        feats.append(f.cpu())
        out_labels.extend(buf_lbl)
        buf.clear(); buf_lbl.clear()

    for p, lbl in items:
        try:
            t = load_image(p, size)
        except Exception:
            continue
        buf.append(t); buf_lbl.append(lbl)
        if len(buf) >= bs:
            flush()
    flush()
    if not feats:
        return torch.empty(0), []
    return torch.cat(feats), out_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["dinov2", "cradio"])
    ap.add_argument("--gallery-dir", required=True)
    ap.add_argument("--query-dir", required=True)
    ap.add_argument("--gallery-per-class", type=int, default=30)
    ap.add_argument("--query-per-class", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    embed_fn, size, desc = load_backbone(args.backbone, device)

    gitems, gcls = gather(args.gallery_dir, args.gallery_per_class)
    qitems, qcls = gather(args.query_dir, args.query_per_class, classes=set(gcls))
    print(f"[{desc}] gallery {len(gitems)} imgs / {len(gcls)} classes | "
          f"query {len(qitems)} imgs / {len(qcls)} classes | {device}")

    G, glabel = embed_items(gitems, embed_fn, size, device, args.batch_size)
    Q, qlabel = embed_items(qitems, embed_fn, size, device, args.batch_size)
    dim = G.shape[1]
    G, Q = G.to(device), Q.to(device)
    glab, qlab = np.array(glabel), np.array(qlabel)

    sims = Q @ G.T
    topk = min(5, sims.shape[1])
    _, idx = sims.topk(topk, dim=1)
    nn_labels = glab[idx.cpu().numpy()]
    top1 = float((nn_labels[:, 0] == qlab).mean())

    def recall_at(k):
        k = min(k, topk)
        return float(np.mean([qlab[i] in nn_labels[i, :k] for i in range(len(qlab))]))

    uniq = sorted(set(glabel))
    cent = torch.stack([G[[j for j, l in enumerate(glabel) if l == c]].mean(0)
                        for c in uniq])
    cent = torch.nn.functional.normalize(cent, dim=1)
    cpred = np.array(uniq)[(Q @ cent.T).argmax(1).cpu().numpy()]
    cent_top1 = float((cpred == qlab).mean())

    perclass = {}
    for c in sorted(set(qlabel)):
        m = qlab == c
        perclass[c] = round(100 * float((nn_labels[m, 0] == c).mean()), 1)

    print(f"  embed_dim={dim}")
    print(f"  NN  top-1={100*top1:.1f}%  recall@3={100*recall_at(3):.1f}%  "
          f"recall@5={100*recall_at(5):.1f}%")
    print(f"  Centroid top-1={100*cent_top1:.1f}%")
    print(f"  per-class NN top-1: {perclass}")
    print(f"  wall={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
