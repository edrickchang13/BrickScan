#!/usr/bin/env python3
"""Finish Phase 2: embed gallery+query with the distilled STUDENT ONNX, build the
on-device gallery_index.json (rn-retrieval partIndex format), and report the
authoritative single-frame + fused top-1. Standalone (onnxruntime + numpy) —
run when the trainer didn't emit the index/eval itself.

  python scripts/student_index_eval.py \
     --student output/student_fastvit_sa24_20260529_025042/student.onnx \
     --gallery-dir training_data/real_photos_v3/train \
     --query-dir   training_data/real_photos_v3/val \
     --out-index   output/student_fastvit_sa24_20260529_025042/gallery_index.json
"""
import argparse, os, json, base64
import numpy as np
from PIL import Image
import onnxruntime as ort

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BASIC = ["3001", "3002", "3003", "3004", "3005"]
SEED = 0


def list_images(d, cap):
    fs = [d + "/" + f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in IMG_EXT]
    fs.sort()
    if cap and len(fs) > cap:
        idx = np.linspace(0, len(fs) - 1, cap).astype(int)
        fs = [fs[i] for i in idx]
    return fs


def gather(root, cap, classes=None):
    cls = sorted([d for d in os.listdir(root) if os.path.isdir(root + "/" + d)])
    if classes is not None:
        cls = [c for c in cls if c in classes]
    items = []
    for c in cls:
        for p in list_images(root + "/" + c, cap):
            items.append((p, c))
    return items, cls


def load_img(path, size=224):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    a = np.asarray(img, np.float32) / 255.0           # student bakes ImageNet norm internally
    return np.transpose(a, (2, 0, 1))                  # CHW, [0,1]


def embed_all(items, sess, iname, oname, bs):
    feats, labels, buf, blab = [], [], [], []

    def flush():
        if not buf:
            return
        x = np.stack(buf).astype(np.float32)
        o = sess.run([oname], {iname: x})[0].astype(np.float32)
        o = o / (np.linalg.norm(o, axis=1, keepdims=True) + 1e-9)
        feats.append(o); labels.extend(blab); buf.clear(); blab.clear()

    for p, l in items:
        try:
            t = load_img(p)
        except Exception:
            continue
        buf.append(t); blab.append(l)
        if len(buf) >= bs:
            flush()
    flush()
    return np.concatenate(feats), np.array(labels)


def build_groups(qlab, N, rng):
    by = {}
    for i, l in enumerate(qlab):
        by.setdefault(l, []).append(i)
    groups = []
    for c, idxs in by.items():
        idxs = np.array(idxs); rng.shuffle(idxs)
        for g in range(len(idxs) // N):
            groups.append((c, idxs[g * N:(g + 1) * N]))
    return groups


def softmax(x):
    e = np.exp(x - x.max())
    return e / (e.sum() + 1e-9)


def evaluate(Q, qlab, G, glab, N, method, rng):
    groups = build_groups(qlab, N, rng)
    if not groups:
        return None
    maxsim = (Q @ G.T).max(1) if (method == "confweighted" and N > 1) else None
    fused, flab = [], []
    for c, idx in groups:
        emb = Q[idx]
        if maxsim is not None:
            w = softmax(maxsim[idx] * 20.0)[:, None]
            v = (emb * w).sum(0)
        else:
            v = emb.mean(0)
        fused.append(v / (np.linalg.norm(v) + 1e-9)); flab.append(c)
    F = np.stack(fused); flab = np.array(flab)
    sims = F @ G.T
    top5 = glab[np.argsort(-sims, axis=1)[:, :5]]
    top1 = float((top5[:, 0] == flab).mean())
    r3 = float(np.mean([flab[i] in top5[i, :3] for i in range(len(flab))]))
    basic = {b: round(100 * float((top5[flab == b, 0] == b).mean()), 1)
             for b in BASIC if (flab == b).any()}
    return {"n": len(groups), "top1": top1, "r3": r3, "basic": basic}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--gallery-dir", required=True)
    ap.add_argument("--query-dir", required=True)
    ap.add_argument("--out-index", required=True)
    ap.add_argument("--gallery-per-class", type=int, default=30)
    ap.add_argument("--query-per-class", type=int, default=20)
    args = ap.parse_args()

    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(args.student, providers=prov)
    iname = sess.get_inputs()[0].name
    oname = sess.get_outputs()[0].name
    bshape = sess.get_inputs()[0].shape[0]
    bs = 1 if isinstance(bshape, int) and bshape == 1 else 64
    print(f"student={args.student} provider={sess.get_providers()[0]} in={iname} out={oname} bs={bs}")

    gitems, gcls = gather(args.gallery_dir, args.gallery_per_class)
    qitems, _ = gather(args.query_dir, args.query_per_class, classes=set(gcls))
    G, glab = embed_all(gitems, sess, iname, oname, bs)
    Q, qlab = embed_all(qitems, sess, iname, oname, bs)
    dim = G.shape[1]
    print(f"gallery={len(glab)}/{len(gcls)}cls  query={len(qlab)}  dim={dim}")

    # --- build gallery_index.json (single global int8 scale = 1/127) ---
    q8 = np.clip(np.round(G * 127.0), -127, 127).astype(np.int8)
    index = {
        "version": 1, "dim": int(dim), "count": int(G.shape[0]),
        "scale": 1.0 / 127.0,
        "vectors": base64.b64encode(q8.tobytes()).decode("ascii"),
        "partNums": glab.tolist(),
    }
    with open(args.out_index, "w") as f:
        json.dump(index, f)
    print(f"wrote {args.out_index}  ({os.path.getsize(args.out_index)//1024} KB, {G.shape[0]} vectors)")

    # --- authoritative eval (int8 ranking == float since global scale) ---
    print(f"{'N':>3} {'method':>12} {'groups':>7} {'top1':>7} {'r@3':>7}   basic")
    for N in [1, 2, 4, 8]:
        for method in (["mean"] if N == 1 else ["mean", "confweighted"]):
            r = evaluate(Q, qlab, G, glab, N, method, np.random.default_rng(SEED))
            if r:
                print(f"{N:>3} {method:>12} {r['n']:>7} {100*r['top1']:>6.1f}% {100*r['r3']:>6.1f}%   {r['basic']}")


if __name__ == "__main__":
    main()
