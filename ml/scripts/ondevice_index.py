#!/usr/bin/env python3
"""On-device retrieval mechanics: int8-quantized k-NN over a gallery embedding index.

Phase-1 proof for brickscan-livescan. Interface contract (fixed):
    RGB 224x224  ->  L2-normalized float embedding (dim TBD by the student model)
    catalog index = gallery images embedded by that model
    retrieval     = cosine k-NN

This script validates the *mechanics* against the existing FROZEN DINOv2 embeddings
as a stand-in. They apply to any model with the interface above, so when the
distilled student lands we just swap the backbone and re-run.

What it proves:
  1. Builds a gallery embedding index from real_photos_v3/train (DINOv2, reused
     from knn_baseline.py).
  2. Quantizes embeddings to int8 two ways:
       - USearch (mobile-deployable; ships Swift/iOS + ObjC bindings) with i8 scalar
         quantization and cosine metric.
       - A transparent pure-numpy symmetric int8 proof (per-vector or global scale)
         so the accuracy delta is reproducible without trusting a library internal.
  3. Shows int8 k-NN top-1 ~= float top-1 on the 439-class val set.
  4. Reports: float-vs-int8 top-1/recall delta, on-disk index size (MB) for 439
     classes, and projected size at ~10k and ~76k parts (x exemplars/part).

Embeddings are cached to --cache-dir (npz) so re-runs are fast and seeded.

Example (on the Spark):
    python scripts/ondevice_index.py \
        --gallery-dir training_data/real_photos_v3/train \
        --query-dir   training_data/real_photos_v3/val \
        --gallery-per-class 16 --query-per-class 8 --k 5
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

# Reuse the exact backbone + embedding path from the k-NN baseline so the
# float reference here is identical to the reported baseline numbers.
from knn_baseline import gather, load_backbone, embed_items  # noqa: E402

SEED = 1234


# ──────────────────────────────────────────────────────────────────────────────
# Embedding (cached)
# ──────────────────────────────────────────────────────────────────────────────

def _cache_key(tag, split_dir, per_class):
    safe = tag.replace(" ", "_").replace("/", "_")
    leaf = Path(split_dir).name
    return f"emb_{safe}_{leaf}_pc{per_class}.npz"


def embed_split(split_dir, per_class, embed_fn, size, device, bs, desc,
                cache_dir, classes=None):
    """Return (embeddings float32 [N,D] L2-normalized, labels str[N], classes list)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / _cache_key(desc, split_dir, per_class)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        feats = z["feats"].astype(np.float32)
        labels = z["labels"].astype(str)
        cls = list(z["classes"].astype(str))
        if classes is not None:
            keep = np.array([l in classes for l in labels])
            feats, labels = feats[keep], labels[keep]
            cls = [c for c in cls if c in classes]
        print(f"  [cache] {cache.name}: {feats.shape[0]} vecs / {len(cls)} classes")
        return feats, labels, cls

    items, cls = gather(split_dir, per_class, classes=classes)
    feats_t, labels = embed_items(items, embed_fn, size, device, bs)
    feats = feats_t.cpu().numpy().astype(np.float32)
    labels = np.array(labels, dtype=object).astype(str)
    np.savez_compressed(cache, feats=feats, labels=labels,
                        classes=np.array(cls, dtype=object))
    print(f"  [embed] {cache.name}: {feats.shape[0]} vecs / {len(cls)} classes "
          f"(cached)")
    return feats, labels, cls


# ──────────────────────────────────────────────────────────────────────────────
# Pure-numpy int8 symmetric quantization proof
# ──────────────────────────────────────────────────────────────────────────────

def quantize_int8(vecs, mode="global"):
    """Symmetric int8 quantization of L2-normalized vectors. Dequant: q * scale.

    Only "global" (a single scale of 1/127 for the whole matrix) is correct for
    cosine retrieval: L2-normalized components are already in [-1,1], and a shared
    scale keeps int8 dot-product ranking identical to cosine ranking.

    A tempting alternative — per-vector scaling (scale = max|component| / 127 per
    row) — is intentionally NOT supported: it gives each gallery vector its own
    scale, so int8 dot products no longer rank-correspond to cosine across the
    gallery. Measured cost: 439-class top-1 collapses ~85% -> ~64%. This is the
    single biggest footgun for the iOS implementation; see ONDEVICE_NOTES.md.
    """
    if mode == "global":
        scale = np.float32(1.0 / 127.0)
        q = np.clip(np.round(vecs / scale), -127, 127).astype(np.int8)
        return q, scale
    raise ValueError(f"unsupported mode {mode!r} (only 'global' is correct here)")


def int8_knn_topk(gallery_q, query_q, k):
    """Cosine-equivalent k-NN using int8 dot products in int32 accumulation.

    With a single GLOBAL scale, score(q,g) = (query_q . gallery_q) * s_q * s_g and
    the scalar factors are constant across the whole gallery, so ranking on the raw
    int32 dot products is identical to ranking on cosine. We accumulate in int32 to
    mirror an ARM SDOT kernel on-device. (This is exact-search over int8 codes — no
    graph approximation — so it isolates pure quantization error.)
    """
    gq = gallery_q.astype(np.int32)
    out_idx = np.empty((query_q.shape[0], k), dtype=np.int64)
    # Chunk queries to bound memory: [Bq, Ng] int32 scores per chunk.
    B = 2048
    for s in range(0, query_q.shape[0], B):
        e = min(s + B, query_q.shape[0])
        scores = query_q[s:e].astype(np.int32) @ gq.T          # [b, Ng] int32
        part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        # order the top-k within each row
        row = np.arange(e - s)[:, None]
        order = np.argsort(-scores[row, part], axis=1)
        out_idx[s:e] = part[row, order]
    return out_idx


def asym_knn_topk(gallery_q, gallery_scale, query_float, k):
    """Asymmetric: int8 gallery (stored on device) vs FLOAT query (fresh from model).

    This is the realistic on-device path — the query embedding is produced live by
    the model in float and never round-tripped through int8. score = query_float .
    (gallery_q * gallery_scale). With a global gallery_scale this is monotonic in
    query_float . gallery_q, so we can keep the gallery in int8 and accumulate the
    mixed product as float32.
    """
    gq = (gallery_q.astype(np.float32) * np.float32(gallery_scale))   # dequant view
    out_idx = np.empty((query_float.shape[0], k), dtype=np.int64)
    B = 2048
    for s in range(0, query_float.shape[0], B):
        e = min(s + B, query_float.shape[0])
        scores = query_float[s:e] @ gq.T
        part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        row = np.arange(e - s)[:, None]
        order = np.argsort(-scores[row, part], axis=1)
        out_idx[s:e] = part[row, order]
    return out_idx


# ──────────────────────────────────────────────────────────────────────────────
# Float reference k-NN (matches knn_baseline)
# ──────────────────────────────────────────────────────────────────────────────

def float_knn_topk(gallery, query, k):
    out_idx = np.empty((query.shape[0], k), dtype=np.int64)
    B = 2048
    for s in range(0, query.shape[0], B):
        e = min(s + B, query.shape[0])
        sims = query[s:e] @ gallery.T
        part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        row = np.arange(e - s)[:, None]
        order = np.argsort(-sims[row, part], axis=1)
        out_idx[s:e] = part[row, order]
    return out_idx


def topk_metrics(nn_idx, glab, qlab, ks=(1, 3, 5)):
    nn_labels = glab[nn_idx]
    m = {}
    for k in ks:
        kk = min(k, nn_idx.shape[1])
        if k == 1:
            m["top1"] = float((nn_labels[:, 0] == qlab).mean())
        else:
            m[f"recall@{k}"] = float(np.mean(
                [qlab[i] in nn_labels[i, :kk] for i in range(len(qlab))]))
    return m, nn_labels


# ──────────────────────────────────────────────────────────────────────────────
# USearch int8 index
# ──────────────────────────────────────────────────────────────────────────────

def usearch_build_and_query(gallery, query, k, on_disk_path=None):
    """Build a USearch i8 cosine index, return (top-k idx, build_s, query_s, bytes).

    USearch is the mobile-deployable target: it ships Swift Package + ObjC headers
    for iOS and stores i8-quantized vectors natively. We use ScalarKind.I8 so the
    on-disk/on-device footprint is 1 byte/dim.
    """
    try:
        from usearch.index import Index, ScalarKind, MetricKind
    except Exception as e:           # pragma: no cover
        return None, None, None, None, f"usearch import failed: {e}"

    dim = gallery.shape[1]
    idx = Index(ndim=dim, metric=MetricKind.Cos, dtype=ScalarKind.I8)
    keys = np.arange(gallery.shape[0], dtype=np.int64)

    t0 = time.time()
    idx.add(keys, gallery.astype(np.float32))   # USearch quantizes to i8 internally
    build_s = time.time() - t0

    t0 = time.time()
    matches = idx.search(query.astype(np.float32), k)
    query_s = time.time() - t0
    nn_idx = np.asarray(matches.keys, dtype=np.int64).reshape(query.shape[0], -1)
    if nn_idx.shape[1] < k:          # pad if fewer than k returned
        pad = np.full((nn_idx.shape[0], k - nn_idx.shape[1]), nn_idx[:, -1:][:, 0:1])
        nn_idx = np.concatenate([nn_idx, pad], axis=1)

    nbytes = None
    if on_disk_path is not None:
        idx.save(str(on_disk_path))
        nbytes = os.path.getsize(on_disk_path)
    return nn_idx[:, :k], build_s, query_s, nbytes, None


# ──────────────────────────────────────────────────────────────────────────────
# Size accounting
# ──────────────────────────────────────────────────────────────────────────────

def index_size_report(n_vectors, dim, usearch_bytes=None):
    """Bytes for raw float32 vs raw int8 vector payload (excludes HNSW graph)."""
    f32 = n_vectors * dim * 4
    i8 = n_vectors * dim * 1
    out = {
        "n_vectors": int(n_vectors),
        "dim": int(dim),
        "float32_MB": round(f32 / 1e6, 2),
        "int8_MB": round(i8 / 1e6, 2),
    }
    if usearch_bytes is not None:
        out["usearch_i8_file_MB"] = round(usearch_bytes / 1e6, 2)
        out["usearch_overhead_vs_raw_i8"] = round(usearch_bytes / max(i8, 1), 2)
    return out


def project_sizes(dim, exemplars_per_part, parts_list, bytes_per_vec_observed=None):
    """Project raw int8 payload (and observed-per-vec if given) at catalog scales."""
    rows = []
    for parts in parts_list:
        n = parts * exemplars_per_part
        row = {
            "parts": parts,
            "exemplars_per_part": exemplars_per_part,
            "n_vectors": n,
            "raw_int8_MB": round(n * dim * 1 / 1e6, 2),
            "raw_float32_MB": round(n * dim * 4 / 1e6, 2),
        }
        if bytes_per_vec_observed is not None:
            row["usearch_i8_proj_MB"] = round(n * bytes_per_vec_observed / 1e6, 2)
        rows.append(row)
    return rows


# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2", choices=["dinov2", "cradio"])
    ap.add_argument("--gallery-dir", default="training_data/real_photos_v3/train")
    ap.add_argument("--query-dir", default="training_data/real_photos_v3/val")
    ap.add_argument("--gallery-per-class", type=int, default=16)
    ap.add_argument("--query-per-class", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--cache-dir", default="output/ondevice_cache")
    ap.add_argument("--index-out", default="output/ondevice_cache/gallery_i8.usearch")
    ap.add_argument("--exemplars-per-part", type=int, default=4,
                    help="exemplars/part assumed for catalog-scale projections")
    ap.add_argument("--proj-parts", type=int, nargs="+", default=[10000, 76000])
    args = ap.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t_all = time.time()
    embed_fn, size, desc = load_backbone(args.backbone, device)
    print(f"[{desc}] device={device} seed={SEED}")

    G, glab, gcls = embed_split(args.gallery_dir, args.gallery_per_class,
                                embed_fn, size, device, args.batch_size, desc,
                                args.cache_dir)
    Q, qlab, _ = embed_split(args.query_dir, args.query_per_class,
                             embed_fn, size, device, args.batch_size, desc,
                             args.cache_dir, classes=set(gcls))
    dim = G.shape[1]
    print(f"  gallery={G.shape} query={Q.shape} dim={dim} classes={len(gcls)}")

    # Re-normalize defensively (cache should already be normalized).
    G /= np.linalg.norm(G, axis=1, keepdims=True).clip(1e-8)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True).clip(1e-8)

    # ---- 1. Float reference ----
    fidx = float_knn_topk(G, Q, args.k)
    fm, _ = topk_metrics(fidx, glab, qlab)
    print(f"\n[float fp32]  top1={100*fm['top1']:.2f}%  "
          f"recall@3={100*fm['recall@3']:.2f}%  recall@5={100*fm['recall@5']:.2f}%")

    # ---- 2. Pure-numpy int8 proofs ----
    # We use a single GLOBAL scale (1/127): for L2-normalized vectors every
    # component is already in [-1,1], so global is near-optimal AND preserves
    # cosine ranking. (Per-vector scaling would break cross-gallery ranking and
    # is intentionally not used — see docstring on int8_knn_topk.)
    results = {"float": fm}
    gq, gs = quantize_int8(G, "global")
    qq, qs = quantize_int8(Q, "global")

    # 2a. Symmetric int8 (gallery int8 + query int8) — exact search over codes.
    sym_idx = int8_knn_topk(gq, qq, args.k)
    sm, _ = topk_metrics(sym_idx, glab, qlab)
    sm_agree = float((sym_idx[:, 0] == fidx[:, 0]).mean())
    results["int8_numpy_sym_global"] = {**sm, "top1_agree_with_float": sm_agree}
    print(f"[int8 sym (q+g i8) ] top1={100*sm['top1']:.2f}%  "
          f"recall@5={100*sm['recall@5']:.2f}%  "
          f"(d={100*(sm['top1']-fm['top1']):+.2f}pp, agree@1={100*sm_agree:.1f}%)")

    # 2b. Asymmetric (gallery int8 on device + FLOAT query from model) — realistic.
    asym_idx = asym_knn_topk(gq, gs, Q, args.k)
    am, _ = topk_metrics(asym_idx, glab, qlab)
    am_agree = float((asym_idx[:, 0] == fidx[:, 0]).mean())
    results["int8_numpy_asym_global"] = {**am, "top1_agree_with_float": am_agree}
    print(f"[int8 asym (f q,i8 g)] top1={100*am['top1']:.2f}%  "
          f"recall@5={100*am['recall@5']:.2f}%  "
          f"(d={100*(am['top1']-fm['top1']):+.2f}pp, agree@1={100*am_agree:.1f}%)")

    # ---- 3. USearch i8 index (mobile-deployable target) ----
    uidx, ub, uq, ubytes, uerr = usearch_build_and_query(G, Q, args.k, args.index_out)
    if uerr:
        print(f"[usearch] SKIPPED: {uerr}")
        results["usearch"] = {"error": uerr}
    else:
        um, _ = topk_metrics(uidx, glab, qlab)
        agree = float((uidx[:, 0] == fidx[:, 0]).mean())
        # Separate HNSW approximation error from quantization error: how often does
        # USearch's top-1 match the EXACT int8 top-1 (sym)?
        hnsw_agree = float((uidx[:, 0] == sym_idx[:, 0]).mean())
        results["usearch_i8"] = {**um, "top1_agree_with_float": agree,
                                 "top1_agree_with_exact_int8": hnsw_agree,
                                 "build_s": round(ub, 3), "query_s": round(uq, 4),
                                 "file_bytes": int(ubytes) if ubytes else None}
        print(f"[usearch i8 (HNSW) ] top1={100*um['top1']:.2f}%  "
              f"recall@5={100*um['recall@5']:.2f}%  "
              f"(d={100*(um['top1']-fm['top1']):+.2f}pp, agree@1_float={100*agree:.1f}%, "
              f"agree@1_exact_i8={100*hnsw_agree:.1f}%)  "
              f"build={ub:.2f}s search={uq:.3f}s")

    # ---- 4. Size accounting ----
    size_439 = index_size_report(G.shape[0], dim, ubytes if not uerr else None)
    bytes_per_vec = (ubytes / G.shape[0]) if (not uerr and ubytes) else None
    proj = project_sizes(dim, args.exemplars_per_part, args.proj_parts, bytes_per_vec)

    print(f"\n[index size @ {G.shape[0]} vecs / 439 classes]")
    print(f"  raw float32 = {size_439['float32_MB']} MB   "
          f"raw int8 = {size_439['int8_MB']} MB"
          + (f"   usearch i8 file = {size_439.get('usearch_i8_file_MB')} MB"
             if 'usearch_i8_file_MB' in size_439 else ""))
    print(f"[catalog-scale projection @ {args.exemplars_per_part} exemplars/part]")
    for r in proj:
        line = (f"  {r['parts']:>6d} parts -> {r['n_vectors']:>9,d} vecs   "
                f"raw_i8={r['raw_int8_MB']:.1f} MB  raw_f32={r['raw_float32_MB']:.1f} MB")
        if 'usearch_i8_proj_MB' in r:
            line += f"  usearch_i8~={r['usearch_i8_proj_MB']:.1f} MB"
        print(line)

    report = {
        "backbone": desc, "device": device, "seed": SEED,
        "dim": dim, "k": args.k,
        "gallery": {"dir": args.gallery_dir, "n": int(G.shape[0]),
                    "classes": len(gcls), "per_class": args.gallery_per_class},
        "query": {"dir": args.query_dir, "n": int(Q.shape[0]),
                  "per_class": args.query_per_class},
        "accuracy": results,
        "index_size_439": size_439,
        "projection": {"exemplars_per_part": args.exemplars_per_part, "rows": proj},
        "wall_s": round(time.time() - t_all, 1),
    }
    out_json = Path(args.cache_dir) / "ondevice_report.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"\n[report] {out_json}  (wall={report['wall_s']}s)")


if __name__ == "__main__":
    main()
