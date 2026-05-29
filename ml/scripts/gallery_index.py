#!/usr/bin/env python3
"""Append-only on-device retrieval gallery — the active-learning FLYWHEEL store.

This is the ingest target for Phase-3's flywheel: a CONFIRMED (crop, part_num,
color_id) is embedded once and its embedding is APPENDED to this gallery, so the
very next scan can retrieve it. There is NO retraining — the recognition spine is
a frozen embedder + cosine k-NN, and new parts/colors are learned purely by
inserting exemplars.

Relationship to scripts/ondevice_index.py
------------------------------------------
`ondevice_index.py` PROVES the mechanics (float-vs-int8 top-1 parity, index size,
USearch as the mobile target). This module makes that store *mutable* and gives
it the lifecycle the flywheel needs:

    build_from_embeddings(feats, keys)   one-shot seed from a precomputed gallery
    append(embedding, part_num, color)   <-- the flywheel hot path (incremental)
    remove(entry_id)                     undo a bad exemplar (USearch supports it)
    search(query_float, k)               cosine k-NN, returns (part_num,color,sim)
    save() / load()                      .usearch file + .meta.json sidecar
    rebuild()                            cheap exact re-add from kept exemplars

Frozen interface contract (identical to ondevice_index.py / the student export):
    RGB 224  ->  L2-normalized float embedding (768-d DINOv2 teacher today,
                 128-d FastViT student soon)  ->  cosine k-NN
The gallery stores ONE int8-quantized vector per exemplar, keyed by a stable
integer `entry_id`. A JSON sidecar maps entry_id -> {part_num, color_id, source,
ts} so retrieval can return labels and so the gallery is human-auditable.

Two int8 invariants are carried over verbatim from ONDEVICE_NOTES.md (they are
the single biggest iOS footgun):
  1. SINGLE GLOBAL int8 scale (1/127). Embeddings are L2-normalized so every
     component is already in [-1, 1]; a per-vector scale would break cross-gallery
     cosine ranking and tanked top-1 ~85% -> ~64% in testing. USearch ScalarKind.I8
     does the right (global) thing; we pass float vectors and let it quantize.
  2. Quantize the GALLERY; keep the QUERY in float. The query embedding is fresh
     from the model — never round-tripped through int8. USearch handles this when
     you search an I8 index with a float query.

Incremental add vs rebuild
--------------------------
USearch's HNSW supports true incremental `add` (no full rebuild) and `remove`,
so `append()` is O(log N) and the next `search()` sees the new exemplar
immediately. `rebuild()` is offered for the rare case where you want a fresh
exact graph (e.g. after many removes) — it re-adds the kept exemplars from the
sidecar in one pass; for the 439-class stand-in that is ~0.1s, and even at 10k
parts x 4 exemplars it is a few seconds, i.e. "cheap".

Example (on the Spark, reproducible — reuses the ondevice_index embedding cache):

    cd ~/brickscan/ml && . venv/bin/activate
    # 1. Seed a gallery from the cached DINOv2 train embeddings, then simulate
    #    the flywheel: append held-out val exemplars and show recall jump.
    python scripts/gallery_index.py demo \
        --gallery-cache output/ondevice_cache/emb_DINOv2_vit_base_patch14_train_pc16.npz \
        --query-cache   output/ondevice_cache/emb_DINOv2_vit_base_patch14_val_pc8.npz \
        --out output/ondevice_cache/flywheel_gallery.usearch

    # 2. Append a single confirmed exemplar from an embedding .npy and persist:
    python scripts/gallery_index.py append \
        --index output/ondevice_cache/flywheel_gallery.usearch \
        --embedding /tmp/confirmed_emb.npy --part-num 3001 --color-id 5
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Carried over from ondevice_index.py — these ARE the on-device quantization
# contract. Do not change without re-running the int8 parity proof.
INT8_SCALE = np.float32(1.0 / 127.0)
SEED = 1234
META_SUFFIX = ".meta.json"


# ──────────────────────────────────────────────────────────────────────────────
# Sidecar entry: the label + provenance for one gallery vector.
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class GalleryEntry:
    entry_id: int            # stable USearch key
    part_num: str
    color_id: Optional[str]  # Rebrickable color id as str, or None (part-only)
    source: str              # "seed" | "flywheel" | <free text>
    ts: float                # unix seconds when added


# ──────────────────────────────────────────────────────────────────────────────
# The mutable gallery.
# ──────────────────────────────────────────────────────────────────────────────
class GalleryIndex:
    """Append-only int8 cosine gallery over L2-normalized embeddings.

    Backed by a USearch I8 HNSW index (mobile-deployable: ships a Swift Package
    + ObjC headers; the .usearch file built here loads as-is on iOS). The Python
    side is the build/curate tool; on-device the same file is queried directly.
    """

    def __init__(self, dim: int, *, expansion_add: int = 128,
                 expansion_search: int = 64, connectivity: int = 16):
        from usearch.index import Index, ScalarKind, MetricKind
        self.dim = int(dim)
        self._index = Index(
            ndim=self.dim,
            metric=MetricKind.Cos,
            dtype=ScalarKind.I8,          # 1 byte/dim on disk and on device
            connectivity=connectivity,
            expansion_add=expansion_add,
            expansion_search=expansion_search,
        )
        # entry_id -> GalleryEntry. Keeps labels + provenance out of USearch
        # (which only stores keys + vectors) and makes the gallery auditable.
        self._meta: Dict[int, GalleryEntry] = {}
        self._next_id: int = 0

    # ── size / introspection ────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._index)

    @property
    def num_parts(self) -> int:
        return len({e.part_num for e in self._meta.values()})

    @property
    def num_colors(self) -> int:
        return len({e.color_id for e in self._meta.values() if e.color_id is not None})

    def stats(self) -> Dict:
        return {
            "vectors": len(self),
            "dim": self.dim,
            "parts": self.num_parts,
            "colors": self.num_colors,
            "sources": _count(e.source for e in self._meta.values()),
        }

    # ── seeding (one-shot, from a precomputed gallery) ───────────────────────
    def build_from_embeddings(
        self, feats: np.ndarray, part_nums: List[str],
        color_ids: Optional[List[Optional[str]]] = None, source: str = "seed",
    ) -> int:
        """Bulk-add a precomputed gallery. feats: [N, dim] float (any norm).

        Vectors are L2-normalized defensively before insert (USearch quantizes
        to int8 with a global scale internally). Returns the number added.
        """
        feats = _l2norm(np.asarray(feats, dtype=np.float32))
        if feats.shape[1] != self.dim:
            raise ValueError(f"dim mismatch: index={self.dim} feats={feats.shape[1]}")
        n = feats.shape[0]
        if color_ids is None:
            color_ids = [None] * n
        keys = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
        self._index.add(keys, feats)
        now = time.time()
        for k, pn, cid in zip(keys.tolist(), part_nums, color_ids):
            self._meta[k] = GalleryEntry(k, str(pn), _norm_color(cid), source, now)
        self._next_id += n
        return n

    # ── the flywheel hot path: append ONE confirmed exemplar ─────────────────
    def append(
        self, embedding: np.ndarray, part_num: str,
        color_id: Optional[str] = None, source: str = "flywheel",
    ) -> int:
        """Insert one confirmed exemplar. Incremental — no rebuild.

        `embedding` is the FROZEN embedder's output for the confirmed crop
        (L2-normalized here defensively). After this call the next `search()`
        retrieves it. Returns the new entry_id.
        """
        emb = _l2norm(np.asarray(embedding, dtype=np.float32).reshape(1, -1))
        if emb.shape[1] != self.dim:
            raise ValueError(f"dim mismatch: index={self.dim} emb={emb.shape[1]}")
        entry_id = self._next_id
        self._next_id += 1
        self._index.add(np.array([entry_id], dtype=np.int64), emb)   # float in, i8 stored
        self._meta[entry_id] = GalleryEntry(
            entry_id, str(part_num), _norm_color(color_id), source, time.time())
        return entry_id

    # ── undo a bad exemplar ──────────────────────────────────────────────────
    def remove(self, entry_id: int) -> bool:
        """Remove one exemplar by id (USearch supports lazy delete). Idempotent."""
        if entry_id not in self._meta:
            return False
        try:
            self._index.remove(np.array([entry_id], dtype=np.int64))
        except Exception:
            pass  # already gone from the graph; drop the label regardless
        del self._meta[entry_id]
        return True

    # ── query ────────────────────────────────────────────────────────────────
    def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[str, Optional[str], float]]:
        """Cosine k-NN. `query` is a FLOAT embedding (not int8 — see module docs).

        Returns up to k (part_num, color_id, cosine_similarity) tuples, best
        first. cosine = 1 - USearch's cosine distance.
        """
        if len(self) == 0:
            return []
        q = _l2norm(np.asarray(query, dtype=np.float32).reshape(1, -1))
        kk = min(k, len(self))
        m = self._index.search(q, kk)
        keys = np.atleast_1d(np.asarray(m.keys, dtype=np.int64)).ravel()
        dists = np.atleast_1d(np.asarray(m.distances, dtype=np.float32)).ravel()
        out: List[Tuple[str, Optional[str], float]] = []
        for key, dist in zip(keys.tolist(), dists.tolist()):
            e = self._meta.get(int(key))
            if e is None:
                continue
            out.append((e.part_num, e.color_id, float(1.0 - dist)))
        return out

    def margin(self, query: np.ndarray) -> Optional[float]:
        """Top-1 vs top-2 cosine gap — the uncertainty signal the flywheel uses
        to decide whether to flag a scan for review (small margin = uncertain).

        Returns None when fewer than two DISTINCT part_nums are retrievable.
        """
        hits = self.search(query, k=8)
        if not hits:
            return None
        top_part = hits[0][0]
        top_sim = hits[0][2]
        for pn, _cid, sim in hits[1:]:
            if pn != top_part:                  # gap to the next *different* part
                return float(top_sim - sim)
        return None

    # ── persistence ────────────────────────────────────────────────────────────
    def save(self, path: str | os.PathLike) -> None:
        """Write the .usearch index + a .meta.json sidecar (labels + provenance).

        The sidecar also pins dim, the int8 scale and next_id so load() restores
        the exact lifecycle state. Ship BOTH files together.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._index.save(str(path))
        meta = {
            "version": 1,
            "dim": self.dim,
            "int8_scale": float(INT8_SCALE),    # documents the global-scale contract
            "next_id": self._next_id,
            "count": len(self),
            "entries": [asdict(e) for e in self._meta.values()],
        }
        Path(str(path) + META_SUFFIX).write_text(json.dumps(meta, separators=(",", ":")))

    @classmethod
    def load(cls, path: str | os.PathLike) -> "GalleryIndex":
        """Restore a gallery saved by save(). Reads dim from the sidecar."""
        path = Path(path)
        sidecar = Path(str(path) + META_SUFFIX)
        if not sidecar.exists():
            raise FileNotFoundError(f"missing sidecar {sidecar} (built by save())")
        meta = json.loads(sidecar.read_text())
        g = cls(dim=int(meta["dim"]))
        g._index.load(str(path))
        g._meta = {
            int(e["entry_id"]): GalleryEntry(
                int(e["entry_id"]), str(e["part_num"]),
                _norm_color(e.get("color_id")), e.get("source", "seed"),
                float(e.get("ts", 0.0)),
            )
            for e in meta.get("entries", [])
        }
        g._next_id = int(meta.get("next_id", (max(g._meta) + 1) if g._meta else 0))
        return g

    # ── cheap exact rebuild (after many removes, or to defragment) ────────────
    def rebuild(self) -> "GalleryIndex":
        """Return a fresh GalleryIndex re-built from the kept exemplars.

        Vectors are read back out of the current index (USearch stores them), so
        no re-embedding is needed. Cheap: one bulk add over the kept entries.
        Entry ids and labels are preserved.
        """
        kept = sorted(self._meta.keys())
        fresh = GalleryIndex(dim=self.dim)
        if kept:
            keys = np.array(kept, dtype=np.int64)
            vecs = np.asarray(self._index.get(keys), dtype=np.float32).reshape(len(kept), self.dim)
            fresh._index.add(keys, _l2norm(vecs))
            fresh._meta = {k: self._meta[k] for k in kept}
        fresh._next_id = self._next_id
        return fresh


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────
def _l2norm(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-8)


def _norm_color(cid) -> Optional[str]:
    if cid is None:
        return None
    s = str(cid).strip()
    return s or None


def _count(it) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in it:
        out[v] = out.get(v, 0) + 1
    return out


def _load_cache(npz_path: str):
    """Load an ondevice_index.py embedding cache: feats[N,D] float, labels str[N]."""
    z = np.load(npz_path, allow_pickle=True)
    return z["feats"].astype(np.float32), z["labels"].astype(str)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def _cmd_demo(args):
    """Seed a gallery, then SIMULATE the flywheel and show retrieval improve.

    Protocol:
      - Build the gallery from the cached TRAIN embeddings (the seed catalogue).
      - Split each val class's cached embeddings in half: HALF act as "future
        scans" we never add (the eval queries), the OTHER half are the
        "confirmed corrections" the flywheel appends.
      - Measure val top-1 BEFORE any append, then APPEND the confirmed half and
        measure again. The jump is the immediate, no-retrain benefit.
    This is reproducible (seeded) and reuses the exact DINOv2 caches the int8
    proof used, so the embeddings match the reported baseline.
    """
    rng = np.random.default_rng(SEED)
    Gfeat, Glab = _load_cache(args.gallery_cache)
    Qfeat, Qlab = _load_cache(args.query_cache)
    dim = Gfeat.shape[1]
    print(f"[demo] seed gallery={Gfeat.shape} val={Qfeat.shape} dim={dim}")

    g = GalleryIndex(dim=dim)
    g.build_from_embeddings(Gfeat, list(Glab), source="seed")
    print(f"[demo] seeded: {g.stats()}")

    # Per-class split of the val embeddings into eval vs confirm halves.
    eval_idx, confirm_idx = [], []
    for c in sorted(set(Qlab.tolist())):
        idx = np.where(Qlab == c)[0]
        rng.shuffle(idx)
        h = len(idx) // 2
        eval_idx.extend(idx[:h].tolist())
        confirm_idx.extend(idx[h:].tolist())
    eval_idx = np.array(eval_idx)
    confirm_idx = np.array(confirm_idx)

    def top1(idxs):
        if len(idxs) == 0:
            return 0.0
        hit = 0
        for i in idxs:
            res = g.search(Qfeat[i], k=1)
            if res and res[0][0] == Qlab[i]:
                hit += 1
        return hit / len(idxs)

    before = top1(eval_idx)
    print(f"[demo] val top-1 BEFORE flywheel append : {100*before:.2f}%  "
          f"(eval n={len(eval_idx)})")

    # FLYWHEEL: append the confirmed half one-by-one (incremental, no rebuild).
    t0 = time.time()
    for i in confirm_idx:
        g.append(Qfeat[i], part_num=str(Qlab[i]), color_id=None, source="flywheel")
    dt = time.time() - t0
    print(f"[demo] appended {len(confirm_idx)} confirmed exemplars in {dt:.2f}s "
          f"({1000*dt/max(len(confirm_idx),1):.2f} ms/append) -> {g.stats()}")

    after = top1(eval_idx)
    print(f"[demo] val top-1 AFTER  flywheel append : {100*after:.2f}%  "
          f"(same eval set, NO retraining)")
    print(f"[demo] delta = {100*(after-before):+.2f} pp")

    # Margin uncertainty signal sanity check on a held-out eval query.
    if len(eval_idx):
        mg = g.margin(Qfeat[eval_idx[0]])
        print(f"[demo] sample top1-top2 margin on an eval query: "
              f"{mg:.4f}" if mg is not None else "[demo] margin: n/a")

    if args.out:
        g.save(args.out)
        sz = os.path.getsize(args.out) / 1e6
        print(f"[demo] saved gallery -> {args.out} ({sz:.2f} MB) + {args.out}{META_SUFFIX}")

    # Round-trip check: load it back and confirm size + a query match.
    if args.out:
        g2 = GalleryIndex.load(args.out)
        assert len(g2) == len(g), "round-trip size mismatch"
        r1 = g.search(Qfeat[eval_idx[0]], k=3) if len(eval_idx) else []
        r2 = g2.search(Qfeat[eval_idx[0]], k=3) if len(eval_idx) else []
        ok = [a[0] for a in r1] == [b[0] for b in r2]
        print(f"[demo] reload round-trip: len ok, top-3 parts match={ok}")


def _cmd_append(args):
    """Append a single confirmed embedding (.npy) to an existing saved gallery."""
    emb = np.load(args.embedding).astype(np.float32)
    g = GalleryIndex.load(args.index)
    eid = g.append(emb, part_num=args.part_num, color_id=args.color_id,
                   source=args.source)
    g.save(args.index)
    print(f"[append] entry_id={eid} part={args.part_num} color={args.color_id} "
          f"-> gallery now {g.stats()}")


def _cmd_stats(args):
    g = GalleryIndex.load(args.index)
    print(json.dumps(g.stats(), indent=2))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="seed + simulate flywheel append, show top-1 jump")
    d.add_argument("--gallery-cache", required=True,
                   help="ondevice_index npz cache for the seed (train) embeddings")
    d.add_argument("--query-cache", required=True,
                   help="ondevice_index npz cache for the val embeddings")
    d.add_argument("--out", default=None, help="optional .usearch output path")
    d.set_defaults(func=_cmd_demo)

    a = sub.add_parser("append", help="append one confirmed embedding to a gallery")
    a.add_argument("--index", required=True, help="existing .usearch path (save() output)")
    a.add_argument("--embedding", required=True, help=".npy of an L2-normalized embedding")
    a.add_argument("--part-num", required=True)
    a.add_argument("--color-id", default=None)
    a.add_argument("--source", default="flywheel")
    a.set_defaults(func=_cmd_append)

    s = sub.add_parser("stats", help="print gallery stats")
    s.add_argument("--index", required=True)
    s.set_defaults(func=_cmd_stats)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
