#!/usr/bin/env python3
"""Phase 4 INNER-LOOP closed-loop eval harness.

Replays recorded real frames through a faithful Python mirror of the on-device
live-scan pipeline (mobile/src/ml/liveScanEngine.ts) and emits structured
telemetry JSON that is *diffable across builds*. This is the autonomous half of
the two-loop design: it needs no phone — the OUTER loop (the user's physical
iPhone sweep, instrumented by the same telemetry schema) feeds it real device
captures later. Today the per-piece "frames" are the recorded real photos in
`real_photos_v3/val` used as stand-in views of the same piece.

What it reproduces, step-for-step, from the device code:

  per piece -> one synthetic TRACK of N frames:
    per frame:
      load img -> student.onnx embed -> L2 norm                (embeddingRetrieval.ts)
      int8 global-scale gallery k-NN, top-1 = maxSim           (partIndex.ts)
      fold (embedding, maxSim) into the track's fusion pool    (trackFusion.updateTrack)
    after each frame:
      fused vector = confidence-weighted pool, re-L2-norm      (trackFusion.computeFused)
      fused top-k k-NN; push fused top-1 into stability window (trackFusion.fusedTopK)
      commit gate = stability window all-equal AND margin>tau  (trackFusion.isCommitted)
    color: classify the FIRST frame's crop (color_model.py)    (colorClassifier.ts)

The fusion recipe (softmax(maxSim*20) pool) and the commit semantics
(commitStability consecutive identical fused-top-1 + fused #1-#2 margin > tau)
are byte-for-byte the same constants as DEFAULT_FUSION_OPTS in trackFusion.ts.

Telemetry (one JSON file):
  - meta:   model, gallery, options, dataset, timing, host, git-ish digest
  - tracks: per track -> per-frame {top5, maxSim, fused_top5, committed_here, latency_ms}
                          + final {fused_top1, correct, color, commit_frame, latencies}
  - aggregate: single-frame top-1, fused top-1, recall@3, commit-rate, commit-purity,
               mean frames-to-commit, latency percentiles, per-"basic-part" accuracy

Embeddings are cached to <cache>/emb_<student-stem>_<size>.npz keyed by file path
so re-runs (and parameter sweeps) are fast and CPU-light — the render is paused on
the Spark, so we keep the frame set modest and the CPU work cached.

USAGE (on the Spark; onnxruntime there is CPU):
  ssh spark
  cd /home/edrick/brickscan/ml
  ./venv/bin/python scripts/livescan_harness.py \
     --student     output/student_fastvit_sa24_20260529_025042/student.onnx \
     --gallery-dir training_data/real_photos_v3/train \
     --frames-dir  training_data/real_photos_v3/val \
     --color-artifact models/color_v1/color_model.npz \
     --out         output/livescan_telemetry.json \
     --frames-per-track 4 --max-pieces 60

Standalone deps: onnxruntime (CPU ok), numpy, Pillow. color_model.py is imported
from the same scripts/ dir for color (optional: skipped if --color-artifact absent).
"""
import argparse
import base64
import json
import os
import platform
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import onnxruntime as ort

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# "Basic" bricks we always want a per-class read on (same set as student_index_eval.py).
BASIC = ["3001", "3002", "3003", "3004", "3005"]
SEED = 0

# ---------------------------------------------------------------------------
# Fusion + commit constants — MUST mirror mobile/src/utils/trackFusion.ts
# DEFAULT_FUSION_OPTS. If you change one here, change it there (and re-baseline).
# ---------------------------------------------------------------------------
SOFTMAX_SCALE = 20.0     # softmaxScale: weight = softmax(maxSim * 20)
TOPK = 5                 # topK: fused k-NN depth (need #2 for the margin)
COMMIT_STABILITY = 4     # commitStability: consecutive identical fused-top-1
COMMIT_MARGIN = 0.05     # commitMargin: fused #1 - #2 must exceed this
MAX_FRAMES = 32          # maxFrames: ring-buffer cap on retained per-frame views

# Int8 gallery quantization — mirrors partIndex.ts / ondevice_index.py (global 1/127).
INT8_SCALE = 1.0 / 127.0


# ===========================================================================
# Image / dataset helpers
# ===========================================================================
def list_images(d: str, cap: Optional[int]) -> List[str]:
    # os.path.exists follows symlinks, so this drops broken symlinks (the real
    # dataset has dangling /tmp/phase3_crops links) as well as missing files,
    # keeping the path list aligned with what we can actually embed.
    fs = [os.path.join(d, f) for f in os.listdir(d)
          if os.path.splitext(f)[1].lower() in IMG_EXT
          and os.path.exists(os.path.join(d, f))]
    fs.sort()
    if cap and len(fs) > cap:
        idx = np.linspace(0, len(fs) - 1, cap).astype(int)
        fs = [fs[i] for i in idx]
    return fs


def gather(root: str, cap: Optional[int], classes=None) -> Tuple[List[Tuple[str, str]], List[str]]:
    cls = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    if classes is not None:
        cls = [c for c in cls if c in classes]
    items = []
    for c in cls:
        for p in list_images(os.path.join(root, c), cap):
            items.append((p, c))
    return items, cls


def load_img_chw(path: str, size: int = 224) -> np.ndarray:
    """RGB -> CHW float32 in [0,1]. The student bakes ImageNet norm internally
    (ONDEVICE_NOTES #4), so we DON'T normalize here — same as load_img in
    student_index_eval.py and the EMBED preprocessing in embeddingRetrieval.ts."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    a = np.asarray(img, np.float32) / 255.0
    return np.transpose(a, (2, 0, 1))


# ===========================================================================
# Embedding (student.onnx) with an on-disk cache
# ===========================================================================
class Embedder:
    """Wraps the student ONNX session and a path-keyed embedding cache.

    The cache stores L2-normalized float32 embeddings so a re-run (or a sweep
    over fusion params) costs zero model inference. Keyed by absolute file path;
    invalidated implicitly by the cache filename (student stem + input size)."""

    def __init__(self, student_path: str, size: int, cache_dir: Optional[str], batch: int):
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(student_path, providers=prov)
        self.iname = self.sess.get_inputs()[0].name
        self.oname = self.sess.get_outputs()[0].name
        bshape = self.sess.get_inputs()[0].shape[0]
        # Respect a static batch=1 export; else batch for speed.
        self.batch = 1 if (isinstance(bshape, int) and bshape == 1) else batch
        self.size = size
        self.provider = self.sess.get_providers()[0]
        self.dim: Optional[int] = None

        self._cache: Dict[str, np.ndarray] = {}
        self._cache_path = None
        self._cache_dirty = False
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(student_path))[0]
            self._cache_path = os.path.join(cache_dir, f"emb_{stem}_{size}.npz")
            if os.path.exists(self._cache_path):
                z = np.load(self._cache_path, allow_pickle=True)
                paths = z["paths"]
                vecs = z["vecs"]
                for i, p in enumerate(paths):
                    self._cache[str(p)] = vecs[i]
                self.dim = int(vecs.shape[1]) if len(vecs) else None

    def _run_batch(self, paths: List[str]) -> np.ndarray:
        buf = []
        for p in paths:
            buf.append(load_img_chw(p, self.size))
        x = np.stack(buf).astype(np.float32)
        o = self.sess.run([self.oname], {self.iname: x})[0].astype(np.float32)
        o = o / (np.linalg.norm(o, axis=1, keepdims=True) + 1e-9)
        return o

    def embed_paths(self, paths: List[str]) -> np.ndarray:
        """Return [len(paths), dim] L2-normalized embeddings, using/filling cache."""
        missing = [p for p in paths if p not in self._cache]
        for i in range(0, len(missing), self.batch):
            chunk = missing[i:i + self.batch]
            out = self._run_batch(chunk)
            for j, p in enumerate(chunk):
                self._cache[p] = out[j]
            self._cache_dirty = True
        if self.dim is None and self._cache:
            self.dim = int(next(iter(self._cache.values())).shape[0])
        return np.stack([self._cache[p] for p in paths])

    def flush(self) -> None:
        if self._cache_path and self._cache_dirty:
            paths = list(self._cache.keys())
            vecs = np.stack([self._cache[p] for p in paths]).astype(np.float32)
            np.savez_compressed(self._cache_path, paths=np.array(paths), vecs=vecs)
            self._cache_dirty = False


# ===========================================================================
# Gallery index — int8 global-scale k-NN, the exact on-device path (partIndex.ts)
# ===========================================================================
class GalleryIndex:
    """Dequantized int8 gallery + cosine k-NN. Mirrors partIndex.ts: gallery
    vectors are int8 with a single global scale (1/127); the query stays float
    and is scored against the dequantized gallery (asymmetric path). Ranking is
    exact and score is the cosine estimate."""

    def __init__(self, gallery: np.ndarray, labels: np.ndarray):
        # Quantize then dequantize so retrieval scores match the bundled index
        # exactly (int8 round-trip, not the raw float gallery).
        q = np.clip(np.round(gallery / INT8_SCALE), -127, 127).astype(np.int8)
        self.gallery = (q.astype(np.float32) * INT8_SCALE)
        self.labels = labels
        self.dim = gallery.shape[1]
        self.count = gallery.shape[0]
        self._q8 = q  # retained for index export

    def search(self, vec: np.ndarray, k: int) -> List[Tuple[str, float]]:
        sims = self.gallery @ vec  # (count,)
        k = max(1, min(k, self.count))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(str(self.labels[i]), float(sims[i])) for i in idx]

    def export_index_json(self, path: str) -> int:
        index = {
            "version": 1, "dim": int(self.dim), "count": int(self.count),
            "scale": INT8_SCALE,
            "vectors": base64.b64encode(self._q8.tobytes()).decode("ascii"),
            "partNums": self.labels.tolist(),
        }
        with open(path, "w") as f:
            json.dump(index, f)
        return os.path.getsize(path)


# ===========================================================================
# TrackFusion — Python mirror of mobile/src/utils/trackFusion.ts
# ===========================================================================
def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max()
    e = np.exp(logits - m)
    return e / (e.sum() + 1e-12)


class TrackFusion:
    """One live track's accumulated views + fused retrieval + commit state.

    Faithful to trackFusion.ts: retains per-frame (embedding, maxSim) records
    (ring-buffer capped at MAX_FRAMES), recomputes the confidence-weighted fused
    vector on demand, tracks the recent fused-top-1 stability window, and gates
    commit on (window all-equal & non-empty) AND (fused #1-#2 margin > tau)."""

    def __init__(self):
        self.frames: List[Tuple[np.ndarray, float]] = []  # (L2 emb, maxSim)
        self.recent_top1: List[str] = []

    def update(self, embedding: np.ndarray, max_sim: float) -> None:
        self.frames.append((_l2(embedding.astype(np.float32)), float(max_sim)))
        if len(self.frames) > MAX_FRAMES:
            self.frames = self.frames[-MAX_FRAMES:]

    def fused_vector(self) -> Optional[np.ndarray]:
        if not self.frames:
            return None
        if len(self.frames) == 1:
            return self.frames[0][0].copy()
        embs = np.stack([f[0] for f in self.frames])
        sims = np.array([f[1] for f in self.frames], dtype=np.float32)
        w = _softmax(sims * SOFTMAX_SCALE)[:, None]
        return _l2((embs * w).sum(0))

    def fused_topk(self, index: GalleryIndex) -> List[Tuple[str, float]]:
        """k-NN on the fused vector; also pushes fused top-1 into the stability
        window (matching the side effect of trackFusion.fusedTopK)."""
        fused = self.fused_vector()
        if fused is None:
            return []
        matches = index.search(fused, TOPK)
        top1 = matches[0][0] if matches else ""
        self.recent_top1.append(top1)
        if len(self.recent_top1) > COMMIT_STABILITY:
            self.recent_top1 = self.recent_top1[-COMMIT_STABILITY:]
        return matches

    def is_committed(self, index: GalleryIndex) -> bool:
        if len(self.recent_top1) < COMMIT_STABILITY:
            return False
        newest = self.recent_top1[-1]
        if not newest or any(p != newest for p in self.recent_top1):
            return False
        fused = self.fused_vector()
        if fused is None:
            return False
        matches = index.search(fused, TOPK)
        if not matches or matches[0][0] != newest:
            return False
        margin = matches[0][1] - (matches[1][1] if len(matches) > 1 else 0.0)
        return margin > COMMIT_MARGIN


# ===========================================================================
# Color classifier (optional) — reuse the reference Python model verbatim
# ===========================================================================
class ColorRunner:
    """Thin adapter over scripts/color_model.ColorClassifier. The on-device port
    (colorClassifier.ts) is a TS reimplementation of exactly this; using the
    Python source here keeps the harness's color readout reference-faithful.

    color_model.ColorClassifier.predict(path|img, topk) returns a list of
    {id, name, hex} dicts (best first, distance-weighted kNN), or [] if no brick
    body could be segmented from the crop — same empty-result contract the TS
    classifier returns as colorId ''."""

    def __init__(self, artifact_path: str):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import color_model  # noqa: E402  (scripts/color_model.py)
        self._cm = color_model
        self.clf = color_model.ColorClassifier(artifact_path)

    def classify_path(self, path: str, topk: int = 3) -> Optional[List[Dict[str, str]]]:
        preds = self.clf.predict(path, topk=topk)  # [{id, name, hex}, ...]
        return preds if preds else None


# ===========================================================================
# Per-track replay
# ===========================================================================
def replay_track(piece_id: str, frame_paths: List[str], frame_embs: np.ndarray,
                 index: GalleryIndex, color: Optional[ColorRunner]) -> Dict:
    """Run one piece's frames through the live loop, returning a telemetry dict.

    `frame_embs` are the precomputed L2-normalized embeddings aligned to
    `frame_paths`. We time only the per-frame retrieval+fusion+commit math
    (embedding latency is measured separately and reported in meta) so the
    per-frame latency reflects the on-device retrieval cost, not disk/cache IO."""
    fusion = TrackFusion()
    per_frame = []
    commit_frame: Optional[int] = None

    for fi, (path, emb) in enumerate(zip(frame_paths, frame_embs)):
        t0 = time.perf_counter()
        # 1. single-frame top-k; top-1 score is this frame's maxSim.
        top = index.search(emb, TOPK)
        max_sim = top[0][1] if top else 0.0
        fusion.update(emb, max_sim)
        # 2. fused retrieval (also advances the stability window).
        fused = fusion.fused_topk(index)
        committed_now = fusion.is_committed(index)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        if committed_now and commit_frame is None:
            commit_frame = fi + 1  # 1-indexed: "committed after this many frames"

        per_frame.append({
            "frame": fi,
            "file": os.path.basename(path),
            "top5": [{"partNum": p, "score": round(s, 4)} for p, s in top],
            "maxSim": round(max_sim, 4),
            "fused_top5": [{"partNum": p, "score": round(s, 4)} for p, s in fused],
            "committed_here": committed_now,
            "latency_ms": round(dt_ms, 3),
        })

    final_fused = fusion.fused_topk(index)  # final read (no extra commit needed)
    fused_top1 = final_fused[0][0] if final_fused else ""
    single_top1 = per_frame[0]["top5"][0]["partNum"] if per_frame and per_frame[0]["top5"] else ""

    color_pred = None
    if color is not None and frame_paths:
        try:
            cp = color.classify_path(frame_paths[0])  # [{id, name, hex}, ...] or None
            if cp:
                color_pred = {
                    "colorId": cp[0]["id"],
                    "colorName": cp[0].get("name", ""),
                    "topk": [{"colorId": c["id"], "colorName": c.get("name", "")} for c in cp],
                }
        except Exception as e:  # color is best-effort, exactly like on device
            color_pred = {"error": str(e)}

    latencies = [f["latency_ms"] for f in per_frame]
    return {
        "piece_id": piece_id,
        "n_frames": len(frame_paths),
        "frames": per_frame,
        "single_frame_top1": single_top1,
        "single_frame_correct": single_top1 == piece_id,
        "fused_top1": fused_top1,
        "fused_top5": [{"partNum": p, "score": round(s, 4)} for p, s in final_fused],
        "fused_correct": fused_top1 == piece_id,
        "recall3": piece_id in [m[0] for m in final_fused[:3]],
        "committed": commit_frame is not None,
        "commit_frame": commit_frame,
        "commit_correct": (commit_frame is not None and fused_top1 == piece_id),
        "color": color_pred,
        "latency_ms": {
            "mean": round(float(np.mean(latencies)), 3) if latencies else 0.0,
            "max": round(float(np.max(latencies)), 3) if latencies else 0.0,
        },
    }


# ===========================================================================
# Aggregate
# ===========================================================================
def pct(arr: List[float], q: float) -> float:
    return round(float(np.percentile(arr, q)), 3) if arr else 0.0


def aggregate(tracks: List[Dict]) -> Dict:
    n = len(tracks)
    if n == 0:
        return {}
    single = sum(t["single_frame_correct"] for t in tracks)
    fused = sum(t["fused_correct"] for t in tracks)
    r3 = sum(t["recall3"] for t in tracks)
    committed = [t for t in tracks if t["committed"]]
    commit_correct = sum(t["commit_correct"] for t in tracks)
    ftc = [t["commit_frame"] for t in committed if t["commit_frame"] is not None]
    all_lat = [f["latency_ms"] for t in tracks for f in t["frames"]]

    # Per "basic part" fused accuracy (the everyday bricks).
    basic = {}
    for b in BASIC:
        members = [t for t in tracks if t["piece_id"] == b]
        if members:
            basic[b] = round(100 * sum(m["fused_correct"] for m in members) / len(members), 1)

    return {
        "n_tracks": n,
        "single_frame_top1": round(100 * single / n, 2),
        "fused_top1": round(100 * fused / n, 2),
        "recall_at_3": round(100 * r3 / n, 2),
        "commit_rate": round(100 * len(committed) / n, 2),
        # purity = of the tracks that committed, what fraction committed the RIGHT part
        "commit_purity": round(100 * commit_correct / max(1, len(committed)), 2),
        "mean_frames_to_commit": round(float(np.mean(ftc)), 2) if ftc else None,
        "retrieval_latency_ms": {
            "mean": round(float(np.mean(all_lat)), 3) if all_lat else 0.0,
            "p50": pct(all_lat, 50), "p90": pct(all_lat, 90), "p99": pct(all_lat, 99),
            "max": round(float(np.max(all_lat)), 3) if all_lat else 0.0,
        },
        "basic_parts_fused_top1": basic,
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", required=True, help="student .onnx (768-d, baked ImageNet norm)")
    ap.add_argument("--gallery-dir", required=True, help="root of class-subdir gallery images")
    ap.add_argument("--frames-dir", required=True,
                    help="root of class-subdir frames (stand-in live views per piece)")
    ap.add_argument("--out", required=True, help="output telemetry JSON path")
    ap.add_argument("--color-artifact", default=None, help="color_model.npz (optional)")
    ap.add_argument("--export-index", default=None,
                    help="optional: also write the int8 gallery_index.json used here")
    ap.add_argument("--gallery-per-class", type=int, default=30)
    ap.add_argument("--frames-per-track", type=int, default=4,
                    help="N views fused per piece (the live multi-frame depth)")
    ap.add_argument("--max-pieces", type=int, default=60,
                    help="cap pieces (classes) replayed — keep modest on the Spark")
    ap.add_argument("--cache-dir", default=None,
                    help="embedding cache dir (default: <out dir>/.emb_cache)")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    t_start = time.time()
    cache_dir = args.cache_dir or os.path.join(os.path.dirname(os.path.abspath(args.out)) or ".",
                                               ".emb_cache")

    # --- gallery ---
    gitems, gcls = gather(args.gallery_dir, args.gallery_per_class)

    # --- frames: pick the pieces (classes) present in BOTH gallery and frames,
    #     keep only those with at least frames-per-track views, cap to max-pieces. ---
    fitems, _ = gather(args.frames_dir, None, classes=set(gcls))
    by_piece: Dict[str, List[str]] = {}
    for p, c in fitems:
        by_piece.setdefault(c, []).append(p)
    eligible = sorted([c for c, ps in by_piece.items() if len(ps) >= args.frames_per_track])
    # Prefer the BASIC bricks first (always observed), then fill to max-pieces.
    ordered = [c for c in BASIC if c in eligible] + [c for c in eligible if c not in BASIC]
    if args.max_pieces:
        ordered = ordered[:args.max_pieces]

    print(f"[harness] gallery={len(gitems)} imgs / {len(gcls)} classes; "
          f"replaying {len(ordered)} pieces x {args.frames_per_track} frames", flush=True)

    emb = Embedder(args.student, size=224, cache_dir=cache_dir, batch=args.batch)
    print(f"[harness] student={os.path.basename(args.student)} provider={emb.provider} "
          f"batch={emb.batch}", flush=True)

    # --- embed gallery (timed for the meta) ---
    t_g = time.perf_counter()
    G = emb.embed_paths([p for p, _ in gitems])
    g_embed_ms = (time.perf_counter() - t_g) * 1000.0
    glab = np.array([c for _, c in gitems])
    index = GalleryIndex(G, glab)
    if emb.dim is None:
        emb.dim = int(G.shape[1])

    if args.export_index:
        sz = index.export_index_json(args.export_index)
        print(f"[harness] wrote {args.export_index} ({sz // 1024} KB)", flush=True)

    color = None
    if args.color_artifact and os.path.exists(args.color_artifact):
        try:
            color = ColorRunner(args.color_artifact)
            print(f"[harness] color model loaded ({os.path.basename(args.color_artifact)})", flush=True)
        except Exception as e:
            print(f"[harness] color model unavailable, skipping color: {e}", flush=True)

    # --- replay each piece ---
    # Per-piece frames: take the first frames-per-track views (deterministic).
    all_frame_paths = []
    track_slices = []
    for c in ordered:
        fps = sorted(by_piece[c])[:args.frames_per_track]
        track_slices.append((c, len(all_frame_paths), len(fps)))
        all_frame_paths.extend(fps)
    t_q = time.perf_counter()
    Qall = emb.embed_paths(all_frame_paths)
    q_embed_ms = (time.perf_counter() - t_q) * 1000.0
    per_frame_embed_ms = q_embed_ms / max(1, len(all_frame_paths))

    tracks = []
    for (c, off, cnt) in track_slices:
        fps = all_frame_paths[off:off + cnt]
        fe = Qall[off:off + cnt]
        tracks.append(replay_track(c, fps, fe, index, color))

    emb.flush()
    agg = aggregate(tracks)

    telemetry = {
        "schema": "brickscan.livescan.telemetry/v1",
        "source": "inner_loop_replay",
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "host": platform.node(),
            "student": os.path.abspath(args.student),
            "student_basename": os.path.basename(args.student),
            "embedding_dim": emb.dim,
            "provider": emb.provider,
            "gallery_dir": os.path.abspath(args.gallery_dir),
            "gallery_size": int(len(gitems)),
            "gallery_classes": int(len(gcls)),
            "frames_dir": os.path.abspath(args.frames_dir),
            "frames_per_track": args.frames_per_track,
            "pieces_replayed": len(ordered),
            "fusion_opts": {
                "softmaxScale": SOFTMAX_SCALE, "topK": TOPK,
                "commitStability": COMMIT_STABILITY, "commitMargin": COMMIT_MARGIN,
                "maxFrames": MAX_FRAMES, "int8Scale": INT8_SCALE,
            },
            "timing_ms": {
                "gallery_embed_total": round(g_embed_ms, 1),
                "query_embed_total": round(q_embed_ms, 1),
                "per_frame_embed": round(per_frame_embed_ms, 2),
                "wall_total": round((time.time() - t_start) * 1000.0, 1),
            },
        },
        "aggregate": agg,
        "tracks": tracks,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(telemetry, f, indent=2)

    # --- console summary (the diffable headline numbers) ---
    print(f"[harness] wrote {args.out}", flush=True)
    print("─" * 64, flush=True)
    print(f"  pieces            {agg['n_tracks']}", flush=True)
    print(f"  single-frame top1 {agg['single_frame_top1']:.2f}%", flush=True)
    print(f"  FUSED top1        {agg['fused_top1']:.2f}%   (recall@3 {agg['recall_at_3']:.2f}%)", flush=True)
    print(f"  commit-rate       {agg['commit_rate']:.2f}%   purity {agg['commit_purity']:.2f}%", flush=True)
    print(f"  frames-to-commit  {agg['mean_frames_to_commit']}", flush=True)
    lat = agg["retrieval_latency_ms"]
    print(f"  retrieval latency p50={lat['p50']}ms p90={lat['p90']}ms p99={lat['p99']}ms", flush=True)
    print(f"  per-frame embed   {per_frame_embed_ms:.2f}ms (provider {emb.provider})", flush=True)
    if agg.get("basic_parts_fused_top1"):
        print(f"  basic parts       {agg['basic_parts_fused_top1']}", flush=True)
    print("─" * 64, flush=True)


if __name__ == "__main__":
    main()
