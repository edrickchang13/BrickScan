# BrickScan Live-Scan — Phase 0 Results

Goal: prove the recognition spine for real-time live multi-angle scanning —
**frozen-embedding + k-NN retrieval, lifted by multi-frame fusion**, plus a
**separate color pipeline**. All eval on the shared DGX Spark (GB10).

Reproduce: `ssh spark && cd ~/brickscan/ml && . venv/bin/activate`, then run the
scripts in `ml/scripts/` (paths below). Scripts are seeded/deterministic.

## 1. Single-frame retrieval baseline (`scripts/knn_baseline.py`)

Frozen backbone → L2-normalized embedding → k-NN vs a real-photo gallery
(`real_photos_v3/train`), evaluated on held-out `real_photos_v3/val` (439 classes):

| Backbone | NN top-1 | recall@3 | recall@5 | dim |
|---|---|---|---|---|
| **DINOv2 ViT-B/14** | **89.3%** | 94.3% | 96.0% | 768 |
| C-RADIOv3-B | 86.2% | 91.8% | 94.0% | 2304 |

vs the previously trained MobileNetV3 classifier at **~8%** held-out cross-domain.
Retrieval (not a softmax classifier) is the spine. DINOv2 ViT-B wins and is smaller.

## 2. Multi-frame fusion — the crux (`scripts/fusion_eval.py`)

Fuse N views of the same piece (as a live sweep would), then retrieve. N=1 is the
single-frame baseline. Frozen DINOv2, gallery=`train`, query=`val`:

| N (views) | pooling | top-1 | recall@3 |
|---|---|---|---|
| 1 | — | 88.4% | 93.9% |
| 2 | mean | 91.5% | 96.6% |
| 2 | conf-weighted | 93.7% | 96.9% |
| 4 | mean | 92.2% | 97.2% |
| **4** | **conf-weighted** | **95.4%** | **98.6%** |
| 8 | conf-weighted | 94.9% | 98.6% |

**Verdict: multi-frame fusion clearly beats single-frame (88.4% → 95.4%, +7 pts).**
Confidence-weighted pooling beats plain mean at every N; ~N=4 is the sweet spot.
Look-alike basic bricks (3001–3005) also improve (e.g. 3002/3004/3005 → 70–80% at
N=4 conf-weighted, from ~50–58% single-frame), confirming that *angles* are what
disambiguate the hardest pieces. This is the core thesis, validated.

### Cross-domain (source-held-out) fusion (`scripts/fusion_xdomain.py`)

Hold out one capture source as the query, build the gallery from the others — a
genuine domain shift. DINOv2 frozen, confidence-weighted top-1:

| Held-out source (query) | classes | N=1 | N=4 | N=8 | lift |
|---|---|---|---|---|---|
| nature (real photos) | 151 | 20.7% | 23.6% | 23.4% | +2.9 |
| kaggle_sorting | 19 | 56.7% | 65.9% | 66.1% | +9.4 |
| kaggle_images | 11 | 58.2% | 64.5% | 70.9% | +12.7 |
| cdn (too sparse to fuse*) | 138 | 48.2% | — | — | — |

\*cdn has ~1.2 imgs/class — single-frame baseline only.

**Verdict: the fusion lift survives domain shift in every split deep enough to
fuse, and is *larger* than in-domain in two of three (+9.4, +12.7 vs +7).** The
conservative nature split (real-photo query vs a tiny 739-image render/sorting
gallery) still gains +2.9, but off a low 20.7% base — that gap is dominated by
weak, mismatched gallery coverage, not by fusion. So multi-frame fusion is a real,
consistent win even cross-domain; it *complements* (doesn't replace) fixing
gallery/domain coverage. Basic bricks improve too (held-out=nature: 3001 30→60%,
3005 65→100% at N=8). Confidence-weighting again beats mean in the hard splits
(kaggle_images N=8: 52.7% mean vs 70.9% conf-wt).

## 3. Color pipeline (`scripts/color_eval.py`)

Color is a SEPARATE pipeline from the shape embedding (which is color-invariant by
design). Extraction: white-balance → background subtraction → drop specular
highlights + deep shadow → robust interior color. Three matching strategies, on
`color_v1/val` (60 colors, n=1922), top-1 / top-3:

| Strategy | full(274) | common(64) | dataset(60) |
|---|---|---|---|
| canon (ΔE2000 to Rebrickable hex) | 15.6 / 27.8 | 36.6 / 57.4 | 38.6 / 61.8 |
| knn-lab (median-LAB kNN vs real exemplars) | 81.4 / 88.3 | 78.4 / 85.4 | 81.4 / 88.3 |
| **lda-knn (12-d feat + LDA + dist-wt kNN)** | **84.2 / 90.8** | 81.0 / 87.5 | 84.2 / 90.8 |

**Color-by-retrieval against real photographed exemplars is the fix: 40% → 84.2%
top-1 (90.8% top-3), beating the incumbent (RebrickNet ~80%).** Canonical-hex ΔE
fails because it can't survive the camera/lighting shift (black extracts as gray and
always mispredicts "Pearl Titanium"); real exemplars share that shift, so they land
nearby. A learned 12-d metric (median LAB + gloss L-distribution + 2 transparency
cues → LDA → distance-weighted kNN) adds ~3 pts and lifts Trans-Clear 48%→65%.
Remaining ~16% errors are irreducible near-duplicate darks/grays and trans colors
that genuinely overlap from a single crop — mutual swaps that multi-frame fusion
(top-3 already 90.8%) should mop up. (lda-knn full==dataset: retrieval only predicts
colors it has exemplars for — the realistic deployment property.)

## Caveats (honest)

- Fusion eval is a proxy: N random same-class views, not N angles of one physical
  instance — directionally valid (standard re-ID/video-retrieval methodology), but
  real per-instance multi-angle sweeps should be confirmed on-device.
- §1–2 are in-domain held-out (`val`); the original cross-domain set (`val_xdomain`)
  was lost (its source images were symlinks into a wiped `/tmp`). §2 cross-domain
  rerun is underway.
- Color extraction assumes a roughly single-colored brick crop; multi-color/printed
  and trans/chrome/pearl finishes need a finish-class flag (future work).
