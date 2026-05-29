# Active-Learning Flywheel — trigger → review → confirm → append → improve

Phase-3 of BrickScan live-scan. The recognition spine is a **frozen embedder +
cosine k-NN over a gallery** (Phase 0/1). The flywheel's defining property:

> **New parts and colors are learned by INSERTING exemplar embeddings into the
> gallery. There is NO retraining of any network.** A confirmed scan improves the
> next scan within milliseconds.

This is deliberately different from the weekly distillation retrain in
`scripts/active_learning_cron.sh` (Track D). That path rebuilds the *student
model* from accumulated corrections on a schedule. The flywheel here is the
**hot path**: it never touches model weights — it only grows the gallery the
k-NN searches. The two compose (retrain occasionally to compress the gallery
back into the backbone), but the flywheel alone already raises accuracy.

```
   ┌─────────┐   margin<τ OR        ┌────────┐  user picks/types   ┌─────────┐
   │  SCAN   │──  correction  ─────▶│ REVIEW │──  the right part ─▶│ CONFIRM │
   └─────────┘   (uncertainty)      └────────┘    + color          └────┬────┘
        ▲                                                                │
        │                                                    embed crop ONCE
        │            next scan of the same brick                        │
        │            retrieves the new exemplar              ┌──────────▼──────────┐
        └──────────  (NO retraining)  ──────────────────────│  APPEND to galleries │
                                                             │  • part (on-device   │
                                                             │    USearch int8)     │
                                                             │  • part (server      │
                                                             │    visual_search +   │
                                                             │    EmbeddingLibrary) │
                                                             │  • color gallery     │
                                                             └─────────────────────┘
```

---

## 1. Trigger — uncertainty sampling

A scan is flagged for human review when **either** holds:

| signal | condition | rationale |
|---|---|---|
| **low margin** | `sim(top-1) − sim(next prediction with a *different* part) < τ` | the model can't separate the top two candidates — the most informative thing to ask about |
| **low confidence** | `sim(top-1) < τ_abs` | a uniformly weak scan, even with nothing close behind it |
| **correction** | user overrides the top pick | ground-truth disagreement is always worth capturing |

- **τ (margin) = 0.05** cosine-similarity units (`FLYWHEEL_MARGIN_TAU`).
  Chosen relative to the spine's confident-match band: `EmbeddingLibrary` treats
  cosine distance ≤ 0.30 (similarity ≥ 0.85) as a confident match and
  `visual_search` uses a 0.55 similarity floor; a 0.05 gap between the top two
  *different* parts means they're effectively tied. Retune from the
  `FeedbackStatsScreen` data via the env var — no code change.
- **τ_abs (confidence floor) = 0.55** (`FLYWHEEL_CONF_FLOOR`), matching
  `visual_search.search(min_similarity=0.55)`.
- The margin is measured to the next prediction whose `part_num` *differs* from
  the top one (mold/colour variants like `3001` vs `3001a` are collapsed first),
  so we don't prompt on a brick that's confidently itself.

Implementation: `backend/app/local_inventory/flywheel_routes.py::should_flag_for_review`
(pure function, unit-tested) and `POST /flywheel/check-uncertainty`. This reuses
and complements the existing absolute-confidence selector in
`feedback_routes.py::get_pending_review` (`GET /feedback/pending-review`) — that
endpoint surfaces the review queue; the margin signal here is the better
in-the-moment trigger because k-NN margin is the spine's native uncertainty.

The existing `ScanFeedback` table + `mobile/src/services/feedbackApi.ts`
(`submitFeedback`, `getPendingReview`) are the review UI/storage; nothing there
changes.

## 2. Review → Confirm

The user either confirms the top pick or corrects it (existing
`/scan-feedback`), and for the flywheel calls **`POST /flywheel/confirm`** with
the crop + confirmed `part_num` + `color_id`. A correction and a confirmation
are the same ingest — both produce a confirmed `(crop, part, color)` triple.

## 3. Append — three galleries, no retraining

`ingest_confirmed()` embeds the crop **once** with the frozen encoder
(`ModelManager.encode_image`, RGB 224 → L2-normalized vector) and appends it:

### a. On-device part gallery — `scripts/gallery_index.py`
The mobile target. A **USearch `ScalarKind.I8` cosine HNSW** index that supports
true **incremental `add`** (no rebuild) and `remove`. `append(embedding,
part_num, color_id)` inserts one int8-quantized vector keyed to a stable id; the
next `search()` sees it immediately. A `.meta.json` sidecar maps id →
`{part_num, color_id, source, ts}` (labels + provenance + audit). `rebuild()`
does a cheap exact re-add from the kept exemplars when you want a fresh graph.

Two int8 invariants are carried verbatim from `scripts/ONDEVICE_NOTES.md` (the
biggest iOS footgun): **single global 1/127 scale** (per-vector scaling breaks
cross-gallery cosine ranking — top-1 collapsed ~85%→~64% in testing) and
**quantize the gallery, keep the query float**.

### b. Server part galleries — `backend/app/ml/flywheel_ingest.py`
Appends to the two indices the scan cascade already reads in
`hybrid_recognition._safe_local_predict`:
- **`visual_search`** (element-level: part + colour). New runtime
  `add_entry()` stacks the row, invalidates the index (lazy single rebuild on
  next query), and persists the pickle. Bootstraps an empty catalogue too.
- **`EmbeddingLibrary`** (part-level k-NN). New `add_exemplar()` **merges** the
  confirmed vector into the part's running prototype (re-normalized) and
  persists, so repeated confirmations *sharpen* the prototype and the index
  never bloats (one vector per part).

### c. Color gallery — `scripts/color_gallery_append.py`
The colour model (`models/color_v1/color_model.npz`) is a frozen extractor +
**baked LDA** + distance-weighted k-NN over projected exemplars. `append`
projects the confirmed crop through the *same* frozen z-score + LDA the artifact
uses at inference and stacks one row onto `gallery_proj` / `gallery_y` — **no
sklearn refit**. A brand-new colour id is registered into
`color_ids/names/hex` (named from `colors.csv`) so it's predictable immediately.
On the backend, confirmed colour crops are journaled to
`backend/data/flywheel/color_exemplars/<color_id>/` for the next gallery rebuild
(the backend has no LDA at query time), mirroring how `feedback_images/` feeds
training.

## 4. Immediate improvement (measured, no retraining)

Reproducible demo over the **cached frozen DINOv2 embeddings** (439-class val,
seed 1234), simulating the flywheel by appending held-out exemplars:

```
seed gallery = 6931 vecs / 439 parts
val top-1 BEFORE flywheel append : 85.09%   (eval n=1502)
append 1625 confirmed exemplars  : 0.16 s total  (0.10 ms / append, incremental)
val top-1 AFTER  flywheel append : 87.08%   (same eval set, NO retraining)
delta                            : +2.00 pp
save→reload round-trip           : OK (size + top-3 parts identical)
```

A single confirmed exemplar for a **never-seen part** (`append --part-num
99999_NEW`) moves the gallery from 439→440 parts on the spot; a never-seen
**color id** (`append --color-id 1000`) moves it 60→61 colors — both with zero
retraining.

---

## Files

| file | role |
|---|---|
| `scripts/gallery_index.py` | on-device append-only USearch int8 gallery: `build_from_embeddings`, **`append`**, `remove`, `search`, `margin`, `save`/`load`, `rebuild`; CLI `demo` / `append` / `stats` |
| `scripts/color_gallery_append.py` | append confirmed colour crop(s) to `color_model.npz` via the baked LDA (no refit); CLI `append` / `append-dir` / `inspect` |
| `backend/app/ml/flywheel_ingest.py` | server orchestrator: embed once → append part galleries → journal colour exemplar; `ingest_confirmed`, `gallery_status` |
| `backend/app/local_inventory/flywheel_routes.py` | `should_flag_for_review` (τ), `POST /flywheel/confirm`, `POST /flywheel/check-uncertainty`, `GET /flywheel/status` |
| `backend/app/services/visual_search.py` | **+`add_entry`** (runtime append + persist + lazy rebuild) |
| `backend/app/ml/embedding_library.py` | **+`add_exemplar`** (running-mean prototype merge + persisted counts) |
| `backend/main.py` | registers `flywheel_router` |

## How to run

```bash
# On the Spark (reproducible; reuses the ondevice_index embedding caches):
cd ~/brickscan/ml && . venv/bin/activate

# 1. On-device gallery: seed + simulate flywheel + show the top-1 jump
python scripts/gallery_index.py demo \
    --gallery-cache output/ondevice_cache/emb_DINOv2_vit_base_patch14_train_pc16.npz \
    --query-cache   output/ondevice_cache/emb_DINOv2_vit_base_patch14_val_pc8.npz \
    --out output/ondevice_cache/flywheel_gallery.usearch

# 2. Append one confirmed exemplar (what the backend calls per confirmation)
python scripts/gallery_index.py append \
    --index output/ondevice_cache/flywheel_gallery.usearch \
    --embedding /tmp/confirmed_emb.npy --part-num 3001 --color-id 5

# 3. Color gallery: append a confirmed crop (no LDA refit), then re-eval
python scripts/color_gallery_append.py append \
    --artifact models/color_v1/color_model.npz \
    --image    training_data/color_v1/val/5/<crop>.jpg --color-id 5
python scripts/color_model.py eval --artifact models/color_v1/color_model.npz \
    --val-dir training_data/color_v1/val
```

Backend (after `pip install -r requirements.txt`): the router auto-registers in
`main.py`. A confirmed scan:

```bash
curl -X POST localhost:8000/api/local-inventory/flywheel/confirm \
  -H 'Content-Type: application/json' \
  -d '{"scan_id":"scan_123","part_num":"3001","color_id":"5","image_base64":"<jpeg b64>"}'
# -> {"ingested":true,"gallery_updated":true,"embedded":true,...}
curl localhost:8000/api/local-inventory/flywheel/status
```

## Notes & limits

- Everything degrades gracefully: if the encoder ONNX isn't deployed yet, the
  part-gallery append is skipped (reported in the response) and the colour
  exemplar is still journaled — the scan cascade is never broken by ingest.
- The frozen interface (RGB 224 → L2-normalized vector → cosine k-NN) is shared
  with `scripts/ondevice_index.py` and the student export. When the FastViT
  student replaces the DINOv2 teacher, the galleries are rebuilt once from the
  student's embeddings and the flywheel continues unchanged.
- Gallery growth is bounded: `EmbeddingLibrary` keeps one merged prototype per
  part; `visual_search` and the on-device index grow per-exemplar but at int8
  (1 byte/dim) — at 4 exemplars/part the projections in `ONDEVICE_NOTES.md` stay
  within mobile budget. Use `gallery_index.remove()` to drop bad exemplars and
  `rebuild()` to defragment.
