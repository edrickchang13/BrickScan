# BrickScan vs Brickit / Brickify — head-to-head benchmark protocol

This document is the spec for an apples-to-apples accuracy + latency
comparison of BrickScan against the two leading commercial competitors.
The methodology is designed so we can rerun it after every meaningful
model change and have defensible numbers for marketing copy, App Store
screenshots, and investor decks.

## Why this benchmark exists

We can ship Phase 6 / 7 / 8 features all day, but until we measure against
the apps users actually have installed, we don't know whether we're
ahead or behind. Marketing claims like "99% top-3 cross-domain" are
correct on a held-out Hex:Lego split — they say nothing about how we
fare against Brickit on a real desk under household lighting.

## Test pile — fixed reference

Lay out **30 LEGO bricks** that meet these criteria:

- 20 common parts (2x4 brick, 1x2 plate, 2x2 plate, 1x1 round plate, etc.)
- 5 mid-frequency parts (slopes, hinges, technic pins)
- 5 long-tail / unusual parts (rare connectors, printed tiles, minifig
  accessories) — these are where models distinguish themselves
- Mix of colours, including at least 1 trans / translucent piece
- Mix of conditions: 22 pristine, 8 worn (small scratches / faded studs)

Photograph the pile **5 times** in five different conditions:

| Condition | Description |
|---|---|
| L1: studio | Bright overhead, white desk, no shadows |
| L2: window-day | Indirect daylight from a window, neutral wood desk |
| L3: dim-evening | Single yellowish desk lamp, mostly side-lit |
| L4: harsh-direct | Hard direct sunlight or LED panel, sharp shadows |
| L5: low-contrast | Cloudy daylight, white pile on white paper |

Total ground-truth set: **30 bricks × 5 conditions = 150 samples**.

Capture all 5 photos with the same iPhone, hand-held at roughly the same
distance (~30cm above the pile). Save each as a 1024×768 JPEG at quality
85 in `docs/benchmark_assets/L{1..5}.jpg`.

Manually fill out the ground truth in
`docs/benchmark_assets/ground_truth.csv`:

```csv
photo_id,brick_index,part_num,part_name,color_id,color_name,is_worn
L1,1,3001,Brick 2 x 4,4,Red,false
L1,2,3023,Plate 1 x 2,1,Blue,false
...
```

`brick_index` is a stable per-photo integer matching the position you
labelled in the photo (number them on a sticky note or use an annotation
tool — anything stable across runs).

## App-by-app procedure

For each app (BrickScan, Brickit, Brickify):

1. Cold-start the app
2. Open the multi-piece / pile scan mode
3. For each photo L1..L5, take/upload that photo and capture the
   identification result
4. Record the **wall-clock time** from "tap scan" to "result visible"
5. For each detected brick, record `(predicted_part_num, predicted_color_name)`
6. If the app misses a brick (no detection at all), mark it as `MISS`
7. If the app produces a detection that doesn't match any ground-truth
   brick, mark it as `EXTRA` (a false positive)

Save results in `docs/benchmark_assets/results_{app}.csv`:

```csv
photo_id,brick_index,predicted_part_num,predicted_color_name,latency_ms
L1,1,3001,Red,3200
L1,2,3023,Blue,3200
L1,3,MISS,,3200
...
```

## Metrics produced by the harness

The Python harness in `scripts/benchmark_score.py` (see below) computes
the same numbers for every app:

- **Top-1 part accuracy** — % of brick-index rows where
  `predicted_part_num == ground_truth.part_num`
- **Top-1 colour accuracy** — % of rows where the colour name matches
  exactly (after the canonical-name normalisation in
  `app/services/part_num_normalizer.py`)
- **Top-1 part+colour joint accuracy** — both must match
- **Recall** — % of ground-truth bricks the app emitted any prediction for
- **Precision** — % of emitted predictions that were correct
- **Mean / median / p95 latency** — wall-clock seconds per pile photo
- **Failure rate** — % of pile photos where the app crashed / errored / no result
- **Per-condition breakdown** — same metrics broken out by L1..L5
- **Per-class confusion matrix** — top-10 confused (gt → pred) pairs

Output is two files:
- `docs/benchmark_assets/results_summary.csv` — flat table of per-app metrics
- `docs/benchmark_assets/results_summary.md` — pretty markdown table for
  pasting into the marketing site

## Scoring harness — `scripts/benchmark_score.py`

A pure-Python script — no GPU, no backend dependency. Runs on the Mac
in <2 seconds. See `scripts/benchmark_score.py` in this repo.

## What "winning" looks like

Realistic targets for BrickScan to claim parity with Brickit/Brickify
in marketing:

| Metric | BrickScan today (Phase 2 + Phase 6) | Target |
|---|---:|---:|
| Top-1 part accuracy | ? | ≥ Brickit-1pt |
| Top-1 colour accuracy | ? | ≥ Brickit + 5pts (we trained explicitly for colour, they treat it as a side feature) |
| Joint part+colour | ? | ≥ Brickit-2pts |
| Recall | ? | ≥ Brickit (we use YOLO; their detector is unknown) |
| Mean latency | ? | ≤ Brickit + 30% (they're snapshot, we're continuous fusion) |

Anything that beats Brickify by 3+ points on any metric should land in
the App Store screenshots. Anything we're behind on by >5 points becomes
the next sprint target.

## When to rerun

- After every model deploy (`deploy_color_classifier`, `deploy_yolo`, etc.)
- Before any App Store release
- Quarterly even when nothing changes — competitors are improving too

## Calendar of past runs

(append a row each time the harness produces a `results_summary.csv`)

| Date | BrickScan version | Brickit version | Brickify version | Top-1 winner |
|---|---|---|---|---|
| _2026-05-08_ | _initial_ | _TBD_ | _TBD_ | _TBD_ |
