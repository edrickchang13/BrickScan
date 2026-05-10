# Public LEGO datasets for evaluating BrickScan

A curated list of labeled LEGO datasets you can point `eval_app_against_dataset.py`
at to measure BrickScan's accuracy. Every dataset listed has been confirmed to
have ground-truth labels we can score against.

## Quick start — the fastest test

If you've already pulled the Spark `rebrickable_cdn` to your Mac (or you have
the Hex:Lego dataset locally), this works out of the box:

```bash
cd /Users/edrickchang/Documents/Claude/Projects/Lego/brickscan/backend

# Make sure backend is up
docker compose up -d
curl localhost:8000/health   # expect: {"status":"ok"}

# Run the eval against Hex:Lego (Roboflow YOLO format, ~150 cropped bricks)
python3 tools/eval_app_against_dataset.py \
    --layout yolo \
    --root  /path/to/hex-lego_v3 \
    --max-per-class 6 \
    --concurrency 4
```

You'll get a printed report:
```
============================================================
  total          : 90
  top-1 accuracy : 86.67%
  top-3 accuracy : 95.56%
  top-5 accuracy : 98.89%
  color top-1    : 91.11%  (n=90)
  wall time      : 24.6s   (3.7 img/s)
============================================================

  top miss confusion (gt → predicted) — most common 10 classes:
    3001       → 3001b×1, 3003×1
    3022       → 3024×1
    ...
```

Plus a full `eval_app_report.json` with every per-image prediction.

## Recommended datasets (download links + scoring layout)

### 1. Hex:Lego v3 — Roboflow 🏆 (best signal-to-effort ratio)
- **Source:** [universe.roboflow.com/hexhewwie/hex-lego/dataset/3](https://universe.roboflow.com/hexhewwie/hex-lego/dataset/3)
- **Size:** 8,320 images, 28 classes (color×shape combinations)
- **Labels:** YOLO bounding boxes
- **License:** CC BY 4.0
- **Download:** Free with a Roboflow account (or the API key on Spark)
- **Layout flag:** `--layout yolo`
- **Why use it:** Mixed lighting/backgrounds, real photos, includes color
  ground truth. Closest to "what users will actually scan."

### 2. Brickognize test set
- **Source:** [tramacsoft.com/brickognize](https://www.tramacsoft.com/brickognize)
- **Size:** 76 bricks, ~1500 controlled + uncontrolled photos
- **Labels:** part_num
- **License:** Permissive research
- **Layout flag:** `--layout part_num`
- **Why use it:** Standard benchmark used in the Brickognize paper. Lets you
  cite published numbers as comparison points.

### 3. pacogarciam3 Sorting (Kaggle)
- **Source:** [kaggle.com/datasets/pacogarciam3/lego-brick-sorting-image-recognition](https://www.kaggle.com/datasets/pacogarciam3/lego-brick-sorting-image-recognition)
- **Size:** 18,325 photos, 20 classes (semantic names like "Brick_2x4")
- **Labels:** Semantic class folder names
- **License:** Permissive
- **Layout flag:** `--layout semantic --semantic-map tools/pacog_to_partnum.json`
- **Why use it:** Real photos on white paper background. Great for "how does
  the model do on a clean studio shot?"

### 4. Nature 2023 / Boiński et al.
- **Source:** [nature.com/articles/s41597-023-02682-2](https://www.nature.com/articles/s41597-023-02682-2)
  (data on MOST Wiedzy under the Polish open-data portal)
- **Size:** 77,535 real photos, 432 part_num classes
- **Labels:** part_num (folder names = part numbers)
- **License:** CC BY 4.0
- **Layout flag:** `--layout part_num`
- **Why use it:** Largest publicly available real-photo dataset. Long-tail
  coverage of less common parts.

### 5. B200C — Kaggle
- **Source:** [kaggle.com/datasets/ronanpickell/b200c-lego-classification-dataset](https://www.kaggle.com/datasets/ronanpickell/b200c-lego-classification-dataset)
- **Size:** 800,000 SYNTHETIC renders, 200 part_num classes
- **License:** Permissive
- **Layout flag:** `--layout part_num`
- **Caveat:** All synthetic — useful as a stress test on coverage breadth,
  NOT representative of phone photos. Expect lower accuracy than real-photo
  sets because BrickScan was trained primarily on real images.

### 6. lego-s6zjh — Roboflow
- **Source:** [universe.roboflow.com](https://universe.roboflow.com/) (search "lego")
- **Size:** 3,302 images
- **Labels:** YOLO format
- **Layout flag:** `--layout yolo`
- **Caveat:** Class labels are Bosch-style PLC identifiers (`X1-Y1-Z2`),
  not Rebrickable part numbers. The eval script's `hex_label_to_part`
  mapper won't recognise them — would need a custom mapper.

## Quick sanity-check kit (10-image smoke test)

If you don't want to pull a full dataset, the following 10 part numbers are
a representative cross-section of the model's vocabulary. Hand-photograph
or screenshot each one, save them under `<root>/<part_num>/sample.jpg`,
and run with `--layout part_num --max-per-class 1`:

| part_num | name | typical confidence |
|---|---|---|
| 3001 | Brick 2 × 4 | very high — most common brick |
| 3003 | Brick 2 × 2 | high |
| 3022 | Plate 2 × 2 | high |
| 3024 | Plate 1 × 1 | medium — small part |
| 3623 | Plate 1 × 3 | medium |
| 6141 | Plate Round 1 × 1 | medium — easy to confuse with 3024 |
| 3039 | Slope 45 2 × 2 | medium |
| 3795 | Plate 2 × 6 | medium |
| 3460 | Plate 1 × 8 | medium |
| 87580 | Plate w/ Tube | low — uncommon |

## Reading the report

Generated `eval_app_report.json` shape:

```json
{
  "total": 120,
  "top1": 0.875,
  "top3": 0.967,
  "top5": 0.992,
  "color_top1": 0.892,
  "n_with_color_gt": 120,
  "wall_s": 31.4,
  "rate_img_per_s": 3.8,
  "top_misses": {
    "3001": [["3001b", 2], ["3010", 1]],
    "3022": [["3024", 1]]
  },
  "detail": [
    {
      "image": "/tmp/eval_app_crops/2x4_red_001.jpg",
      "gt_part": "3001",
      "gt_color": "red",
      "top1_part": "3001",
      "top1_color": "Red",
      "top1_conf": 0.94,
      "top1_source": "brickognize",
      "top5_parts": ["3001", "3001b", "3010", "3003", "3002"]
    }
  ]
}
```

`top_misses` is the most actionable section — it tells you exactly which
classes the model confuses with which others. If you see `3001 → 3001b` in
the misses, that's a mold-variant collapse issue (set `SCAN_COLLAPSE_VARIANTS=true`
in `docker-compose.yml`).
