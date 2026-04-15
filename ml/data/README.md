# BrickScan — ML Data Directory

Everything that feeds the training pipeline lives here. This file is the
source of truth for **what exists on disk**, **where training scripts expect
to find things**, and **how to re-download / regenerate anything that's missing**.

None of the large data subdirectories are committed to git (see `.gitignore`).

---

## Directory layout

```
ml/data/
├── ldraw/                 — LDraw parts library (727 MB)
│                             Used by Blender to build 3D meshes. Install once
│                             via download_ldraw_parts.py, then forget about it.
├── synthetic_dataset      — SYMLINK → ~/Desktop/synthetic_dataset
│                             ~500 parts × ~540 renders (268k images, 5.8 GB)
│                             Produced by ml/blender/blender_render.py runs.
│                             Class-folder layout: synthetic_dataset/<part_num>/*.png
│                             This is a symlink because the user's existing
│                             Blender pipeline writes here; we don't want to
│                             duplicate 5.8 GB or risk breaking in-flight renders.
├── nature_2023/           — Nature 2023 real-photo + rendered corpus (pending)
│   ├── detection/
│   │   ├── images/*.jpg
│   │   ├── annotations/*.xml     (PASCAL VOC)
│   │   └── labels/*.txt          (auto-converted YOLO)
│   └── classification_real/<part_num>/*.jpg
├── renders/               — Legacy empty directory; ignore.
├── test_renders/          — Original 4-image smoke test from Day 1.
├── colors_example.csv     — Sample colour palette (for docs).
├── index.csv              — Part metadata index (Rebrickable subset).
└── parts_subset_example.txt — Example subset for focused training runs.
```

---

## What feeds what

| Training script                        | Expects data at                                         | Shape                                 |
|---------------------------------------|---------------------------------------------------------|---------------------------------------|
| `ml/train_contrastive.py`              | `--data-dir` → class-folder layout                      | `<root>/<part_num>/*.{jpg,png}`       |
| `ml/train_yolo.py`                     | `--data` → YAML config pointing at images/labels        | YOLO txt labels, class 0              |
| `ml/train_two_stage.py`                | `BrickClassifierDataset` reads class-folder layout      | `<root>/<part_num>/*.{jpg,png}`       |
| `ml/retrain_from_feedback.py`          | `--feedback-csv` + `--base-data`                        | CSV from `/feedback/export.csv`       |

**Our Desktop-symlinked `synthetic_dataset/` already matches the format**
`train_contrastive.py` and `train_two_stage.py` expect — zero conversion
needed. The Nature 2023 classification set arrives in the same format after
extraction.

---

## Download commands

### 1. LDraw parts library (required for any rendering)

```bash
cd ml/data
python3 download_ldraw_parts.py
# Creates ldraw/ (~727 MB)
```

### 2. Nature 2023 corpus (real photos + bounding boxes)

```bash
cd <repo-root>
./backend/venv/bin/python3 ml/data/download_nature_2023.py \
    --dest ml/data/nature_2023 \
    --priority real
```

- `--priority real` (default) pulls the two **real-photo** datasets first:
  the tagged bounding-box set (~3 GB, for YOLO training) and the small
  real-photo classification set (~6 GB). **This is the highest-value
  addition** because we already have ~268k synthetic renders on disk.
- `--priority renders` pulls the 40 GB synthetic 447-class set — only add
  this if you want to augment the Blender corpus.
- `--priority all` pulls everything (~50 GB total).
- The script is **resumable** — interrupt it any time, re-run, it picks up
  where it stopped.
- Downloads from `mostwiedzy.pl` (Gdańsk University of Technology). Some
  corporate / sandboxed networks block this host — if `Could not resolve
  host: mostwiedzy.pl` appears, the script will print manual-download steps.

### 3. HuggingFace pvrancx/legobricks (alternative synthetic set)

Handled by `ml/download_training_data.py` which runs on Spark via
`spark_relaunch.sh`. Not typically run locally on the Mac.

### 4. Rebrickable catalog (part names / categories)

```bash
cd ml/data
python3 download_rebrickable.py
```

Populates `index.csv` and related CSVs used by the backend for
part-name enrichment on scan results.

---

## Regenerate the synthetic dataset

The Desktop-symlinked `synthetic_dataset/` was built by running
`ml/blender/blender_render.py` per part. To add new parts or re-render
with the new augmentation flags introduced in this upgrade, run:

```bash
# Single part, legacy (backward-compatible with existing renders)
/Applications/Blender.app/Contents/MacOS/Blender --background \
    --factory-startup \
    --python ml/blender/blender_render.py -- \
    --part-id 3001 \
    --ldraw-dir ml/data/ldraw/ldraw \
    --output-dir ml/data/synthetic_dataset \
    --color 4 --num-angles 36 --num-lights 5 --num-zooms 3 --resolution 224

# With new upgrades: hemisphere camera + aggressive augmentation
/Applications/Blender.app/Contents/MacOS/Blender --background \
    --factory-startup \
    --python ml/blender/blender_render.py -- \
    --part-id 3001 \
    --ldraw-dir ml/data/ldraw/ldraw \
    --output-dir ml/data/synthetic_dataset_v2 \
    --color 4 --num-angles 36 --num-lights 5 --num-zooms 3 --resolution 224 \
    --elevation-mode hemisphere --aggressive-aug

# Or use the comprehensive domain-randomisation variant (HDRI backgrounds,
# material jitter, per-angle scale jitter):
/Applications/Blender.app/Contents/MacOS/Blender --background \
    --factory-startup \
    --python ml/blender/blender_render_dr.py -- \
    --part-id 3001 \
    --ldraw-dir ml/data/ldraw/ldraw \
    --output-dir ml/data/synthetic_dataset_dr \
    --num-angles 36 --num-lights 5 --num-zooms 3 --resolution 224
```

For a batch run over many parts at once, use `ml/blender/batch_render.py`.

---

## Dry-run validation before shipping to Spark

Whenever you change `retrain_from_feedback.py` or the feedback CSV schema,
run this smoke test first — catches import / column-drift issues on the Mac
instead of 40 min into a remote Spark run:

```bash
bash ml/scripts/smoke_retrain_dry_run.sh
# Expected: "Dry run OK — pipeline ready for Spark." + exit 0
```

Backed by pytest at `backend/tests/test_retrain_dry_run.py`; runs in CI
automatically whenever the backend test suite is invoked, provided
`pandas` and `torch` are installed.

---

## Disk footprint cheat-sheet

| Asset                                          | Size     |
|-----------------------------------------------|----------|
| `ldraw/` (LDraw parts library)                 |  ~727 MB |
| `synthetic_dataset/` (symlink, Desktop)        |  ~5.8 GB |
| `nature_2023/detection/` (bounding boxes)      |  ~3 GB   |
| `nature_2023/classification_real/`             |  ~6 GB   |
| `nature_2023/classification_renders/` (if)     | ~40 GB   |
| HuggingFace legobricks (Spark-only, not here)  | ~20 GB   |

**Minimum viable local footprint**: LDraw + Desktop synthetic + Nature real = ~15 GB.

---

## See also

- `mobile/DEVELOPMENT.md` — the per-device development workflow (USB IP drift, Metro, Xcode)
- `ml/blender/ARCHITECTURE.md` — the render pipeline's internal design
- `.claude/plans/memoized-crunching-thunder.md` — the most recent implementation plan
