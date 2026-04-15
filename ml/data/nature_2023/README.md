# Nature 2023 LEGO Bricks Dataset — Local Mirror

**Paper:** Boiński et al., *"Photos and rendered images of LEGO bricks"*,
Nature Scientific Data, Nov 2023 — [nature.com/articles/s41597-023-02682-2](https://www.nature.com/articles/s41597-023-02682-2)

**Source:** MostWiedzy (Gdańsk University of Technology institutional repo)

This folder is populated by `../download_nature_2023.py`. Until you run
that (on a network that can reach `mostwiedzy.pl`), the subdirectories
below are empty placeholders.

## Layout once populated

```
nature_2023/
├── detection/
│   ├── images/*.jpg           — real photos + renders (~2.9k + ~2.9k)
│   ├── annotations/*.xml      — PASCAL VOC bounding boxes (source format)
│   └── labels/*.txt           — YOLO txt (auto-converted, class 0 = lego_piece)
├── classification_real/
│   └── <part_num>/*.jpg       — real photos per class (~52k across 447 classes)
├── classification_renders/    — (optional, ~40 GB) only with --priority renders|all
└── conveyor/                  — (optional) conveyor-belt video clips
```

## How to populate

```bash
# From repo root:
./backend/venv/bin/python3 ml/data/download_nature_2023.py \
    --dest ml/data/nature_2023 \
    --priority real
```

The script is resumable — interrupt and re-run at will. Downloads are
checkpointed per dataset name in `.download_manifest.json`.

## Consumers

| Dataset                    | Used by                                          | How                                         |
|---------------------------|--------------------------------------------------|---------------------------------------------|
| `detection/`               | `ml/train_yolo.py`                               | `--data ../data/nature_lego.yaml`           |
| `classification_real/`     | `ml/train_contrastive.py`                        | `--data-dir ml/data/nature_2023/classification_real` |
| `classification_renders/`  | `ml/train_contrastive.py` (large-batch run)      | `--data-dir ml/data/nature_2023/classification_renders` |

## License

MostWiedzy distributes these datasets under CC BY 4.0. If you publish any
model trained on this corpus, cite the Nature 2023 paper.

## Why this dataset matters for BrickScan

Our existing `ml/data/synthetic_dataset/` (symlinked to ~/Desktop) has ~268k
renders across ~500 classes — plenty of *synthetic* data. What that corpus
lacks is **real photographs**. Nature 2023 ships **52k real photos** with
the same 447-class taxonomy used by the ICCS 2022 benchmark (ResNet-50 @
87.4% top-1 on this set). Fine-tuning on these after synthetic pre-training
is the single biggest lever we have for closing the sim-to-real gap.
