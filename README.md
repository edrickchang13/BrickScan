# BrickScan — LEGO piece scanner & inventory

BrickScan is a mobile app that identifies LEGO pieces from a phone camera in
real time and keeps a personal inventory of what you own. Point the camera at
a brick, a pile of bricks, or a whole desk of loose parts — the app returns
part numbers, colors, and names, and lets you tap-to-add to your collection.

Underneath the camera view is a multi-source recognition cascade that
combines an on-device YOLO detector, a local CNN classifier, a CLIP-based
k-NN catalog voter, a Gemini vision fallback, and a feedback flywheel that
retrains on user corrections. Scans happen on-device when possible and
fall back to the backend only when confidence is low.

## Features

### Recognition

- **Continuous live scanning** — streaming camera view with per-piece bbox
  tracking, Kalman-smoothed overlays, and tap-to-confirm. Persists session
  state, so closing and reopening the app restores the in-progress scan.
- **Multi-piece detection** — point the camera at a pile of bricks and the
  YOLOv8 detector returns a bounding box + classification for each one.
- **On-device inference (Phase 5)** — int8-quantized YOLOv8-L (58 MB) runs
  locally via `onnxruntime-react-native`, with transparent fallback to the
  backend when the on-device model is disabled or fails. Toggle in
  Settings → *On-Device Detection*.
- **Hybrid recognition cascade** — every scan is evaluated by up to four
  independent sources and their predictions are merged with source-specific
  confidence calibration:
  - **Brickognize API** (external classifier, bulk of accuracy today)
  - **Gemini 2.5** (vision-language fallback, optional grounded-prompt mode)
  - **Local ONNX classifier** (MobileNetV3 trained on our pipeline)
  - **CLIP catalog k-NN voter** (similarity-based embeddings)
- **Feedback flywheel** — every scan where the user corrects the top pick
  is logged as a labeled training example; the model is periodically
  retrained on confirmed corrections via `ml/scripts/retrain_from_feedback.py`.
- **Grad-CAM explainability** — see where the model is actually looking
  in the scanned image.

### Collection management

- **Inventory tracking** — parts, colors, quantities, with filter/sort/search.
- **Local-only mode** — the entire app runs against a local SQLite inventory
  without any account or cloud dependency (toggle in Settings).
- **BrickLink wanted-list export** — generate a BrickLink-compatible XML
  from your missing-pieces list for any LEGO set (in progress).
- **Set browsing** — search the Rebrickable catalog (76K+ parts, 26K+ sets,
  16K+ minifigs), view part lists, and check completion against your
  inventory.

### Inference quality infrastructure

- **Env-gated feature flags** — A/B test improvements in production without
  redeploying. All default off, enable per-deployment:
  - `SCAN_TTA_ENABLED` — test-time augmentation (4× compute, ~1-2% accuracy)
  - `SCAN_GROUNDED_GEMINI` — pass classifier's top-3 candidates to Gemini
    as disambiguation hints instead of open-set classification
  - `SCAN_COLLAPSE_VARIANTS` — regex-based mold/print variant merging
    (3001a/3001b/3001pr0001 → 3001)
  - `SCAN_COLOR_RERANK` — downweight candidates whose catalog color is far
    from the scan's dominant RGB
  - `SCAN_USE_CALIBRATION` — apply per-source temperature scaling from
    `backend/data/calibration_temperatures.json`
  - `SCAN_ALWAYS_RUN_GEMINI` — bypass the "skip Gemini if Brickognize
    confident" optimization
- **Per-source temperature calibration** — fit a scalar temperature per
  recognition source against the feedback eval set so reported confidences
  actually match accuracy (`ml/scripts/calibrate_temperatures.py`).
- **Eval harness** — `ml/scripts/eval_against_feedback.py` replays feedback
  images against the live backend and emits top-1/top-3 accuracy broken
  down by source, with confusion pair analysis.
- **A/B runner** — `ml/scripts/run_ab_eval.sh` sweeps the flag space
  (baseline / grounded / collapse / color / all-on) and emits a markdown
  comparison report.

## Tech stack

### Backend
- **FastAPI** (Python 3.11) + async SQLAlchemy 2.0
- **PostgreSQL 16** for the catalog + user data
- **Redis 7** for rate-limiting and response caching
- **ONNX Runtime** for local model inference
- **Alembic** for schema migrations
- **Pytest** (pinned to 8.3.5) for the regression suite

### Mobile
- **React Native** (Expo SDK 55) with TypeScript
- **Zustand** for state management (lightweight, no Redux boilerplate)
- **onnxruntime-react-native 1.24.3** for on-device YOLO inference
- **jpeg-js** + **expo-asset** for the frame preprocessing pipeline
- **Jest** unit tests (16/16 green for the ML pipeline — letterbox, RGBA→NCHW, NMS)

### ML pipeline
- **PyTorch 2.x** + torchvision for classifier training
- **Ultralytics YOLOv8** for multi-piece detection
- **open_clip_torch** (CLIP ViT-B/32) for catalog embeddings
- **onnxruntime.quantization** for int8 QDQ quantization
- **Blender 4** + LDraw for synthetic training renders
- **NVIDIA DGX Spark (GB10, 128 GB VRAM)** for training runs

### Infrastructure
- **Docker Compose** for the local dev stack (backend + db + redis + Adminer)
- **GitHub Actions** for lint + type-check + test CI
- **Tailscale** for remote Spark access

## Quick start

```bash
git clone <repository-url>
cd brickscan

# 1. Environment
cp backend/.env.example backend/.env
# Fill in REBRICKABLE_API_KEY, BRICKLINK_API_KEY (optional), GEMINI_API_KEY
# (optional), JWT_SECRET_KEY (openssl rand -hex 32)

# 2. Bring up services
make up

# 3. Schema
make migrate

# 4. Rebrickable catalog import (one-time, ~2 min)
cd data_pipeline && ./download_rebrickable.sh ./rebrickable_data && cd ..
make import-data && make verify-data

# 5. Mobile
make install-mobile
cd mobile && npx expo start          # Metro bundler
cd mobile/ios && pod install          # iOS native deps (needed after Phase 5)
npx expo run:ios                      # build + launch to simulator
```

## API

Interactive docs once the backend is running:
- **Swagger UI** — http://localhost:8000/docs
- **ReDoc** — http://localhost:8000/redoc

### Route layout

The backend has two parallel route trees reflecting the app's hybrid mode:

| Prefix | Purpose |
|---|---|
| `/auth` | Registration, login, JWT refresh |
| `/api/scan` | Legacy single-piece classifier endpoint |
| `/api/inventory` | User inventory CRUD (requires auth) |
| `/api/parts`, `/api/sets` | Rebrickable catalog search |
| `/api/local-inventory/scan` | Hybrid cascade single-piece scan |
| `/api/local-inventory/scan-multi` | Multi-piece detect + classify |
| `/api/local-inventory/scan-video` | Temporal-aggregation video scan |
| `/api/local-inventory/feedback` | Feedback flywheel ingest + eval export |
| `/bricklink` | BrickLink price lookup + wanted-list XML |

## ML training

Training scripts live in `ml/` and run on the DGX Spark. The full pipeline:

```bash
# On Spark:
ssh spark

# 1. Build merged training set (HF synthetic + Nature 2023 real + labeled)
python3 ~/brickscan/ml/merge_all_datasets.py \
    --output-dir ~/brickscan/ml/training_data/merged \
    --hf-dir ~/brickscan/ml/training_data/huggingface_legobricks/images \
    --nature-dir ~/brickscan/ml/training_data/nature_2023_real

# 2. Train MobileNetV3-Large classifier
python3 ~/brickscan/ml/train_mobilenetv3_gpu.py \
    --data-dir ~/brickscan/ml/training_data/merged \
    --output-dir ~/brickscan/ml/output/mobilenetv3_$(date +%Y%m%d) \
    --epochs 25 --batch-size 384 --model-size large

# 3. Precompute CLIP catalog embeddings for the k-NN voter
python3 ~/brickscan/ml/precompute_clip_gpu.py \
    --data-dir ~/brickscan/ml/training_data/merged \
    --output ~/brickscan/ml/output/embeddings_cache.pkl

# 4. Quantize the YOLO detector for on-device use
backend/venv/bin/python ml/export/quantize_yolo_int8.py \
    --input  backend/models/yolo_lego.onnx \
    --output backend/models/yolo_lego.int8.onnx \
    --calib-dir /path/to/yolo_dataset/images/val
```

### Current model versions

| Artifact | Path | Size | Notes |
|---|---|---|---|
| Local classifier | `backend/models/lego_classifier.onnx` + `.data` | 18 MB | MobileNetV3-Small, 502 classes, 91.6% cross-domain top-1 |
| Multi-piece detector (server) | `backend/models/yolo_lego.onnx` | 167 MB | YOLOv8-L, 28 classes (size, color). Not tracked in git. |
| On-device detector | `mobile/assets/models/yolo_lego.int8.onnx` | 58 MB | int8 QDQ-quantized YOLOv8-L. 0.964 mean IoU@matched vs fp32. |
| CLIP catalog embeddings | `backend/data/embeddings_cache.pkl` | 2 MB | 1072 classes × 512-dim |
| Per-source calibration | `backend/data/calibration_temperatures.json` | <1 KB | Temperatures fit from feedback eval set |

## Testing

```bash
# Backend (runs in the docker container, pytest 8.3.5 pinned)
make test-backend

# Mobile (Jest unit tests)
cd mobile && npx jest

# ML pipeline specifically
cd mobile && npx jest src/__tests__/ml/
# → 16/16 green: letterbox math, RGBA→NCHW conversion, NMS, unletterbox

# Backend flag-gated cascade tests
docker-compose exec -T backend pytest \
    tests/test_calibration.py \
    tests/test_cascade_flags.py \
    tests/test_eval_set_export.py
```

The backend test suite had a bootstrap bug (`conftest.py` importing
`app.main` that never existed) that was fixed in commit `47b1710`. The pre-existing
`test_auth.py` / `test_inventory.py` / `test_scan*.py` suites still have
known breakage (UUID columns vs SQLite in-memory; sync tests requesting async
fixtures) — tracked as separate work items.

## Development workflow

```bash
# Start/stop dev stack
make up        # Services on 8000 (backend), 5432 (db), 6379 (redis), 8080 (adminer)
make down

# Logs + shells
make logs
make backend-shell
make db-shell

# Lint & format
make lint-backend
docker-compose exec backend black app/ tests/

# Migrations
docker-compose exec backend alembic revision -m "description"
docker-compose exec backend alembic downgrade -1

# Mobile
cd mobile && npx tsc --noEmit         # type check
cd mobile && npx prettier --write src/
```

## Project layout

```
brickscan/
├── backend/
│   ├── main.py                      # FastAPI app entry point
│   ├── app/
│   │   ├── api/                     # Authenticated REST routes
│   │   ├── local_inventory/         # Local-mode scan + feedback routes
│   │   ├── services/
│   │   │   ├── hybrid_recognition.py   # Cascade + env-gated flags
│   │   │   ├── brickognize_client.py
│   │   │   ├── gemini_service.py
│   │   │   ├── ml_inference.py         # ONNX runner for local classifier
│   │   │   ├── part_num_normalizer.py  # Mold/variant collapse
│   │   │   └── color_extractor.py      # Dominant-color histogram
│   │   ├── ml/                      # EmbeddingLibrary (CLIP k-NN)
│   │   └── core/                    # Auth, DB, config
│   ├── models/                      # ONNX artifacts (gitignored except canonical)
│   ├── data/                        # Calibration temps, CLIP embeddings
│   ├── tests/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── requirements-dev.txt
│
├── mobile/
│   ├── src/
│   │   ├── screens/
│   │   │   ├── ContinuousScanScreen.tsx  # Live streaming scan
│   │   │   ├── ScanScreen.tsx
│   │   │   ├── InventoryScreen.tsx
│   │   │   └── SettingsScreen.tsx        # On-device / local-only toggles
│   │   ├── ml/                           # Phase 5 on-device pipeline
│   │   │   ├── preprocess.ts             # Letterbox + NCHW Float32
│   │   │   ├── postprocess.ts            # YOLOv8 head decode + NMS
│   │   │   ├── yoloDetector.ts           # onnxruntime session wrapper
│   │   │   └── scanPipeline.ts           # JPEG → RGBA → detect → pieces
│   │   ├── store/                        # Zustand stores (auth, inventory)
│   │   ├── utils/
│   │   │   ├── bboxTracker.ts            # IoU-based cross-frame tracking
│   │   │   ├── settingsFlags.ts          # AsyncStorage flag keys
│   │   │   └── scanSession.ts            # Session persistence
│   │   └── services/                     # apiClient + screen helpers
│   ├── assets/models/                    # int8 YOLO bundled into the app
│   ├── __tests__/ml/                     # Jest unit tests
│   └── app.json
│
├── ml/
│   ├── scripts/                     # Training, eval, calibration scripts
│   ├── export/                      # ONNX export + int8 quantization
│   ├── training/                    # Training pipelines (incl. YOLO distillation)
│   ├── blender/                     # Synthetic render pipeline
│   └── data/                        # Training data symlinks (all gitignored)
│
├── data_pipeline/                   # Rebrickable CSV import utilities
├── docker-compose.yml
├── Makefile
└── docs/                            # Phase 5 status, design notes
```

## Getting API keys

| Service | URL | What it buys you |
|---|---|---|
| Rebrickable | https://rebrickable.com/api/ | Free catalog data: 76K+ parts, 26K+ sets, 16K+ minifigs |
| BrickLink | https://www.bricklink.com | Free price guide data (OAuth 1.0a) — requires seller account |
| Google Gemini | https://ai.google.dev | Vision-language fallback for the recognition cascade |

Add each to `backend/.env`:
```
REBRICKABLE_API_KEY=...
BRICKLINK_CONSUMER_KEY=...
BRICKLINK_CONSUMER_SECRET=...
BRICKLINK_TOKEN_VALUE=...
BRICKLINK_TOKEN_SECRET=...
GEMINI_API_KEY=...
JWT_SECRET_KEY=...   # openssl rand -hex 32
```

## Troubleshooting

```bash
# Backend won't start: check the stack
docker-compose ps
docker-compose logs backend

# Database fresh reset (destroys data)
docker-compose down -v
make up && make migrate && make import-data

# Mobile build fails after Phase 5 changes
cd mobile && rm -rf node_modules && npm install
cd ios && pod install && cd ..
npx expo start --clear

# ONNX model missing on backend startup
# The canonical classifier is tracked at backend/models/lego_classifier.onnx.
# The YOLO detector (backend/models/yolo_lego.onnx, 167 MB) is NOT tracked —
# regenerate with `ml/export/to_onnx.py` or download from the team's model
# registry. The int8 derivative ships via git LFS / direct commit.
```

## What's not in this repo

- Trained `.pt` checkpoints (live on the Spark, not in git)
- The 167 MB fp32 YOLO ONNX (exceeds GitHub's 100 MB limit — only the 58 MB
  int8 derivative is tracked)
- Training datasets (Rebrickable CSVs, HuggingFace LEGO Bricks, Nature 2023
  real photos — all regenerable via the download scripts)
- Spark-side training artifacts (`backend/models/spark_*/` is gitignored)

## CI/CD

GitHub Actions run on every push:
- `backend_ci.yml` — ruff + black lint, mypy, pytest against PostgreSQL + Redis
- `mobile_ci.yml` — TypeScript type-check, Jest unit tests, ESLint

## License

MIT.

## Support

Issues, feature requests, bug reports: GitHub Issues on this repo.
