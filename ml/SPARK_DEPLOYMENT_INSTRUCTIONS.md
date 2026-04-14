# DGX Spark ML Training Deployment — Context for Parallel Chat

## What This Is
You're deploying and running ML training pipelines on an NVIDIA DGX Spark (GB10 Blackwell GPU, 130.6GB VRAM, CUDA 13.0, PyTorch 2.12.0, aarch64 Linux).

## SSH Access
- **Tailscale IP:** 100.124.143.19
- **Alias:** "spark" (should be in ~/.ssh/config)
- **User:** Check `ssh spark whoami` — likely `edrick` or default user
- **Working directory on Spark:** `~/brickscan/ml/`

## What's Already Done
1. **HuggingFace dataset downloaded** — 400,000 rendered LEGO images across 1,000 part classes
   - Location on Spark: `~/brickscan/ml/training_data/huggingface_legobricks/`
   - Was saving images organized by class into `images/` subdirectory — check if complete
   - Dataset loaded confirmed: "Dataset loaded: 400000 images, took 478s, Unique classes: 1000"

2. **Rebrickable CSV catalog downloaded** — 61,898 parts metadata
   - Location: `~/brickscan/ml/training_data/rebrickable_csv/`
   - Files: parts.csv, part_categories.csv, colors.csv, elements.csv, part_relationships.csv

3. **Python venv exists** — `~/brickscan/ml/venv/` with PyTorch, timm, etc.
   - Activate: `source ~/brickscan/ml/venv/bin/activate`

## Scripts to Deploy (from this repo)
Copy these from the local repo at `brickscan/ml/` to the Spark at `~/brickscan/ml/`:

### 1. `train_dinov2.py` — DINOv2 ViT fine-tuning
- Two-stage: frozen backbone warmup (5 epochs) → unfreeze last 4 transformer blocks (30 epochs)
- DINOv2 ViT-B/14 from timm, 518x518 input, AMP, cosine annealing
- Combines HuggingFace + existing sparse dataset
- **To upgrade to ViT-Giant:** Change model name to `vit_giant_patch14_dinov2` and reduce batch size to 16-32
- Exports ONNX at the end
- **Launch command:**
```bash
source ~/brickscan/ml/venv/bin/activate
nohup python3 ~/brickscan/ml/train_dinov2.py \
  --hf-data-dir ~/brickscan/ml/training_data/huggingface_legobricks/images/ \
  --sparse-data-dir ~/brickscan/ml/data/images/ \
  --output-dir ~/brickscan/ml/output/dinov2_$(date +%Y%m%d) \
  --batch-size 64 \
  --epochs 35 \
  --lr 1e-4 \
  > ~/brickscan/ml/logs/dinov2_train.log 2>&1 &
```

### 2. `train_yolo_detector.py` — YOLOv8 multi-piece detector
- Generates synthetic cluttered scenes (3-15 parts on varied backgrounds)
- Creates 5000 train + 1000 val synthetic images in YOLO format
- Trains YOLOv8m single-class detection ("lego_piece")
- 100 epochs, early stopping patience=15, ONNX export
- **Dependencies:** `pip install ultralytics`
- **Launch command:**
```bash
source ~/brickscan/ml/venv/bin/activate
nohup python3 ~/brickscan/ml/train_yolo_detector.py \
  --source-images ~/brickscan/ml/training_data/huggingface_legobricks/images/ \
  --output-dir ~/brickscan/ml/output/yolo_$(date +%Y%m%d) \
  > ~/brickscan/ml/logs/yolo_train.log 2>&1 &
```

### 3. Blender Rendering Pipeline
- Scripts: `blender/render_parts.py` (renders inside Blender), `blender/batch_render.py` (orchestrator)
- Setup: Run `blender/setup_ldraw.sh` first to download the LDraw parts library (~200MB)
- Blender may need to be installed: `sudo apt install blender` or download from blender.org
- **Check if Blender is available:** `which blender` or `blender --version`
- **Launch command (after LDraw setup):**
```bash
# First, get LDraw library
cd ~/brickscan/ml && bash blender/setup_ldraw.sh

# Then get Rebrickable colors
cp ~/brickscan/ml/training_data/rebrickable_csv/colors.csv ~/brickscan/ml/data/colors.csv

# Render top 500 parts × 10 popular colors × 36 angles × 3 elevations = ~540,000 images
nohup python3 ~/brickscan/ml/blender/batch_render.py \
  --blender $(which blender) \
  --ldraw-dir ~/brickscan/ml/data/ldraw/ldraw \
  --colors-csv ~/brickscan/ml/data/colors.csv \
  --output-dir ~/brickscan/ml/data/renders \
  --num-angles 36 \
  --resolution 518 \
  --workers 2 \
  --colors 0 1 4 5 7 14 15 19 70 71 \
  --skip-existing \
  > ~/brickscan/ml/logs/blender_render.log 2>&1 &
```
Note: Color IDs are Rebrickable IDs — 0=Black, 1=Blue, 4=Red, 5=Orange, 7=Green, 14=Yellow, 15=White, 19=Tan, 70=Dark Brown, 71=Light Gray

## Priority Order
1. **Check HF image saving status** — `ls ~/brickscan/ml/training_data/huggingface_legobricks/images/ | wc -l` should show 1000 class directories
2. **Deploy all scripts** via SCP
3. **Install missing deps** — `pip install ultralytics tqdm` in the venv
4. **Start Blender setup** (LDraw download) — this is the bottleneck since rendering takes hours
5. **Launch DINOv2 training** — can start immediately on HF data
6. **Launch YOLO training** — can start after DINOv2 is running (they'll share GPU via AMP)
7. **Launch Blender rendering** — CPU-bound, runs alongside GPU training

## Monitoring
```bash
# Check GPU utilization
nvidia-smi

# Watch training logs
tail -f ~/brickscan/ml/logs/dinov2_train.log
tail -f ~/brickscan/ml/logs/yolo_train.log
tail -f ~/brickscan/ml/logs/blender_render.log

# Check running jobs
ps aux | grep python3
```

## Known Issues
- Rebrickable render downloads mostly 404'd — don't bother retrying those
- The HF dataset uses integer labels (0-999), the class mapping is in the dataset metadata
- DGX Spark is aarch64 — some pip packages may need `--no-binary` flag
- Blender on headless Linux needs `--background` flag (already in scripts)
- SSH via Tailscale may need `ssh edrick@100.124.143.19` if alias isn't configured

## What Success Looks Like
- DINOv2 training running, loss decreasing over epochs, val accuracy climbing
- YOLO training generating synthetic scenes then training, mAP improving
- Blender rendering ~15 images/minute per worker, accumulating in data/renders/
- All three processes coexisting on the Spark without OOM errors
