# DGX Spark Setup - Complete File Index

Complete reference for all BrickScan DGX Spark setup files.

## Quick Navigation

- **Start here**: [QUICKSTART.md](QUICKSTART.md) - 30-minute setup
- **Full docs**: [README.md](README.md) - Complete guide with troubleshooting
- **This file**: [INDEX.md](INDEX.md) - File reference

## Directory Structure

```
dgx_spark/
├── README.md                              (14 KB) Full documentation
├── QUICKSTART.md                          (3 KB)  30-minute setup
├── INDEX.md                               (This file)
│
├── setup/                                 Initial DGX Spark setup
│   └── install_dependencies.sh            (5 KB)  Install Ollama, FastAPI, Blender, Python
│
├── vision_server/                         Local vision model API
│   ├── server.py                          (13 KB) FastAPI app for LEGO identification
│   ├── start_server.sh                    (2 KB)  Run server manually
│   ├── systemd_service.sh                 (3 KB)  Auto-start on boot
│   └── requirements.txt                   (0.3 KB) Python dependencies
│
├── backend_integration/                   Backend connection
│   └── dgx_vision_service.py              (7 KB)  Drop-in Gemini API replacement
│
├── training/                              ML model training
│   ├── setup_ml_env.sh                    (3 KB)  Install PyTorch + ML libs
│   └── train_on_dgx.sh                    (6 KB)  Run training on GPU
│
├── rendering/                             Synthetic data generation
│   └── blender_dgx_render.sh              (5 KB)  GPU-accelerated Blender
│
└── network/                               Network utilities
    ├── find_dgx.sh                        (3 KB)  Discover DGX on local network
    └── test_connection.py                 (5 KB)  Benchmark vision server
```

## File Descriptions

### Documentation

#### `README.md` (14 KB)
Complete setup guide covering:
- Hardware prerequisites
- Part 1: Local vision model setup
- Part 2: Blender GPU rendering
- Part 3: PyTorch training
- Network configuration
- Backend integration
- Troubleshooting section
- Performance benchmarks

#### `QUICKSTART.md` (3 KB)
Fast path to working vision server in 30 minutes:
- 5 steps from unboxing to inference
- Troubleshooting quick reference
- File guide and next steps

#### `INDEX.md` (This file)
Complete file reference with descriptions and usage.

---

### Setup Scripts

#### `setup/install_dependencies.sh` (5 KB)
**Runs on**: DGX Spark
**What it does**:
- Updates system packages
- Installs Ollama (LLM runtime)
- Pulls LLaVA 13B and Moondream vision models (~10GB)
- Creates Python virtual environment
- Installs FastAPI + dependencies
- Installs Blender headless

**Time**: 15 minutes (mostly downloading models)

**Usage**:
```bash
bash setup/install_dependencies.sh
```

---

### Vision Server (Local AI Fallback)

#### `vision_server/server.py` (13 KB)
**Runs on**: DGX Spark (port 8001)
**Purpose**: FastAPI server exposing local vision inference

**Features**:
- POST `/identify` - Identify LEGO piece from base64 image
- GET `/health` - Check server and Ollama status
- GET `/models` - List available models
- POST `/pull-model` - Download models from registry

**API Response**:
```json
{
  "predictions": [
    {"part_num": "3001", "part_name": "Brick 2 x 4", "color_name": "Red", "confidence": 0.95}
  ],
  "model_used": "llava:13b",
  "processing_time_ms": 1250
}
```

**Implementation Details**:
- Uses Ollama API internally for model inference
- Automatically retries failed requests
- Parses vision model JSON output
- Handles image base64 encoding/decoding
- Includes detailed logging and error handling

---

#### `vision_server/start_server.sh` (2 KB)
**Runs on**: DGX Spark
**Purpose**: Start the vision server manually

**Usage**:
```bash
cd vision_server
bash start_server.sh              # Default port 8001
PORT=9000 bash start_server.sh   # Custom port
```

**Output**: Server runs on `http://<dgx-ip>:8001`

---

#### `vision_server/systemd_service.sh` (3 KB)
**Runs on**: DGX Spark (with sudo)
**Purpose**: Install vision server as systemd service (auto-start on boot)

**Usage**:
```bash
sudo bash systemd_service.sh
```

**Enables**:
```bash
sudo systemctl status brickscan-vision
sudo systemctl restart brickscan-vision
sudo journalctl -u brickscan-vision -f
```

---

#### `vision_server/requirements.txt` (0.3 KB)
Python dependencies for vision server:
- fastapi (web framework)
- uvicorn (ASGI server)
- httpx (async HTTP client)
- pillow (image processing)
- numpy (numeric arrays)
- pydantic (data validation)

---

### Backend Integration

#### `backend_integration/dgx_vision_service.py` (7 KB)
**Usage location**: Drop into BrickScan backend
**Purpose**: Replace Gemini API calls with local inference

**Key Functions**:
- `identify_piece()` - Main inference function (identical API to gemini_service.py)
- `health_check()` - Verify DGX is reachable
- `list_available_models()` - Get Ollama models
- `pull_model()` - Download models to DGX

**Drop-in Replacement**:
```python
# Before
from app.vision.gemini_service import identify_piece

# After
from app.vision.dgx_vision_service import identify_piece

# No code changes needed - same function signature
```

**Configuration** (backend `.env`):
```bash
DGX_VISION_URL=http://192.168.1.100:8001
VISION_BACKEND=dgx
```

---

### Model Training

#### `training/setup_ml_env.sh` (3 KB)
**Runs on**: DGX Spark
**Purpose**: Install PyTorch and ML libraries

**Installs**:
- PyTorch with CUDA (ARM64-optimized from NVIDIA)
- Training: timm, albumentations, wandb
- Processing: opencv, pillow, numpy
- Export: coremltools, onnx

**Time**: 5-10 minutes

**Usage**:
```bash
bash training/setup_ml_env.sh
source ~/brickscan-ml-env/bin/activate
```

---

#### `training/train_on_dgx.sh` (6 KB)
**Runs on**: DGX Spark
**Purpose**: Train LEGO classification model on GPU

**What it does**:
- Activates ML environment
- Validates training data
- Counts classes and samples
- Runs training with optimal DGX settings
- Logs to `training_log_TIMESTAMP.txt`
- Saves checkpoints to `checkpoints/`

**Time**: 2-4 hours for 3000 classes (vs 12h+ on laptop)

**Configuration**:
```bash
EPOCHS=50 BATCH_SIZE=64 LR=0.001 bash training/train_on_dgx.sh
```

**Required**: Training data at `~/brickscan-training/data/train/{part_num}/{images}`

---

### Data Generation

#### `rendering/blender_dgx_render.sh` (5 KB)
**Runs on**: DGX Spark
**Purpose**: Generate synthetic LEGO training data with Blender GPU rendering

**What it does**:
- Scans LDraw parts library
- Renders each part in multiple poses (80 images default)
- Uses GPU for 10-20x speedup
- Saves to `~/brickscan-training/data/synthetic/`

**Time**: ~5-10 seconds per render (DGX GPU)

**Performance**:
- 80 images per part: 7-13 minutes
- 3000 parts: 30-40 hours

**Usage**:
```bash
NUM_RENDERS=100 MAX_WORKERS=4 bash rendering/blender_dgx_render.sh
```

---

### Network Utilities

#### `network/find_dgx.sh` (3 KB)
**Runs on**: Your Mac
**Purpose**: Discover DGX Spark IP on local network

**Features**:
- Scans local network for port 8001 (vision server)
- Uses nmap if available (faster)
- Falls back to sequential scanning
- Returns DGX IP and server status

**Usage**:
```bash
bash network/find_dgx.sh
# Output: Found BrickScan Vision Server at: 192.168.1.100:8001
```

---

#### `network/test_connection.py` (5 KB)
**Runs on**: Your Mac
**Purpose**: Test DGX vision server and benchmark inference

**Features**:
- Check server health
- List available models
- Benchmark inference with real image
- Report timing statistics

**Usage**:
```bash
python3 network/test_connection.py --url http://192.168.1.100:8001
python3 network/test_connection.py --url http://192.168.1.100:8001 --image lego_brick.jpg --runs 5
```

**Output**:
```
Server health: healthy
Available models: ['llava:13b', 'moondream']

Inference Benchmark (5 runs):
  Run 1... OK (1.23s)
  Run 2... OK (1.18s)
  ...
  Average: 1.21s
  Success rate: 5/5
```

---

## Setup Workflow

### 1. Initial Setup (15 min)
```bash
ssh ubuntu@<dgx-spark-ip>
cd ~/brickscan/dgx_spark
bash setup/install_dependencies.sh
```

### 2. Start Vision Server (2 min)
```bash
cd vision_server
bash start_server.sh
# Keep this running or install as service
sudo bash systemd_service.sh
```

### 3. Verify from Mac (5 min)
```bash
bash network/find_dgx.sh
python3 network/test_connection.py --url http://<dgx-ip>:8001
```

### 4. Configure Backend (2 min)
Update `.env`:
```bash
DGX_VISION_URL=http://192.168.1.100:8001
VISION_BACKEND=dgx
```

### 5. Optional: Train Models (2-4 hours)
```bash
bash training/setup_ml_env.sh
bash training/train_on_dgx.sh
```

### 6. Optional: Generate Synthetic Data (30-40 hours)
```bash
bash rendering/blender_dgx_render.sh
```

---

## Performance Benchmarks

| Operation | Time | Hardware |
|-----------|------|----------|
| Single piece identification | 1.2-1.8s | DGX Spark GPU |
| 10 pieces (parallel) | 2.5-3.5s | DGX Spark GPU |
| Cold start (model load) | 3-5s | First request |
| Train 3000-class model | 2-4 hours | DGX Spark GPU |
| Laptop training | 12-18 hours | MacBook Pro M4 |
| Synthetic render (1 part) | 5-10 seconds | DGX Spark GPU |
| Full dataset rendering | 30-40 hours | DGX Spark GPU |

---

## Troubleshooting Matrix

### Problem → Solution

**Vision server won't start**
→ Check Ollama: `sudo systemctl status ollama`
→ Pull models: `ollama pull llava:13b`

**Cannot find DGX on network**
→ Check WiFi: `ping 192.168.1.1`
→ Get IP: SSH to DGX, run `hostname -I`

**Inference is slow (>5s)**
→ Check GPU: `nvidia-smi` on DGX
→ Restart Ollama: `sudo systemctl restart ollama`

**Out of memory during training**
→ Reduce batch size in `train_on_dgx.sh`
→ Enable gradient accumulation

**Models not listed**
→ Check downloaded: `ollama list`
→ Pull: `ollama pull llava:13b moondream`

---

## Next Steps

1. Start with [QUICKSTART.md](QUICKSTART.md)
2. Follow detailed [README.md](README.md) for advanced setup
3. Use network tools to verify connectivity
4. Integrate with backend via `dgx_vision_service.py`
5. Scale with synthetic data generation and model training

---

## Support

For issues with:
- **Vision inference**: Check `/vision_server/server.py` logs
- **Model training**: Check `training_log_*.txt`
- **Rendering**: Check Blender output
- **Network**: Run `find_dgx.sh` and `test_connection.py`

See `README.md` troubleshooting section for detailed diagnostics.
