# BrickScan DGX Spark Setup Guide

Complete setup for running BrickScan's local AI vision model and GPU-accelerated Blender rendering on NVIDIA's DGX Spark personal supercomputer.

## Overview

The DGX Spark (Grace Blackwell, 128GB unified memory) serves two critical functions:

1. **Local Vision Model** - Replace Gemini API calls with fast, private on-device inference using LLaVA/Moondream via Ollama
2. **GPU-Accelerated Rendering** - Generate synthetic training data 10-20x faster than CPU rendering

## Prerequisites

### Hardware
- **DGX Spark** with:
  - Ubuntu 22.04 LTS or later (via JetPack)
  - 128GB unified memory
  - GB10 Superchip (Grace + Blackwell)
  - Connected to same network as your Mac/development machine
  - Static IP address recommended (see network configuration)

### Network
- Mac and DGX Spark on same local network
- DGX Spark has reliable WiFi or ethernet
- Port 8001 open between devices (vision API)
- Port 8000 optional (Blender render API)

### Software on DGX Spark
- Ubuntu 22.04+ (comes with JetPack)
- NVIDIA JetPack 6.0+ with CUDA Toolkit
- Internet connection for initial setup

## Part 1: Setting Up Local Vision Model

### 1.1 Initial Setup on DGX Spark

SSH into your DGX Spark:
```bash
ssh ubuntu@<dgx-spark-ip>
cd ~
```

Run the dependency installation script:
```bash
# Copy the setup script to DGX Spark first
# From your Mac: scp setup/install_dependencies.sh ubuntu@<dgx-ip>:~/
bash install_dependencies.sh
```

This will:
- Update system packages
- Install Ollama (local LLM/vision model runtime)
- Download LLaVA 13B and Moondream vision models
- Create Python virtual environment
- Install FastAPI and dependencies

**Expected time**: 10-15 minutes (mostly downloading models)

### 1.2 Start the Vision Server

On the DGX Spark:
```bash
# Navigate to vision_server directory
cd ~/brickscan-vision-server

# Make startup script executable
chmod +x start_server.sh

# Start the server (runs in foreground for now)
./start_server.sh
```

You should see:
```
Starting BrickScan Vision Server on 192.168.x.x:8001
INFO:     Uvicorn running on http://0.0.0.0:8001
```

### 1.3 Test Vision Server from Mac

From your Mac, find the DGX Spark IP:
```bash
bash network/find_dgx.sh
```

Or manually test:
```bash
# Replace 192.168.x.x with your DGX Spark IP
curl http://192.168.x.x:8001/health
```

Should return JSON with available models:
```json
{
  "status": "healthy",
  "available_models": ["llava:13b", "moondream"]
}
```

### 1.4 Install as System Service (Auto-Start)

To run vision server automatically on DGX boot:
```bash
# On DGX Spark
cd ~/brickscan-vision-server
sudo bash systemd_service.sh
```

This creates `/etc/systemd/system/brickscan-vision.service`. Verify:
```bash
sudo systemctl status brickscan-vision
```

### 1.5 Connect Backend to DGX Vision

In your BrickScan backend `.env`:
```bash
# Replace with your DGX Spark IP
DGX_VISION_URL=http://192.168.1.100:8001
VISION_BACKEND=dgx
```

The backend will now use the local vision model instead of Gemini for all piece identification. Zero code changes needed—same interface.

## Part 2: Setting Up Blender GPU Rendering

### 2.1 Verify Blender Installation

The `install_dependencies.sh` script installs Blender headless. Verify:
```bash
# On DGX Spark
blender --version
```

Should output: `Blender 4.1.0`

### 2.2 Enable GPU Rendering

Blender on DGX Spark can use NVIDIA GPU for rendering. This is configured in the rendering script automatically, but verify in Blender preferences if rendering manually:
- Preferences → Render → Cycle Render Devices → CUDA

### 2.3 Generate Synthetic Training Data

```bash
# On DGX Spark
# First, download LDraw part library (required once)
mkdir -p ~/ldraw
cd ~/ldraw
wget https://library.ldraw.org/latest-parts.html -O parts.zip
unzip parts.zip

# Now render training data
cd ~/brickscan-training
chmod +x rendering/blender_dgx_render.sh
./rendering/blender_dgx_render.sh
```

Configuration options (pass as environment variables):
```bash
LDRAW_DIR=~/ldraw \
NUM_RENDERS=80 \
MAX_WORKERS=4 \
./rendering/blender_dgx_render.sh
```

Expected performance:
- **Per render**: 5-10 seconds on DGX Spark GPU (vs 60-120s on laptop CPU)
- **80 images/part**: ~7-13 minutes per part
- **Full synthetic dataset (3000 parts)**: ~30-40 hours (run overnight)

### 2.4 Monitor Rendering Progress

In another terminal on DGX:
```bash
# Watch rendering directory size grow
watch -n 5 'find ~/brickscan-training/data/synthetic -name "*.png" | wc -l'
```

Or check individual part status:
```bash
ls -lh ~/brickscan-training/data/synthetic/ | head -20
```

## Part 3: Setting Up PyTorch Training Environment

### 3.1 Install ML Dependencies

On DGX Spark:
```bash
bash training/setup_ml_env.sh
```

This installs:
- PyTorch with CUDA support (ARM64 wheels from NVIDIA)
- Training dependencies: timm, albumentations, wandb, etc.
- Export tools: coremltools, onnx

**Expected time**: 5-10 minutes

### 3.2 Prepare Training Data

Ensure you have training data:
```bash
# Training data structure:
# ~/brickscan-training/data/train/
#   ├── 3001/  (part numbers as class names)
#   │   ├── image1.jpg
#   │   ├── image2.jpg
#   │   └── ...
#   ├── 3002/
#   └── ...

# Count training samples
find ~/brickscan-training/data/train -type f | wc -l
find ~/brickscan-training/data/train -type d | wc -l  # Number of classes
```

### 3.3 Run Training on DGX Spark

```bash
cd ~/brickscan-training
source ~/brickscan-ml-env/bin/activate
bash training/train_on_dgx.sh
```

This will:
- Check dataset completeness
- Start training with optimized DGX settings
- Log to `training_log_YYYYMMDD_HHMMSS.txt`
- Save checkpoints to `checkpoints/`

**Expected training time**: 2-4 hours for full 3000-class LEGO set (vs 1-2 days on CPU)

Monitoring training:
```bash
# In another terminal, tail the log
tail -f ~/brickscan-training/training_log_*.txt

# Or monitor resource usage
nvidia-smi -l 1  # Update every 1 second
```

### 3.4 Export Trained Model

After training completes:
```bash
# Export to CoreML (for iOS app)
python export/to_coreml.py --checkpoint checkpoints/best.pt

# Export to ONNX (for Android/web)
python export/to_onnx.py --checkpoint checkpoints/best.pt

# Export to TorchScript (for PyTorch Lite)
python export/to_torchscript.py --checkpoint checkpoints/best.pt
```

Models are saved to `exports/` directory.

## Network Configuration

### Finding Your DGX Spark IP

```bash
# On the DGX Spark terminal
hostname -I

# Or from Mac, scan your local network
bash network/find_dgx.sh
```

### Recommended: Static IP Address

To ensure consistent connectivity, assign a static IP to the DGX Spark:

**Via DHCP reservation** (recommended):
1. Note DGX Spark MAC address: `ip addr show`
2. Log into your router (192.168.1.1 or similar)
3. Find DHCP settings
4. Reserve an IP (e.g., 192.168.1.100) for the DGX Spark's MAC address

**Or manually on DGX Spark** (if needed):
```bash
# Edit netplan config (Ubuntu)
sudo nano /etc/netplan/00-installer-config.yaml
```

Example static IP config:
```yaml
network:
  version: 2
  ethernets:
    eth0:
      dhcp4: no
      addresses:
        - 192.168.1.100/24
      gateway4: 192.168.1.1
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]
```

Then apply:
```bash
sudo netplan apply
```

### Testing Network Connectivity

From Mac:
```bash
# Ping DGX Spark
ping 192.168.1.100

# Test SSH
ssh ubuntu@192.168.1.100

# Test vision API
curl http://192.168.1.100:8001/health
```

## Backend Integration

The BrickScan backend connects to DGX Spark via two APIs:

### Vision API (Port 8001)

**Endpoint**: `POST /identify`

Request:
```json
{
  "image_base64": "iVBORw0KGgoAAAANS...",
  "top_k": 3
}
```

Response:
```json
{
  "predictions": [
    {
      "part_num": "3001",
      "part_name": "Brick 2 x 4",
      "color_name": "Red",
      "confidence": 0.95
    }
  ],
  "model_used": "llava:13b",
  "processing_time_ms": 1250
}
```

**Backend code** (`backend_integration/dgx_vision_service.py`):
```python
from app.vision.dgx_vision_service import identify_piece

# In your API endpoint:
predictions = await identify_piece(image_bytes)
```

### Health Check

```bash
# From Mac or backend server
curl http://192.168.1.100:8001/health
```

Returns:
```json
{
  "status": "healthy",
  "available_models": ["llava:13b", "moondream"]
}
```

## Troubleshooting

### Vision Server Not Reachable

1. Check DGX Spark is on and on the network
2. Verify Ollama is running: `sudo systemctl status ollama`
3. Check vision server: `sudo systemctl status brickscan-vision`
4. View logs: `sudo journalctl -u brickscan-vision -f`
5. Test locally on DGX: `curl http://localhost:8001/health`

### Ollama Models Not Downloaded

```bash
# On DGX Spark
ollama list  # Check what's installed

# If missing, pull manually
ollama pull llava:13b
ollama pull moondream
```

### Blender Rendering Too Slow

- Check only 1 render instance running: `ps aux | grep blender`
- Verify GPU is being used: `nvidia-smi` during render
- Reduce `NUM_RENDERS` for initial testing
- Check output directory has space: `df -h ~/brickscan-training/data/synthetic`

### PyTorch CUDA Not Available

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If False:
- Check NVIDIA drivers: `nvidia-smi`
- Reinstall PyTorch from NVIDIA wheels: `pip install --upgrade torch torchvision --index-url https://pypi.ngc.nvidia.com`

### Out of Memory During Training

Reduce batch size in `training/config.yaml`:
```yaml
training:
  batch_size: 32  # Default 128, try 64 or 32
  gradient_accumulation: 2  # Accumulate gradients over 2 steps
```

## Performance Benchmarks

### Vision Inference (LLaVA 13B on DGX Spark)

| Task | Time |
|------|------|
| Single piece identification | 1.2-1.8s |
| Batch 10 pieces (parallel) | 2.5-3.5s |
| Ollama model load (first request) | 3-5s |

### Blender Rendering (Synthetic Data)

| Configuration | Time per Render |
|---|---|
| Single part, 80 views | 5-10 seconds |
| Multi-part scene, 2 parts | 15-20 seconds |
| With post-processing | +2-3 seconds |

### PyTorch Training (Full LEGO Dataset)

| Setup | Training Time |
|---|---|
| DGX Spark GPU | 2-4 hours |
| MacBook Pro M4 | 12-18 hours |
| Laptop CPU | 24-48 hours |

## Next Steps

1. **Immediate**: Test vision API works end-to-end with sample piece images
2. **Short-term**: Generate synthetic training data for high-volume parts
3. **Medium-term**: Train on combined real + synthetic data
4. **Long-term**: Evaluate model accuracy and iterate on data collection

## References

- [Ollama Documentation](https://ollama.ai)
- [LLaVA Vision Model](https://github.com/haotian-liu/LLaVA)
- [NVIDIA JetPack](https://developer.nvidia.com/jetpack)
- [Blender Python API](https://docs.blender.org/api/current/)
- [PyTorch on Jetson](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)

## Support

For issues with:
- **Vision model**: Check Ollama logs: `journalctl -u ollama -f`
- **Blender rendering**: Check temp directory: `ls /tmp/brickscan-*`
- **PyTorch training**: Review training log: `tail -f training_log_*.txt`
- **Network connectivity**: Run `bash network/find_dgx.sh`
