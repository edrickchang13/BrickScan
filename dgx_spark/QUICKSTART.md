# BrickScan DGX Spark - Quick Start Guide

Complete setup path from DGX Spark box to local vision model in 30 minutes.

## Step 1: Initial DGX Spark Setup (15 min)

On the DGX Spark, clone or copy project files:

```bash
# SSH into DGX Spark
ssh ubuntu@<dgx-spark-ip>

# Clone your repo or copy files
cd ~
git clone <your-repo> brickscan
cd brickscan/dgx_spark

# Run setup
bash setup/install_dependencies.sh
# This installs: Ollama, LLaVA models, FastAPI, Blender
# Expected: 10-15 minutes (downloading models)
```

## Step 2: Start Vision Server (2 min)

On DGX Spark:

```bash
cd ~/brickscan/dgx_spark/vision_server
bash start_server.sh
# Should see: "INFO: Uvicorn running on http://0.0.0.0:8001"
```

## Step 3: Find DGX Spark IP (2 min)

On your Mac:

```bash
bash dgx_spark/network/find_dgx.sh
# Returns: Found BrickScan Vision Server at: 192.168.1.100:8001
```

## Step 4: Configure Backend (5 min)

In your backend `.env`:

```bash
DGX_VISION_URL=http://192.168.1.100:8001
VISION_BACKEND=dgx
```

## Step 5: Test It Works (5 min)

From Mac:

```bash
# Health check
curl http://192.168.1.100:8001/health

# Test with real image
python3 network/test_connection.py --url http://192.168.1.100:8001 --image lego_brick.jpg
```

Done! Your backend now uses local LEGO identification.

---

## Advanced: Train Models on DGX (2-4 hours)

After you have training data:

```bash
# On DGX Spark
bash training/setup_ml_env.sh   # One-time setup
bash training/train_on_dgx.sh   # Start training
```

Expected: 2-4 hours for 3000 LEGO classes (vs 12+ hours on laptop)

## Advanced: Generate Synthetic Data (30-40 hours)

For unlimited training examples:

```bash
# On DGX Spark
bash rendering/blender_dgx_render.sh
# Renders 80 images per LEGO part @ 10s/image on GPU
```

---

## Troubleshooting

**Vision server won't start**
```bash
# Check Ollama is running
sudo systemctl status ollama

# Check models downloaded
ollama list
# Should show: llava:13b, moondream

# Pull if missing
ollama pull llava:13b
```

**Cannot find DGX on network**
```bash
# From Mac: check DGX is on same WiFi
ping 192.168.1.1  # Ping router
nmap 192.168.1.0/24 | grep "report"  # Scan network

# On DGX: check IP
hostname -I
```

**Inference is slow (>5 seconds)**
```bash
# Check GPU is being used
nvidia-smi  # On DGX
# If memory not increasing: Ollama using CPU

# Restart Ollama
sudo systemctl restart ollama
```

---

## File Reference

```
dgx_spark/
├── README.md                          # Full documentation
├── setup/
│   └── install_dependencies.sh        # Initial setup script
├── vision_server/
│   ├── server.py                      # FastAPI inference server
│   ├── start_server.sh                # Run server manually
│   ├── systemd_service.sh             # Auto-start on boot
│   └── requirements.txt               # Python dependencies
├── backend_integration/
│   └── dgx_vision_service.py          # Drop-in replacement for backend
├── training/
│   ├── setup_ml_env.sh                # PyTorch environment setup
│   └── train_on_dgx.sh                # Start training
├── rendering/
│   └── blender_dgx_render.sh          # Synthetic data generation
└── network/
    ├── find_dgx.sh                    # Discover DGX on network
    └── test_connection.py             # Benchmark vision server
```

---

## Performance

| Task | Time | Notes |
|------|------|-------|
| Single LEGO identification | 1.2-1.8s | Cold start 3-5s (model load) |
| 10 pieces (parallel) | 2.5-3.5s | Much faster than cloud API |
| Train 3000-class model | 2-4 hours | DGX GPU vs 12h+ on laptop |
| Synthetic render (1 part) | 5-10s | 80 images per part |

---

## Next Steps

1. **Verify**: `curl http://<dgx-ip>:8001/health` returns healthy
2. **Deploy**: Update backend `.env` with `DGX_VISION_URL`
3. **Monitor**: Check inference times with `test_connection.py`
4. **Scale**: Generate synthetic data or train new models

For full documentation, see `README.md`.
