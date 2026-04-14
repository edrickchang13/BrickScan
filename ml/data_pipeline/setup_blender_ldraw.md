# BrickScan LDraw Blender Setup Guide

Complete step-by-step instructions for setting up Blender with LDraw for synthetic LEGO piece rendering.

## Overview

This guide covers:
1. Installing Blender 4.x
2. Downloading the LDraw parts library (~17,000 pieces)
3. Installing and configuring the LDraw importer addon
4. Testing with a simple part
5. Running the full rendering pipeline
6. Using GPU acceleration on DGX Spark
7. Troubleshooting common issues

---

## 1. Install Blender 4.x

### macOS

```bash
# Download from official Blender website
# https://www.blender.org/download/release/Blender4.0/

# Or using Homebrew
brew install blender

# Verify installation
blender --version
# Output: Blender 4.0.0 (hash)
```

### Linux (Ubuntu/Debian)

```bash
# Add Blender repository and install
sudo apt-get update
sudo apt-get install blender

# Or download from official website and extract
cd ~/tools
wget https://www.blender.org/download/release/Blender4.0/blender-4.0.0-linux-x64.tar.xz
tar -xf blender-4.0.0-linux-x64.tar.xz
export PATH=~/tools/blender-4.0.0-linux-x64:$PATH

# Verify
blender --version
```

### DGX Spark Cluster (preferred for GPU rendering)

```bash
# SSH into DGX node
ssh user@spark-node-1.company.com

# Load NVIDIA modules
module load cuda/12.0
module load cudnn/12.0

# Install or use pre-installed Blender
which blender
# If not available: /usr/local/blender-4.0.0/bin/blender

# Verify CUDA support
blender --version
# Should show NVIDIA GPU support enabled
```

---

## 2. Download LDraw Parts Library

The complete LDraw library contains ~17,000 official LEGO piece definitions.

### Official Download

```bash
# Create directory for LDraw
mkdir -p ~/ldraw
cd ~/ldraw

# Download complete library (130MB)
# Option 1: Using wget
wget http://www.ldraw.org/library/update/complete.zip

# Option 2: Using curl
curl -O http://www.ldraw.org/library/update/complete.zip

# Extract
unzip complete.zip

# Directory structure after extraction
ls -la ~/ldraw/
# parts/       - All .dat files for individual parts
# p/           - Primitive parts (used by other parts)
# models/      - Example LEGO models
# ldraw.ldr    - Index file

# Verify parts directory
ls ~/ldraw/parts | head -20
# Output: 1-10cyli.dat, 1-16chrt.dat, 2-4chrt.dat, 3001.dat, ...

# Part file naming convention:
# - 3001.dat = 2x4 brick (most common part)
# - 3626.dat = 2x2 plate
# - 32054.dat = Slope brick
```

### Alternative: Use Pre-Downloaded Library

If downloading during setup is slow, you can download on a faster network:

```bash
# On your local machine (fast connection)
cd ~/Downloads
curl -O http://www.ldraw.org/library/update/complete.zip

# Transfer to server/DGX
scp complete.zip user@spark-node.com:/home/user/ldraw/
ssh user@spark-node.com "cd ldraw && unzip complete.zip"
```

---

## 3. Install LDraw Importer Addon

The LDraw importer addon allows Blender to directly read .dat files.

### Download ImportLDraw Addon

```bash
# GitHub repository
# https://github.com/LDraw-Loader-for-Blender/ldraw-importer

# Clone or download
cd ~/tools
git clone https://github.com/LDraw-Loader-for-Blender/ldraw-importer.git

# The addon directory structure:
ls ldraw-importer/
# __init__.py
# io_scene_importldraw.py
# ldraw_loader.py
# ... other files
```

### Install Addon in Blender

#### Method 1: Copy to Blender Directory

```bash
# Find Blender config directory
BLENDER_CONFIG="$HOME/.config/blender/4.0/scripts/addons"  # Linux/Mac
# OR
BLENDER_CONFIG="$HOME/AppData/Roaming/Blender/4.0/scripts/addons"  # Windows

mkdir -p "$BLENDER_CONFIG"

# Copy addon files
cp -r ~/tools/ldraw-importer "$BLENDER_CONFIG/io_scene_importldraw"

# Verify
ls "$BLENDER_CONFIG/io_scene_importldraw"
```

#### Method 2: Install via Blender GUI (easier)

1. Open Blender
2. Go to Edit > Preferences
3. Click "Add-ons" in left sidebar
4. Click "Install..." at top
5. Navigate to `ldraw-importer/__init__.py`
6. Click "Install Add-on"
7. Search for "LDraw" and enable it
8. In addon settings, set "LDraw Library Path" to `/path/to/ldraw`

---

## 4. Configure LDraw Addon Settings

### Via Blender GUI

1. Edit > Preferences > Add-ons > "LDraw Importer"
2. In addon settings, configure:

```
LDraw Library Path: /home/user/ldraw
(or ~/ldraw if using environment variable expansion)

Import Type: Load as single object
Smooth shading: Enabled
UV mapping: Automatic
```

### Via Python Script (headless)

For automation/DGX, configure programmatically:

```python
import bpy

# Enable addon
bpy.ops.preferences.addon_enable(module="io_scene_importldraw")

# Get addon preferences
addon = bpy.context.preferences.addons.get("io_scene_importldraw")
if addon:
    prefs = addon.preferences
    prefs.ldrawPath = "/path/to/ldraw"
    print(f"LDraw path set to: {prefs.ldrawPath}")
```

### Verify Installation

```bash
# Test in Blender (interactive mode)
blender -b --python <<'EOF'
import bpy

# Try to access addon
addon = bpy.context.preferences.addons.get('io_scene_importldraw')
if addon:
    print("LDraw addon is installed and available")
    print(f"Addon path: {addon.module}")
else:
    print("ERROR: LDraw addon not found!")
    print("Installed addons:", [a.module for a in bpy.context.preferences.addons])
EOF
```

---

## 5. Test with a Simple Part

### Manual Test: Load Part in Blender GUI

```bash
# Download a simple test part if not in LDraw library yet
curl -O http://www.ldraw.org/parts/3001.dat  # 2x4 brick

# Open Blender
blender

# In Blender:
# 1. File > Import > LDraw (.dat)
# 2. Navigate to 3001.dat
# 3. Click Import LDraw
# 4. Part appears in viewport
```

### Automated Test: Python Script

Create `test_ldraw_import.py`:

```python
#!/usr/bin/env python3
"""Test LDraw part importing in Blender"""

import bpy
import sys
from pathlib import Path

# Part to test
PART_FILE = "/home/user/ldraw/parts/3001.dat"  # 2x4 brick
LDRAW_PATH = "/home/user/ldraw"

def test_import():
    """Test importing an LDraw part"""
    
    # Check addon
    addon = bpy.context.preferences.addons.get('io_scene_importldraw')
    if not addon:
        print("ERROR: io_scene_importldraw addon not installed")
        return False
    
    # Try import
    try:
        bpy.ops.import_scene.importldraw(
            filepath=PART_FILE,
            ldrawPath=LDRAW_PATH
        )
        print(f"SUCCESS: Imported {Path(PART_FILE).stem}")
        
        # Check what was imported
        imported = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
        print(f"  Objects created: {len(imported)}")
        if imported:
            obj = imported[0]
            print(f"  Main object: {obj.name}")
            print(f"  Vertices: {len(obj.data.vertices)}")
            print(f"  Faces: {len(obj.data.polygons)}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Import failed: {e}")
        return False

if __name__ == "__main__":
    # Run test
    if test_import():
        print("\nTest PASSED - LDraw import is working!")
        sys.exit(0)
    else:
        print("\nTest FAILED - Check LDraw addon installation")
        sys.exit(1)
```

Run test:

```bash
# Headless Blender with test script
blender --background --python test_ldraw_import.py

# Output:
# SUCCESS: Imported 3001
#   Objects created: 1
#   Main object: Brick_2x4
#   Vertices: 1234
#   Faces: 567
#
# Test PASSED - LDraw import is working!
```

---

## 6. Run the Full Rendering Pipeline

### Single Part Example

```bash
# Render one specific part 80 times with all colors
blender --background --python ldraw_renderer.py -- \
    --part_file ~/ldraw/parts/3001.dat \
    --output_dir ~/brickscan_output \
    --num_renders 80 \
    --colors all \
    --resolution 512 \
    --engine EEVEE \
    --gpu

# Output
# [2024-01-15 14:32:00] BrickScan LDraw Renderer Starting
# [2024-01-15 14:32:01] Setting up EEVEE render engine...
# [2024-01-15 14:32:02] Rendering part 3001: 80 images
# [2024-01-15 14:32:05]   3001: 10/80 complete
# ...
# [2024-01-15 14:34:30] RENDER COMPLETE: 80 images from 1 parts
```

### Batch Render: Top 100 Parts

```bash
# First, get top parts from Rebrickable data
python get_top_parts.py \
    --inventory_parts ~/rebrickable_data/inventory_parts.csv \
    --inventories ~/rebrickable_data/inventories.csv \
    --output top_100_parts.txt \
    --limit 100

# Then render all 100 parts (60 images each = 6,000 total)
blender --background --python ldraw_renderer.py -- \
    --parts_dir ~/ldraw/parts \
    --top_parts_file top_100_parts.txt \
    --output_dir ~/brickscan_output \
    --num_renders 60 \
    --colors common \
    --resolution 512 \
    --engine CYCLES \
    --gpu

# Estimate: 6,000 images x 5 seconds per render = 30,000 seconds = 8.3 hours
# With GPU (DGX): ~1-2 hours
```

### Full Dataset: 3,000 Parts

For production dataset (200,000+ images):

```bash
# Get top 3,000 most common LEGO parts
python get_top_parts.py \
    --inventory_parts ~/rebrickable_data/inventory_parts.csv \
    --inventories ~/rebrickable_data/inventories.csv \
    --output top_3000_parts.txt \
    --limit 3000 \
    --stats

# Output:
# Top 20 most common parts:
# Rank  Part #       Sets
# 1     3001         2145
# 2     3002         1987
# 3     3010         1876
# ...
# Total unique parts: 14,253
# Top 3,000 parts cover: 87.3% of all instances

# Render with minimal settings for speed
blender --background --python ldraw_renderer.py -- \
    --parts_dir ~/ldraw/parts \
    --top_parts_file top_3000_parts.txt \
    --output_dir ~/brickscan_output/full_dataset \
    --num_renders 70 \
    --colors common \
    --resolution 512 \
    --engine EEVEE \
    --gpu

# Estimate: 3,000 parts x 70 renders x 2 seconds (EEVEE) = 420,000 seconds = 117 hours
# On DGX with 8 GPUs: ~15 hours
```

---

## 7. GPU Acceleration on DGX Spark

The DGX Spark cluster provides 8x NVIDIA A100 GPUs, enabling massive parallelization.

### DGX Setup

```bash
# SSH to DGX node
ssh user@spark-node-1.company.com

# Load NVIDIA modules
module load cuda/12.0
module load cudnn/12.0
module load tensorrt/8.5

# Verify NVIDIA environment
nvidia-smi
# Output:
# +----------+------+
# | GPU  Name| Mem  |
# +----------+------+
# | 0 NVIDIA A100| 40GB |
# | 1 NVIDIA A100| 40GB |
# ...
# | 7 NVIDIA A100| 40GB |
```

### Install Blender on DGX

```bash
# Check if Blender is available
which blender

# If not installed, get pre-built GPU-enabled version
cd /opt/software
wget https://www.blender.org/download/release/Blender4.0/blender-4.0.0-linux-x64.tar.xz
tar -xf blender-4.0.0-linux-x64.tar.xz
export PATH=/opt/software/blender-4.0.0-linux-x64:$PATH

# Verify CUDA support
blender --version | grep -i cuda
```

### Parallel Rendering Strategy

**Option 1: Distributed Part Rendering (Recommended)**

Divide 3,000 parts across 8 GPUs:

```bash
#!/bin/bash
# distribute_renders.sh

PARTS_FILE="top_3000_parts.txt"
NUM_GPUS=8

# Split parts list
split -n l/$NUM_GPUS $PARTS_FILE parts_batch_

# Launch one render job per GPU
for i in $(seq 0 $((NUM_GPUS-1))); do
    GPU_ID=$i
    BATCH_FILE="parts_batch_$(printf '%02d' $i)"
    
    # Create batches file for this GPU
    cat "$BATCH_FILE" > "/tmp/batch_$GPU_ID.txt"
    
    # Launch in background
    (
        export CUDA_VISIBLE_DEVICES=$GPU_ID
        blender --background --python ldraw_renderer.py -- \
            --parts_dir ~/ldraw/parts \
            --top_parts_file "/tmp/batch_$GPU_ID.txt" \
            --output_dir ~/output/batch_$GPU_ID \
            --num_renders 70 \
            --colors common \
            --resolution 512 \
            --engine CYCLES \
            --gpu
    ) &
done

# Wait for all jobs to complete
wait
echo "All rendering jobs complete!"

# Merge results
mkdir -p ~/output/merged
for dir in ~/output/batch_*/; do
    cp -r "$dir"/* ~/output/merged/
done
```

Run it:

```bash
chmod +x distribute_renders.sh
./distribute_renders.sh

# Monitor progress
watch -n 5 'ls -la ~/output/merged | wc -l'
```

**Option 2: Job Submission (SLURM)**

On HPC systems with SLURM:

```bash
# sbatch_render.sh
#!/bin/bash
#SBATCH --job-name=brickscan-render
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --time=24:00:00
#SBATCH --mem=256GB

module load cuda/12.0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Render with all GPUs
blender --background --python ldraw_renderer.py -- \
    --parts_dir ~/ldraw/parts \
    --top_parts_file top_3000_parts.txt \
    --output_dir ~/output \
    --num_renders 70 \
    --colors common \
    --resolution 512 \
    --engine CYCLES \
    --gpu
```

Submit:

```bash
sbatch sbatch_render.sh
squeue -u $USER  # Monitor
```

---

## 8. Output Structure

After rendering, your output directory looks like:

```
~/brickscan_output/
├── 3001/           # Brick (2x4)
│   ├── 00000.png
│   ├── 00001.png
│   ├── ...
│   └── 00069.png   # 70 renders
├── 3002/           # Brick (2x2)
│   ├── 00000.png
│   ├── ...
│   └── 00069.png
├── 3010/           # Plate (1x4)
│   └── ...
├── ...
├── render_metadata.json  # Metadata for all renders
└── top_3000_parts.txt    # List of parts rendered
```

### Metadata Format

`render_metadata.json`:

```json
{
  "timestamp": "2024-01-15T14:30:00.123456",
  "parts_rendered": 3000,
  "total_images": 210000,
  "renders": [
    {
      "part_num": "3001",
      "color": "White",
      "background": "white",
      "angle_index": 0,
      "filename": "00000.png"
    },
    {
      "part_num": "3001",
      "color": "White",
      "background": "light_gray",
      "angle_index": 1,
      "filename": "00001.png"
    },
    ...
  ]
}
```

This metadata is essential for training - it labels each image with:
- **part_num**: LEGO piece identifier
- **color**: Which of 25+ LEGO colors
- **background**: Photography context
- **angle_index**: Camera viewing angle

---

## 9. Troubleshooting

### Issue: "LDraw addon not found"

```bash
# Solution 1: Verify addon is installed
ls ~/.config/blender/4.0/scripts/addons/io_scene_importldraw/

# Solution 2: Reinstall addon
# Download fresh: https://github.com/LDraw-Loader-for-Blender/ldraw-importer
# Copy to addons directory
# Restart Blender

# Solution 3: Check addon preferences
blender --python - <<'EOF'
import bpy
addons = [a.module for a in bpy.context.preferences.addons]
print("Installed addons:", addons)
print("LDraw installed:", "io_scene_importldraw" in addons)
EOF
```

### Issue: "LDraw library path not found"

```bash
# Make sure LDraw is downloaded
ls ~/ldraw/parts | head

# If empty, download:
cd ~/ldraw
wget http://www.ldraw.org/library/update/complete.zip
unzip complete.zip

# Update addon settings with correct path
# Or set environment variable:
export LDRAW_PATH=~/ldraw
```

### Issue: "Out of VRAM" on GPU

Reduce render settings:

```bash
# Smaller resolution
--resolution 384  # Instead of 512

# Fewer samples
--samples 32  # For EEVEE (default 64)

# Less demanding engine
--engine EEVEE  # Instead of CYCLES
```

### Issue: "Part file not found" errors

```bash
# Verify part files exist
find ~/ldraw/parts -name "3001.dat" -o -name "3002.dat"

# If missing, check LDraw download:
ls -la ~/ldraw/parts | head

# Make sure top_parts.txt contains valid part numbers
head top_3000_parts.txt
# Output should be: 3001\t2145
# Not: /path/to/ldraw/parts/3001.dat
```

### Issue: Very slow rendering (not using GPU)

Check GPU is available:

```bash
# Verify CUDA is working
blender --python - <<'EOF'
import bpy
scene = bpy.context.scene
scene.render.engine = 'CYCLES'

# Check device type
prefs = bpy.context.preferences.addons['cycles'].preferences
print(f"Compute device type: {prefs.compute_device_type}")

# List GPUs
devices = prefs.devices
for device in devices:
    print(f"Device: {device.name}, Type: {device.type}, Use: {device.use}")
EOF

# If GPU not showing, install CUDA drivers:
# https://developer.nvidia.com/cuda-downloads
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
```

---

## 10. Next Steps: Training

Once you have your synthetic dataset:

```bash
# Count generated images
find ~/output -name "*.png" | wc -l
# Output: 210000 (for 3000 parts x 70 renders)

# Organize for training
python organize_for_training.py \
    --input_dir ~/output \
    --output_dir ~/datasets/brickscan_training \
    --train_split 0.8 \
    --val_split 0.1 \
    --test_split 0.1

# Train YOLO/RetinaNet detector with augmentation
python train_detector.py \
    --dataset ~/datasets/brickscan_training \
    --model yolov8m \
    --epochs 100 \
    --batch_size 32 \
    --gpus 0,1,2,3
```

---

## References

- **Blender**: https://www.blender.org/download/
- **LDraw Library**: http://www.ldraw.org/
- **LDraw Importer**: https://github.com/LDraw-Loader-for-Blender/ldraw-importer
- **Rebrickable Data**: https://rebrickable.com/downloads/
- **NVIDIA CUDA**: https://developer.nvidia.com/cuda-downloads
- **DGX Spark Docs**: Internal documentation in company wiki

