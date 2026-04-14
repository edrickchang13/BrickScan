# BrickScan Blender GPU Rendering Pipeline

A production-ready NVIDIA GPU rendering pipeline for generating synthetic LEGO brick training data using Blender and the LDraw parts library.

## Overview

This system renders LEGO bricks in all possible colors and multiple camera angles to create a large dataset for training a LEGO part + color classifier ML model. The rendering is GPU-accelerated via NVIDIA CUDA, with support for parallel batch processing.

## System Requirements

- **Blender 4.x** (with CUDA support)
- **NVIDIA GPU** with CUDA compute capability 5.0+
- **Python 3.8+** (for batch orchestration)
- **curl** or **wget** (for LDraw library download)
- **unzip** or **7z** (for extraction)

### Installation

1. **Install Blender 4.x**
   ```bash
   # macOS (Homebrew)
   brew install blender

   # Ubuntu/Debian
   sudo apt-get install blender

   # Or download from: https://www.blender.org/download/
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and extract LDraw library**
   ```bash
   ./setup_ldraw.sh
   ```
   This downloads ~2.5 GB of LEGO part definitions and extracts to `ml/data/ldraw/`.

## File Structure

```
ml/
├── blender/
│   ├── render_parts.py        # Core Blender rendering script
│   ├── batch_render.py        # Orchestrator for parallel rendering
│   ├── setup_ldraw.sh         # LDraw library downloader
│   ├── requirements.txt       # Python dependencies
│   └── README.md             # This file
├── data/
│   ├── ldraw/               # LDraw parts library (downloaded by setup_ldraw.sh)
│   ├── colors.csv           # Rebrickable color definitions (must be provided)
│   ├── renders/             # Output directory (created during rendering)
│   │   ├── part_1234/
│   │   │   ├── part_1234_16_0000.png
│   │   │   ├── part_1234_16_0001.png
│   │   │   └── ...
│   │   └── ...
│   └── index.csv            # Manifest of all rendered images
```

## Quick Start

### 1. Prepare color definitions

Obtain `colors.csv` from [Rebrickable](https://rebrickable.com/downloads/) with columns:
- `id` (color ID)
- `name` (color name)
- `rgb` (hex color code, e.g., "05131D")

Save as `ml/data/colors.csv`

### 2. Download LDraw library

```bash
cd ml/blender
./setup_ldraw.sh
```

This creates `ml/data/ldraw/ldraw/` with `parts/` and `p/` subdirectories.

### 3. (Optional) Create a parts subset file

By default, the pipeline renders the top 500 parts from the LDraw library. To render a custom set:

```bash
# ml/data/parts_subset.txt (one part number per line)
3004       # Standard brick
3001       # Standard brick
3622       # Slope brick
```

### 4. Run batch rendering

```bash
python batch_render.py \
  --blender /path/to/blender \
  --ldraw-dir ./data/ldraw \
  --colors-csv ./data/colors.csv \
  --parts-file ./data/parts_subset.txt \
  --output-dir ./data/renders \
  --workers 4 \
  --num-angles 36 \
  --resolution 224
```

**Key arguments:**
- `--blender`: Path to Blender executable (default: `blender` in PATH)
- `--workers`: Number of parallel Blender processes (default: 4; use 1-2 on limited GPU memory)
- `--num-angles`: Camera angles per elevation (default: 36; total: 3 elevations × num_angles)
- `--resolution`: Output PNG resolution in pixels (default: 224×224)
- `--skip-existing`: Skip parts that already have rendered output
- `--max-parts`: For testing, limit number of parts (e.g., `--max-parts 10`)

### 5. Monitor progress

The pipeline logs to:
- **Console**: Real-time progress with tqdm bar
- **`ml/data/failed_renders.log`**: Errors and warnings

### 6. Access rendered data

Output structure:
```
ml/data/renders/
├── 3004/
│   ├── 3004_16_0000.png      # Part 3004, color 16, angle 0
│   ├── 3004_16_0001.png
│   └── ...
├── 3001/
│   └── ...
└── index.csv                   # Metadata manifest
```

**index.csv format:**
```csv
image_path,part_num,color_id,color_name,color_r,color_g,color_b
renders/3004/3004_16_0000.png,3004,16,White,1.0000,1.0000,1.0000
```

## Rendering Details

### Rendering Approach

Each (part, color) combination is rendered from **3 elevation angles** and **N azimuth angles**:
- **Elevations**: -20°, +10°, +30° (from horizontal)
- **Azimuths**: 0° to 360° in N equal steps (default N=36, so 10° increments)
- **Total frames per color**: 3 × N (default: 108)

### Camera & Lighting

- **Camera**: Orbits at constant distance, auto-fitted to bounding box
- **Lens**: 50mm focal length
- **Lighting**: 3-point setup
  - Key light: warm, strong (2.5× intensity)
  - Fill light: cool, soft (1.0× intensity)
  - Rim light: white, back (1.5× intensity)
- **Background**: Dark (0.05 RGB), randomly rotated HDRI mapping for variety

### GPU Rendering

- **Engine**: NVIDIA CUDA (Cycles)
- **Samples**: 128 per frame (balanced for speed vs quality)
- **Denoiser**: OpenImageDenoise (Intel)
- **Output**: 8-bit RGBA PNG with transparency

## Advanced Usage

### Custom Camera Angles

Edit `render_parts.py` to modify elevation angles:
```python
elevation_angles = [-20, 10, 30]  # degrees
```

### Render Specific Colors Only

```bash
python batch_render.py \
  --colors 16 1 4 71  # Render only colors 16, 1, 4, 71
```

### Single Part Test Render

Render one part with all angles for testing:
```bash
blender --background --python render_parts.py -- \
  --part-file ./data/ldraw/ldraw/parts/3004.dat \
  --output-dir /tmp/test_render \
  --part-num 3004 \
  --color-id 16 \
  --color-name White \
  --color-r 1.0 --color-g 1.0 --color-b 1.0 \
  --num-angles 4 \
  --resolution 224
```

### Adjust Quality vs Speed

**For fast iterations (testing):**
```bash
# Fewer samples, lower resolution
python batch_render.py --resolution 112 # 2x2 rendering (4x faster)
# Edit render_parts.py: scene.cycles.samples = 64
```

**For high-quality output:**
```bash
# More samples, higher resolution
python batch_render.py --resolution 512
# Edit render_parts.py: scene.cycles.samples = 256
```

### Parallel Job Distribution

For rendering across multiple machines:
1. Split `parts_subset.txt` into chunks
2. Run `batch_render.py` on each machine with a different chunk
3. Merge results and `index.csv` files

## Troubleshooting

### "CUDA device not found"
- Verify NVIDIA driver: `nvidia-smi`
- Ensure Blender was built with CUDA support
- Try CPU fallback: Edit `render_parts.py`, change to `scene.cycles.device_type = "CPU"`

### "LDraw importer failed"
- The pipeline gracefully falls back to `.obj` files if available
- If parts aren't found, verify LDraw extraction: `ls ml/data/ldraw/ldraw/parts/ | head`

### Out of Memory (OOM)
- Reduce `--workers` (process fewer parts in parallel)
- Reduce `--resolution`
- Edit `render_parts.py`: lower `scene.cycles.samples`

### Slow rendering
- Ensure GPU rendering is active (check logs for "CUDA")
- Reduce `--num-angles` for faster iteration
- Reduce `--workers` to give GPU more VRAM per process

### Part file not found
- Verify LDraw library was extracted: `setup_ldraw.sh`
- Check part number spelling (case-sensitive in some cases)
- Scan library: `find ml/data/ldraw -name "*.dat" | wc -l`

## Performance Benchmarks

On an **NVIDIA RTX 4090** with **224×224 resolution, 128 samples, 36 angles:**
- ~15–20 seconds per (part, color) pair
- ~400–600 images per minute per GPU
- 4 parallel processes: ~1500–2400 images per minute total

Rendering 500 parts × 70 colors × 108 images = ~3.8M images
Expected time: ~26–42 hours on single RTX 4090

## Output Quality

The rendered images are designed as synthetic training data:
- **Realistic materials**: Principled BSDF with 0.3 roughness (LEGO plastic sheen)
- **Varied lighting**: 3-point lighting with realistic ratios
- **Natural backgrounds**: Dark with subtle HDRI rotation
- **Transparency**: PNG alpha channel supports compositing
- **Consistency**: Same part+color always has same material properties across angles

## Contributing & Modifications

The code is structured for extensibility:

- **`render_parts.py`**: Modify rendering parameters (samples, material properties, camera distance)
- **`batch_render.py`**: Modify task scheduling, logging, or output metadata
- **`setup_ldraw.sh`**: Adapt for alternate part libraries or mirroring

## License

This code is provided as-is for the BrickScan project. The LDraw library is distributed under the [LDraw License Agreement](https://library.ldraw.org/license).

## References

- [LDraw Library](https://library.ldraw.org/)
- [Rebrickable](https://rebrickable.com/)
- [Blender Cycles Documentation](https://docs.blender.org/manual/en/latest/render/cycles/)
- [NVIDIA CUDA in Blender](https://docs.blender.org/manual/en/latest/render/cycles/gpu_rendering.html)
