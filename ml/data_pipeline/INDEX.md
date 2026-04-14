# BrickScan Data Pipeline - Complete Reference

## Files Overview

### 1. **ldraw_renderer.py** (742 lines, 25 KB)
**Main Blender rendering engine for synthetic LEGO image generation**

Key components:
- `LEGO_COLORS`: 25+ official LEGO ABS plastic colors with accurate RGB values
- `parse_args()`: Command-line argument parser for headless Blender execution
- `setup_render_engine()`: Configure EEVEE or CYCLES rendering with GPU support
- `create_camera()`: Create perspective camera with 50mm focal length
- `position_camera_orbit()`: Random orbital positioning around LEGO piece
- `setup_3point_lighting()`: Randomized key/fill/rim lighting setup
- `setup_background()`: 5 background types (white, gray, beige, dark, gradient)
- `import_ldraw_part()`: LDraw DAT file importer (addon-based + fallback parser)
- `apply_lego_material()`: Physically-accurate ABS plastic material with PBR
- `center_object()`: Auto-center piece at origin and ground plane
- `randomize_rotation()`: Random 3-axis rotation for angle variation
- `render_part()`: Main render loop with color/angle/background variations
- `save_metadata()`: JSON metadata output for auto-labeling

Features:
- EEVEE: Fast rendering (2-3 sec/image), suitable for large datasets
- CYCLES: Photorealistic rendering (5-10 sec/image), optional
- GPU acceleration: NVIDIA CUDA support
- 3-point lighting: Randomized position, intensity, color
- 5 backgrounds: Simulates different photography contexts
- 25+ colors: Full LEGO official palette including transparents
- DAT parsing: Manual LDraw file parser (fallback if addon unavailable)
- Metadata: JSON output with part_num, color, background, angle index

Usage example:
```bash
blender --background --python ldraw_renderer.py -- \
    --part_file ~/ldraw/parts/3001.dat \
    --output_dir ~/output \
    --num_renders 70 \
    --colors common \
    --resolution 512 \
    --engine EEVEE \
    --gpu
```

Output:
- PNG files: `output/{part_num}/{index:05d}.png`
- Metadata: `output/render_metadata.json`

---

### 2. **get_top_parts.py** (313 lines, 10 KB)
**Extract top LEGO parts by frequency for dataset generation**

Queries Rebrickable database (https://rebrickable.com/downloads/) to identify
the most commonly-used LEGO pieces across all sets.

Key components:
- `build_frequency_table_from_csv()`: Parse Rebrickable CSV files
  - Loads inventory_parts.csv (part listings)
  - Loads inventories.csv (set mappings)
  - Counts unique sets containing each part (not total quantity)
  - Filters out spare parts
- `get_top_parts_from_csv()`: Extract top N parts and save to file
- `generate_statistics()`: Distribution analysis
- Command-line interface with examples

Features:
- CSV parsing: Direct analysis of Rebrickable database
- Frequency counting: Parts ranked by set appearance
- Filtering: Excludes spare parts automatically
- Statistics: Coverage analysis (which top N parts cover X% of usage)
- Output format: Tab-separated (part_num, frequency)

Input files (download from Rebrickable):
- `inventory_parts.csv`: Columns (inventory_id, part_num, color_id, quantity, is_spare)
- `inventories.csv`: Columns (id, set_num, version)

Output:
```
3001    2145    (appears in 2,145 sets)
3002    1987
3010    1876
...
```

Usage example:
```bash
python get_top_parts.py \
    --inventory_parts ./rebrickable_data/inventory_parts.csv \
    --inventories ./rebrickable_data/inventories.csv \
    --output top_3000_parts.txt \
    --limit 3000 \
    --stats
```

Statistics output:
- Total unique parts in database
- Distribution metrics
- Coverage levels (e.g., "X parts needed for 50% coverage")

---

### 3. **setup_blender_ldraw.md** (748 lines, 17 KB)
**Complete setup and troubleshooting guide**

Comprehensive documentation covering:

1. **Installation** (macOS, Linux, DGX Spark)
   - Blender 4.x download and setup
   - Verification steps

2. **LDraw Library** (~17,000 parts, 130 MB)
   - Official download from http://www.ldraw.org/
   - Directory structure explanation
   - File naming convention (3001.dat = 2x4 brick, etc.)

3. **LDraw Importer Addon**
   - GitHub source: https://github.com/LDraw-Loader-for-Blender/ldraw-importer
   - Installation methods (copy vs. GUI)
   - Configuration (set library path)
   - Verification testing

4. **Testing**
   - Manual GUI test (File > Import > LDraw)
   - Automated Python test script
   - Verification output examples

5. **Running the Pipeline**
   - Single part rendering
   - Batch rendering (top 100 parts)
   - Full dataset (3,000 parts)
   - Time/resource estimates

6. **GPU Acceleration (DGX)**
   - NVIDIA module loading
   - CUDA verification
   - Parallel rendering strategies:
     - Distributed across 8 GPUs
     - SLURM job submission examples
     - Performance metrics (160-240 img/min total)

7. **Output Structure**
   - Directory organization
   - Metadata JSON format
   - Example usage in ML training

8. **Troubleshooting**
   - LDraw addon not found
   - Library path errors
   - VRAM issues (reduced settings)
   - GPU not detected (CUDA setup)
   - Missing part files

9. **Next Steps**
   - Training workflow overview
   - PyTorch dataset loading
   - Augmentation strategies

---

### 4. **README_PIPELINE.txt** (332 lines, 10 KB)
**Quick reference guide for the entire pipeline**

High-level overview including:
- Pipeline flow diagram
- File descriptions and command examples
- Rendering specifications (colors, backgrounds, lighting, camera, materials)
- Data volume estimation
- Example workflows (test → scale → production)
- Output structure
- PyTorch integration example
- Performance metrics
- Hardware requirements (minimum, recommended, production)

---

## Quick Start

### Step 1: Setup (first time only)
```bash
# Follow setup_blender_ldraw.md
# - Install Blender 4.x
# - Download LDraw library (130 MB)
# - Install LDraw importer addon
# - Test import
```

### Step 2: Identify Top Parts
```bash
python get_top_parts.py \
    --inventory_parts ~/rebrickable_data/inventory_parts.csv \
    --inventories ~/rebrickable_data/inventories.csv \
    --output top_3000_parts.txt \
    --limit 3000
```

### Step 3: Render Dataset
```bash
blender --background --python ldraw_renderer.py -- \
    --parts_dir ~/ldraw/parts \
    --top_parts_file top_3000_parts.txt \
    --output_dir ~/brickscan_dataset \
    --num_renders 70 \
    --colors common \
    --resolution 512 \
    --engine EEVEE \
    --gpu
```

### Step 4: Use in ML Training
```python
import json
from pathlib import Path
from PIL import Image

# Load metadata
with open("brickscan_dataset/render_metadata.json") as f:
    metadata = json.load(f)

# Create dataset
images = []
labels = []
for render in metadata["renders"]:
    path = f"brickscan_dataset/{render['part_num']}/{render['filename']}"
    images.append(Image.open(path))
    labels.append(render['part_num'])
```

---

## Key Concepts

### LDraw (LEGO Drawing)
- Free, open-source 3D model library for LEGO pieces
- ~17,000 parts with exact geometry
- Files: `.dat` format (text-based 3D representation)
- Source: http://www.ldraw.org/

### LDU (LEGO Drawing Unit)
- 1 LDU = 0.4 mm
- Used in all LDraw files
- Converted to Blender units via 0.01 scaling

### Rebrickable
- Database of all LEGO sets and parts
- CSV exports available
- Used to identify most common pieces

### Rendering Specification
- **Colors**: 25+ ABS plastic colors + transparent variants
- **Backgrounds**: 5 types (white, gray, beige, dark, gradient)
- **Lighting**: 3-point randomized (key/fill/rim)
- **Camera**: Spherical orbit with 3-6m distance
- **Material**: Physically-based ABS plastic shader

### Output Format
```
output/
├── 3001/           (part number)
│   ├── 00000.png   (image 0)
│   ├── 00001.png   (image 1)
│   └── ...
├── 3002/
│   └── ...
└── render_metadata.json
```

Each PNG is auto-labeled with metadata (part_num, color, background, angle).

---

## Performance Summary

| Scenario | Time | Images | Hardware |
|----------|------|--------|----------|
| Single part (10 renders) | 1 min | 10 | 1 GPU |
| Test batch (100 parts, 60 renders) | 8 hours | 6,000 | 1 GPU |
| Full dataset (3,000 parts, 70 renders) | 120 hours | 210,000 | 1 GPU |
| Full dataset (distributed) | 2 hours | 210,000 | 8 GPUs (DGX) |

---

## File Dependencies

```
ldraw_renderer.py
  ├── Requires: Blender 4.x
  ├── Optional: LDraw importer addon
  ├── Reads from: LDraw DAT files
  └── Outputs: PNG + metadata JSON

get_top_parts.py
  ├── Requires: Python 3.8+
  ├── Reads from: Rebrickable CSV files
  └── Outputs: Top parts list (text file)

setup_blender_ldraw.md
  └── References: Official Blender, LDraw, addon docs

README_PIPELINE.txt
  └── Reference document (no dependencies)
```

---

## Troubleshooting Matrix

| Problem | Cause | Solution |
|---------|-------|----------|
| "LDraw addon not found" | Addon not installed | See setup_blender_ldraw.md section 3 |
| "Library path not found" | Wrong LDraw path | Update addon settings to ~/ldraw |
| "Out of VRAM" | Too high quality settings | Reduce resolution, samples, or engine |
| "Part file not found" | Invalid part number | Check top_parts.txt format |
| "Very slow rendering" | Not using GPU | Install CUDA drivers, check CUDA_VISIBLE_DEVICES |

---

## Contact & References

- **LDraw**: http://www.ldraw.org/
- **LDraw Importer**: https://github.com/LDraw-Loader-for-Blender/ldraw-importer
- **Rebrickable**: https://rebrickable.com/downloads/
- **Blender**: https://www.blender.org/
- **NVIDIA CUDA**: https://developer.nvidia.com/cuda-downloads

