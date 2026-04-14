# BrickScan Rendering Examples

Complete examples for common workflows.

## Example 1: First-Time Setup

```bash
cd ml/blender

# 1. Install dependencies
pip install -r requirements.txt

# 2. Download LDraw library (one-time, ~2.5 GB)
./setup_ldraw.sh

# Expected output:
# ================================================== 
# LDraw Library Summary
# ================================================== 
# Parts directory: ml/data/ldraw/ldraw/parts
# Part .dat files found: 6842
# Primitives .dat files: 894
# Total .dat files: 7736
```

## Example 2: Render Top 10 Parts (Testing)

Render the first 10 parts in all 20 example colors, 4 angles each:

```bash
python batch_render.py \
  --blender blender \
  --ldraw-dir ../data/ldraw \
  --colors-csv ../data/colors_example.csv \
  --output-dir ../data/renders \
  --max-parts 10 \
  --num-angles 4 \
  --resolution 224 \
  --workers 2
```

**Expected output:**
```
BrickScan batch render
Loaded 20 colors
Found 6842 parts in LDraw library
Limited to 10 parts
Total tasks: 200 (10 parts × 20 colors)

Rendering: 45%|████▌     | 90/200
```

**Output structure:**
```
ml/data/renders/
├── 3004/
│   ├── 3004_1_0000.png    (part 3004, color 1 "White", angle 0)
│   ├── 3004_1_0001.png
│   ├── 3004_1_0002.png
│   ├── 3004_1_0003.png
│   ├── 3004_2_0000.png    (part 3004, color 2 "Tan", angle 0)
│   └── ...
└── index.csv
```

**index.csv excerpt:**
```csv
image_path,part_num,color_id,color_name,color_r,color_g,color_b
renders/3004/3004_1_0000.png,3004,1,White,1.0000,1.0000,1.0000
renders/3004/3004_1_0001.png,3004,1,White,1.0000,1.0000,1.0000
renders/3004/3004_2_0000.png,3004,2,Tan,0.9490,0.8039,0.2157
```

## Example 3: Custom Parts Subset

Render only a specific set of parts (useful for targeted datasets):

```bash
# Create a custom parts file
cat > custom_parts.txt << EOF
3004
3001
3022
3045
EOF

# Render with custom parts
python batch_render.py \
  --blender blender \
  --ldraw-dir ../data/ldraw \
  --colors-csv ../data/colors.csv \
  --parts-file ./custom_parts.txt \
  --output-dir ../data/renders \
  --num-angles 36 \
  --resolution 224 \
  --workers 4
```

## Example 4: Single Color Across All Parts

Render all parts in just white:

```bash
python batch_render.py \
  --blender blender \
  --ldraw-dir ../data/ldraw \
  --colors-csv ../data/colors.csv \
  --colors 1 \
  --num-angles 36 \
  --resolution 224 \
  --workers 4
```

## Example 5: Single Part Test Render

Render one part in one color from multiple angles (good for manual inspection):

```bash
blender --background --python render_parts.py -- \
  --part-file ../data/ldraw/ldraw/parts/3004.dat \
  --output-dir /tmp/test_render \
  --part-num 3004 \
  --color-id 1 \
  --color-name White \
  --color-r 1.0 \
  --color-g 1.0 \
  --color-b 1.0 \
  --num-angles 36 \
  --resolution 224
```

Check output:
```bash
ls /tmp/test_render/
# 3004_1_0000.png
# 3004_1_0001.png
# ...
# (36 images × 3 elevations = 108 total)
```

## Example 6: High-Resolution Training Data

For production ML model training:

```bash
python batch_render.py \
  --blender blender \
  --ldraw-dir ../data/ldraw \
  --colors-csv ../data/colors.csv \
  --output-dir ../data/renders_hd \
  --num-angles 36 \
  --resolution 512 \
  --workers 2 \
  --skip-existing
```

**Notes:**
- `--resolution 512` increases render time by ~4x
- Reduce `--workers` to avoid GPU OOM
- `--skip-existing` skips parts already fully rendered

## Example 7: Resume Failed Renders

If a batch fails partway through:

```bash
# Check failed_renders.log
tail -100 ../data/failed_renders.log

# Re-run with same parameters
python batch_render.py \
  --blender blender \
  --ldraw-dir ../data/ldraw \
  --colors-csv ../data/colors.csv \
  --output-dir ../data/renders \
  --skip-existing  # Only re-render missing parts
```

## Example 8: Distributed Rendering (Multi-GPU / Multi-Machine)

Split the workload across 4 machines:

**Machine 1:**
```bash
# Render parts 0-25
python batch_render.py \
  --blender blender \
  --ldraw-dir ../data/ldraw \
  --colors-csv ../data/colors.csv \
  --output-dir /shared/renders_m1 \
  --max-parts 25
```

**Machine 2:**
```bash
# Render parts 25-50
python batch_render.py \
  --blender blender \
  --ldraw-dir ../data/ldraw \
  --colors-csv ../data/colors.csv \
  --output-dir /shared/renders_m2 \
  --max-parts 50
```

(Repeat for other machines, adjusting `--max-parts`)

Then merge results:
```bash
# Merge all renders
cp -r /shared/renders_m1/* ../data/renders/
cp -r /shared/renders_m2/* ../data/renders/

# Merge index CSVs (skip header rows after first)
head -1 /shared/renders_m1/index.csv > merged_index.csv
tail -n +2 /shared/renders_m1/index.csv >> merged_index.csv
tail -n +2 /shared/renders_m2/index.csv >> merged_index.csv
```

## Example 9: Debug a Specific Part

If a part fails to import, debug it:

```bash
# Check if .dat file exists
find ../data/ldraw -name "3004.dat"

# Try to render with verbose output
blender --verbose --python render_parts.py -- \
  --part-file ../data/ldraw/ldraw/parts/3004.dat \
  --output-dir /tmp/debug \
  --part-num 3004 \
  --color-id 1 \
  --color-name Test \
  --color-r 0.5 --color-g 0.5 --color-b 0.5

# Check output
ls -lh /tmp/debug/
```

## Example 10: Quality Inspection Workflow

After rendering, inspect output quality:

```bash
# Count rendered images
find ../data/renders -name "*.png" | wc -l

# Check one complete part render
ls ../data/renders/3004/ | head -20

# Verify CSV consistency
head -5 ../data/renders/../index.csv
wc -l ../data/renders/../index.csv

# Find any problematic images (corrupt PNGs)
for f in ../data/renders/*/*.png; do
  file "$f" | grep -q "PNG image" || echo "Bad: $f"
done
```

## Performance Tuning

### Fast Iteration (for testing ML model)
```bash
python batch_render.py \
  --num-angles 8 \
  --resolution 128 \
  --workers 4

# Edit render_parts.py: scene.cycles.samples = 32
```

### Production Quality (for final training set)
```bash
python batch_render.py \
  --num-angles 36 \
  --resolution 512 \
  --workers 2

# Edit render_parts.py: scene.cycles.samples = 256
```

### Memory-Constrained System
```bash
python batch_render.py \
  --workers 1 \
  --resolution 224 \
  --num-angles 12

# Edit render_parts.py: scene.cycles.samples = 64
```

## Monitoring Render Progress

```bash
# In one terminal, watch progress
tail -f ../data/failed_renders.log

# In another, monitor GPU
watch -n 1 nvidia-smi

# Or check file count growth
watch -n 5 'find ../data/renders -name "*.png" | wc -l'
```

## Troubleshooting Examples

### CUDA not found
```bash
# Test Blender CUDA setup
blender --background --python -c "
import bpy
prefs = bpy.context.preferences.addons['cycles'].preferences
print('CUDA devices:', prefs.get_devices())
"
```

### Memory error during rendering
```bash
# Reduce parallelism
python batch_render.py --workers 1

# Or reduce resolution
python batch_render.py --resolution 128

# Edit render_parts.py and reduce samples:
# scene.cycles.samples = 64
```

### Part import failed
```bash
# Check if part exists in LDraw
find ../data/ldraw -name "3004.dat"

# If not found, verify setup_ldraw.sh completed
ls ../data/ldraw/ldraw/parts/ | wc -l  # Should be ~6000+

# If still not found, part may not exist in LDraw
# Check against official catalog: https://library.ldraw.org/
```
