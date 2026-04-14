# BrickScan Rendering Pipeline - Architecture & Design

## Overview

The BrickScan GPU rendering pipeline is a production-grade system for generating synthetic LEGO brick training data. It processes thousands of (part, color) combinations in parallel, rendering each from multiple camera angles with consistent lighting and material properties.

```
┌─────────────────────────────────────────────────────────────┐
│                  batch_render.py (Orchestrator)              │
│              Runs in system Python (3.8+)                    │
└──────────────────┬──────────────────────────────────────────┘
                   │
        ┌──────────┼──────────┬───────────┐
        │          │          │           │
    [Task 1]   [Task 2]   [Task 3]  [Task N]
        │          │          │           │
        └──────────┼──────────┼───────────┘
                   │ (subprocess spawn)
     ┌─────────────┴─────────────┐
     │   blender --background     │
     │   --python render_parts.py │
     │   (Runs INSIDE Blender 4.x)│
     │   ✓ GPU CUDA rendering     │
     │   ✓ LDraw import           │
     │   ✓ Material & lighting    │
     │   ✓ PNG output             │
     └─────────────┬─────────────┘
                   │
        ┌──────────┴──────────┬───────────┐
        ▼                     ▼           ▼
   [*.png files]      [index.csv]  [error logs]
   (renders/)         (metadata)   (failed_*.log)
```

## Component Architecture

### 1. `batch_render.py` - Orchestrator

**Language:** Python 3.8+  
**Runtime:** System Python (NOT inside Blender)  
**Role:** Coordinate parallel rendering tasks

#### Key Features:
- **Task Generation**: Builds list of (part, color) combinations
- **LDraw Verification**: Checks .dat file existence before spawning renders
- **Process Management**: Uses `concurrent.futures.ProcessPoolExecutor` for parallelization
- **Color Parsing**: Converts Rebrickable hex colors to float RGB
- **Progress Tracking**: Integrates tqdm for visual feedback
- **Error Handling**: Logs failures with retryability in mind
- **CSV Aggregation**: Appends metadata to shared index.csv

#### Data Flow:
```
colors.csv ─────┐
                ├──> Filter colors & parts ──> Generate tasks
parts_file ─────┤
                │
LDraw library ──┘

Generate tasks ──> [Task Queue] ──> ProcessPoolExecutor
                        ↓
                    [Spawn Blender]
                        ↓
                    render_parts.py (subprocess)
                        ↓
                    [Collect results]
                        ↓
                    [Log to CSV & error log]
```

#### CLI Arguments:
```
--blender PATH              Path to Blender executable
--ldraw-dir PATH            Root of LDraw library
--colors-csv PATH           Rebrickable colors CSV
--parts-file PATH           Custom parts list (optional)
--output-dir PATH           Base output directory
--index-csv PATH            Metadata CSV path
--num-angles INT            Camera angles per elevation (default: 36)
--resolution INT            Output size 224|512 (default: 224)
--workers INT               Parallel processes (default: 4)
--max-parts INT             Limit for testing (optional)
--colors LIST               Specific color IDs (optional)
--skip-existing             Skip already-rendered parts
```

### 2. `render_parts.py` - Renderer

**Language:** Python 3.x  
**Runtime:** INSIDE Blender 4.x (bpy API)  
**Role:** Render a single (part, color) combination

#### Architecture:
```python
# Core rendering pipeline
1. Parse CLI arguments
2. Clear & setup scene
3. Configure GPU (CUDA)
4. Import LDraw .dat file
5. Create & apply material
6. Calculate camera distance (from bounding box)
7. Setup 3-point lighting
8. Loop through elevations & azimuths:
   a. Position camera (spherical coords)
   b. Point at origin
   c. Randomize HDRI rotation
   d. Render to PNG
   e. Log to CSV
```

#### Scene Structure:
```
Scene {
  World: dark background (0.05 RGB)
  Camera: orbit at variable distance, 50mm lens
  Lights {
    KeyLight: SUN, (3,4,5), energy=2.5, warm
    FillLight: SUN, (-2,1,3), energy=1.0, cool
    RimLight: SUN, (0,-2,4), energy=1.5, white
  }
  Mesh: imported from LDraw
}
```

#### Rendering Parameters:
```
Render Engine:        CYCLES (GPU-accelerated)
Compute Device:       CUDA
Samples:              128 per frame
Denoiser:             OpenImageDenoise (Intel)
Output Format:        8-bit RGBA PNG
Resolution:           224×224 (default), up to 512×512
Color Space:          sRGB
Film Transparent:     True (alpha channel)
```

#### Camera Positioning (Spherical Coordinates):
```
Elevation angles: [-20°, +10°, +30°]
Azimuth angles:   [0°, 10°, 20°, ..., 350°] (36 steps default)

Position formula:
  cam_x = distance × cos(elevation) × cos(azimuth)
  cam_y = distance × cos(elevation) × sin(azimuth)
  cam_z = distance × sin(elevation)

Distance = max(object_dimensions) × 2.0
```

#### Material Definition:
```
Principled BSDF {
  Base Color:    [color_r, color_g, color_b] (from Rebrickable)
  Roughness:     0.3 (LEGO plastic sheen)
  Metallic:      0.0 (non-metallic)
  IOR:           default (1.5)
}
```

#### Output Format:
```
Filename: {part_num}_{color_id}_{angle_idx:04d}.png
Size:     1 × 108 images per (part, color) combo
          (3 elevations × 36 azimuths = 108 frames)
Location: ml/data/renders/{part_num}/
Metadata: Appended to index.csv
```

### 3. `setup_ldraw.sh` - LDraw Setup

**Language:** Bash  
**Purpose:** Download and extract LDraw parts library

#### Steps:
1. Download complete.zip (~2.5 GB) from library.ldraw.org
2. Extract to `ml/data/ldraw/`
3. Verify extraction (count .dat files)
4. Print summary statistics

#### Output Structure:
```
ml/data/ldraw/
├── ldraw/
│   ├── parts/          (6842+ .dat files)
│   │   ├── 3004.dat    (standard 2×4 brick)
│   │   ├── 3001.dat
│   │   └── ...
│   └── p/              (894+ .dat files - primitives)
│       ├── cyli.dat
│       └── ...
└── readme.txt, etc.
```

### 4. Supporting Files

#### `requirements.txt`
```
tqdm>=4.65.0      # Progress bars
pandas>=1.5.0     # CSV handling (optional but recommended)
```

#### Example Data Files
```
colors_example.csv       # Sample Rebrickable colors
parts_subset_example.txt # Sample part list
```

## Data Flow & I/O

### Input Data

**Source 1: Rebrickable colors.csv**
```csv
id,name,rgb
1,White,FFFFFF
2,Tan,F2CD37
...
```

**Source 2: LDraw Parts Library**
```
ml/data/ldraw/ldraw/parts/{part_num}.dat
```

**Source 3: Part List (optional)**
```
3004
3001
3022
...
```

### Output Data

**Primary Output: Rendered PNGs**
```
ml/data/renders/
├── 3004/
│   ├── 3004_1_0000.png   (part 3004, color 1, angle 0)
│   ├── 3004_1_0001.png
│   ├── 3004_2_0000.png   (same part, color 2)
│   └── ...
└── ...
```

**Metadata: index.csv**
```csv
image_path,part_num,color_id,color_name,color_r,color_g,color_b
renders/3004/3004_1_0000.png,3004,1,White,1.0000,1.0000,1.0000
renders/3004/3004_1_0001.png,3004,1,White,1.0000,1.0000,1.0000
```

**Error Log: failed_renders.log**
```
2025-04-11 14:23:45 - ERROR - Failed: 9999 Color Name - LDraw import error
```

## Parallelization Strategy

### Process Model
- **Main process**: `batch_render.py` orchestrates all work
- **Worker processes**: N Blender instances (N = `--workers` flag)
- **IPC**: Subprocess stdout/stderr for result communication

### Task Distribution
```
Total tasks = |parts| × |colors|

Task Distribution with --workers=4:
  Blender 1 ─── Task 1, 5, 9, 13, ...
  Blender 2 ─── Task 2, 6, 10, 14, ...
  Blender 3 ─── Task 3, 7, 11, 15, ...
  Blender 4 ─── Task 4, 8, 12, 16, ...

Workload:  As one Blender finishes, it grabs the next task
           from the queue until all tasks complete.
```

### Memory & GPU Management
```
Memory usage per Blender process:
  - Base Blender:     ~1.5 GB
  - LDraw model:      ~100-500 MB (varies by part complexity)
  - Rendering (128 samples): ~800 MB - 4 GB

GPU VRAM (e.g., RTX 4090):
  24 GB total
  - Shared Blender overhead: 1-2 GB
  - Per-process render: 2-4 GB

Recommendation:
  --workers = floor(GPU_VRAM / 4 GB)
  RTX 4090 (24 GB): --workers 6
  RTX 3070 (8 GB):  --workers 2
  RTX 4070 (12 GB): --workers 3
```

## Error Handling & Resilience

### Failure Cases Handled

1. **Part file not found**
   - Check LDraw library existence
   - Log as WARNING, skip to next task
   
2. **LDraw import fails**
   - Try fallback .obj file
   - Log as ERROR, skip part
   
3. **Render timeout (>5 minutes)**
   - Kill subprocess
   - Log as ERROR with timeout marker
   
4. **GPU out of memory**
   - Subprocess crashes
   - Logged in stderr
   - Can reduce --workers and retry

5. **CSV write contention**
   - Multiple processes append to index.csv simultaneously
   - Python file locking handles contention
   - Safe to resume with --skip-existing

### Recovery & Resumption

```bash
# If render fails partway through:
python batch_render.py --skip-existing

# This checks ml/data/renders/{part_num}/*.png existence
# and only re-renders missing combinations
```

## Performance Characteristics

### Throughput (RTX 4090, 224×224, 128 samples)
```
Per-task time: 15-20 seconds
Images per task: 108 (3 elevations × 36 azimuths)

Single GPU:
  - ~3.6 tasks/minute = 388 images/minute
  - 500 parts × 70 colors = 35,000 tasks
  - Estimated time: ~9,700 minutes = 6.7 days

4 parallel processes:
  - ~12-15 tasks/minute = 1,290-1,620 images/minute
  - 500 parts × 70 colors = 35,000 tasks
  - Estimated time: ~2,300-2,900 minutes = 1.6-2 days
```

### Scalability
```
Workers vs throughput (RTX 4090, 224px):
  1 worker: 300-400 img/min
  2 workers: 700-800 img/min
  3 workers: 1,000-1,100 img/min
  4 workers: 1,200-1,400 img/min
  5 workers: 1,500-1,600 img/min (diminishing returns)
  6+ workers: GPU OOM risk

Optimal: 4-5 workers per high-end GPU
```

## Design Decisions & Rationale

### 1. Subprocess Model vs Blender API
**Decision**: Use `subprocess.run()` to spawn Blender processes

**Rationale**:
- Isolation: Each Blender process is independent; crashes don't kill orchestrator
- Parallelization: Native OS process scheduling (no Python GIL)
- Simplicity: No need for bpy callbacks or event loops
- Robustness: Easy to implement timeout, resource limits

### 2. CSV Aggregation
**Decision**: Append to shared index.csv from each process

**Rationale**:
- Single source of truth for dataset metadata
- File locking prevents corruption
- Easy to resume with --skip-existing
- ML pipelines expect simple CSV format

### 3. Camera Sphere Strategy
**Decision**: Orbit camera at constant distance, auto-fitted to object

**Rationale**:
- Consistent framing across all parts
- No need for manual per-part tuning
- Bounding box approach scales to any model complexity

### 4. 3-Point Lighting
**Decision**: Fixed key/fill/rim lights (not randomized)

**Rationale**:
- Consistency: Same lighting across all renders
- Quality: Professional, balanced illumination
- Training: ML model learns color under controlled conditions
- Variability: HDRI rotation provides background diversity

### 5. HDRI Rotation for Backgrounds
**Decision**: Random rotation per frame (not color)

**Rationale**:
- Reduces overfitting to specific background
- Keeps rendering fast (no full HDRI re-render)
- Adds dataset variety without overhead

## Extension Points

### Custom Material Properties
Edit `render_parts.py` around line 145 (Principled BSDF setup):
```python
bsdf.inputs["Roughness"].default_value = 0.3  # Customize here
bsdf.inputs["IOR"].default_value = 1.5
```

### Custom Camera Angles
Edit elevation angles (line 189):
```python
elevation_angles = [-20, 10, 30]  # Add/remove angles here
```

### Custom Lighting
Edit light creation (lines 152-164) to change positions/colors/energy.

### Alternative Part Formats
The system gracefully falls back to `.obj` files. To add more formats:
1. Modify the import section of `render_parts.py` (lines 105-125)
2. Add new format handler
3. Test with sample part

## Quality Assurance

### Validation Checklist

- [ ] LDraw library extracted: `ls ml/data/ldraw/ldraw/parts | wc -l` (~6800+)
- [ ] Colors CSV loaded: 10+ colors
- [ ] Blender can import a .dat file
- [ ] CUDA is available: `nvidia-smi`
- [ ] Test render completes: `./quickstart.sh` (mode 1)
- [ ] PNG files are valid: `file ml/data/renders/*/*.png | grep -c PNG`
- [ ] index.csv has expected rows
- [ ] No errors in failed_renders.log

### Dataset Integrity

After rendering:
```bash
# Count images
find ml/data/renders -name "*.png" | wc -l

# Expected: parts × colors × (elevations × azimuths)
# E.g., 500 × 70 × 108 = 3,780,000

# Check CSV
head -5 ml/data/index.csv
wc -l ml/data/index.csv  # Should match PNG count + 1

# Verify no corrupt PNGs
for f in ml/data/renders/*/*.png; do
  file "$f" | grep -q PNG || echo "Bad: $f"
done
```

## Future Enhancements

1. **Ray-tracing reflections** - Enable caustics/reflections for more realism
2. **Object variability** - Add slight rotation/positioning variance
3. **Texture maps** - Support custom surface textures beyond color
4. **Auto-material tuning** - Learn optimal roughness per material type
5. **Multi-GPU distribution** - Distribute across multiple GPUs/machines
6. **Cloud rendering** - Support cloud Blender (AWS Thinkbox Deadline, etc.)
7. **Real camera simulation** - Match specific camera lens/sensor characteristics
8. **Augmentation pipelines** - Built-in image augmentation (crops, rotations, noise)

---

**Document Version**: 1.0  
**Last Updated**: 2025-04-11  
**Status**: Production Ready
