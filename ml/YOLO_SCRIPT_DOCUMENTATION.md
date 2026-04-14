# YOLOv8 LEGO Detector - Script Documentation

Complete technical documentation for the `train_yolo_detector.py` script.

## File Structure

```
train_yolo_detector.py
├── Imports & Configuration
├── BackgroundGenerator (Class)
├── SyntheticDataGenerator (Class)
├── ensure_ultralytics_installed()
├── train_yolo_model()
├── export_to_onnx()
├── save_training_metrics()
├── main()
├── parse_arguments()
└── __main__ entry point
```

## Component Breakdown

### 1. BackgroundGenerator Class

Static methods for generating varied backgrounds to ensure model robustness.

#### `solid_color(size: Tuple[int, int]) -> Image.Image`
- Generates random solid color backgrounds
- Colors: white, light/medium/dark gray, off-white
- Returns PIL Image RGB

#### `gradient(size: Tuple[int, int]) -> Image.Image`
- Creates linear color gradients
- Random start and end colors
- Smooth interpolation across Y axis

#### `noise(size: Tuple[int, int]) -> Image.Image`
- Base color with random noise overlay
- 10% pixel mutation with ±30 intensity variance
- Creates textured appearance

#### `texture(size: Tuple[int, int]) -> Image.Image`
- Simulates wood grain texture
- Uses sine waves for pattern
- RGB variation: G = R-20, B = R-40

#### `generate(size: Tuple[int, int]) -> Image.Image`
- Randomly selects and applies one background method
- Ensures diverse backgrounds in training data

### 2. SyntheticDataGenerator Class

Main class for creating YOLO-format detection training data.

#### Constructor
```python
__init__(self, parts_dir, output_dir, image_size=640, 
         num_train=5000, num_val=1000)
```

**Arguments:**
- `parts_dir`: Path to directory with rendered part images (PNG)
- `output_dir`: Base directory for output dataset
- `image_size`: Square image dimensions (pixels)
- `num_train`: Number of training images to generate
- `num_val`: Number of validation images to generate

**Directory Creation:**
- `output_dir/images/train/` - Training images (JPG)
- `output_dir/labels/train/` - Training annotations (TXT)
- `output_dir/images/val/` - Validation images (JPG)
- `output_dir/labels/val/` - Validation annotations (TXT)
- `output_dir/data.yaml` - YOLO dataset configuration

#### `_load_part_images() -> List[Path]`
- Recursively finds all PNG files in parts directory
- Returns list of Path objects
- Raises ValueError if no images found

#### `_load_image_with_alpha(image_path: Path) -> Tuple[Image, Image]`
- Handles multiple image formats:
  - RGBA: Extracts existing alpha channel
  - RGB: Creates opaque alpha mask (all 255)
  - Other: Converts to RGBA first
- Returns (RGB_image, alpha_mask) tuple
- Essential for transparency-aware compositing

#### `_get_part_bbox(alpha_mask: Image, offset_x: int, offset_y: int)`
- Extracts bounding box from non-transparent pixels
- Thresholds alpha at 127 (>127 = opaque)
- Adjusts coordinates for image offset
- Returns `(x_min, y_min, x_max, y_max)` in absolute pixels
- Returns None if mask is empty

#### `_apply_random_transformations(img: Image) -> Image`
- Random rotation: ±15 degrees (70% probability)
- Random scale: 0.8x to 1.2x (50% probability)
- Maintains image quality with LANCZOS resampling

#### `generate_image(is_train: bool = True) -> Tuple[Image, List[str]]`

**Core synthetic data generation method.**

Process:
1. Create randomized background
2. Random part count: 3-15 pieces
3. For each part:
   - Select random part image
   - Apply random transformations
   - Random scale: 5-25% of image width
   - Random position (allows overlaps)
   - Composite onto background with transparency
   - Extract bbox from alpha channel
4. Convert bbox to YOLO normalized format
5. Apply optional final augmentation (contrast, blur)

**YOLO Format Annotation:**
- Class ID: 0 (single class "lego_piece")
- Normalized center_x: (bbox_center_x / image_width)
- Normalized center_y: (bbox_center_y / image_height)
- Normalized width: (bbox_width / image_width)
- Normalized height: (bbox_height / image_height)

Example output:
```
0 0.523400 0.412100 0.185300 0.243800
0 0.712500 0.623400 0.142100 0.198700
```

#### `generate_dataset()`

Main generation loop:
1. Create 5000 training images
   - Save as `img_XXXXX.jpg` (quality=95)
   - Corresponding `img_XXXXX.txt` annotations
2. Create 1000 validation images
3. Create `data.yaml` configuration file

**Output data.yaml example:**
```yaml
path: /full/path/to/dataset
train: images/train
val: images/val

nc: 1
names: ['lego_piece']
```

#### `_create_yolo_yaml()`
- Writes YAML config for YOLOv8 training
- Includes absolute path, train/val splits, class info

### 3. Training Functions

#### `ensure_ultralytics_installed()`
- Checks if `ultralytics` module is available
- Installs via pip if missing: `pip install -q ultralytics`
- Logs version information

#### `train_yolo_model(dataset_config, output_dir, epochs=100, patience=15, batch_size=32, imgsz=640, device='0')`

**Loads YOLOv8m and trains on generated data.**

**Key Training Parameters:**
- `mosaic=1.0`: Full mosaic augmentation (combines 4 images)
- `mixup=0.1`: 10% mixup probability per batch
- `hsv_h=0.015`: ±1.5% hue shift
- `hsv_s=0.7`: ±70% saturation shift
- `hsv_v=0.4`: ±40% value shift
- `lr0=0.01`: Initial learning rate
- `lrf=0.01`: Final learning rate
- `momentum=0.937`: SGD momentum
- `weight_decay=0.0005`: L2 regularization
- `warmup_epochs=3`: Gradual LR warmup

**Geometric Augmentation:**
- `degrees=10`: ±10° rotation
- `translate=0.1`: ±10% translation
- `scale=0.5`: 0.5x to 2.0x scale range
- `flipud=0.5`: 50% vertical flip
- `fliplr=0.5`: 50% horizontal flip

**Training Dynamics:**
- Saves checkpoint every 10 epochs
- Early stopping with configurable patience
- Validates on val set after each epoch
- Selects best model based on fitness metric

#### `export_to_onnx(model_path, output_dir)`

**Exports trained PyTorch model to ONNX format.**

- Loads best.pt model
- Exports with `format='onnx'`, `half=False` (full precision)
- Saves to `output_dir/best.onnx`
- ONNX models are:
  - Framework-agnostic
  - Deployable on CPU/GPU/mobile
  - Compatible with ONNX Runtime

#### `save_training_metrics(results, output_dir)`

**Saves training results to JSON.**

Captured metrics:
- Total epochs completed
- Best fitness score achieved
- Other metrics available from results object

Output file: `output_dir/training_metrics.json`

### 4. Main Pipeline

#### `main(args)`

Three-stage pipeline:

**Stage 1: Data Generation (if --no-generate-data not set)**
- Validates parts directory exists
- Creates SyntheticDataGenerator instance
- Generates training and validation datasets
- Creates YOLO configuration YAML

**Stage 2: Training (if --no-train not set)**
- Validates dataset config exists
- Trains YOLOv8m model
- Saves results and metrics

**Stage 3: Export (if --no-export not set)**
- Validates trained model exists
- Exports to ONNX format

### 5. Argument Parsing

#### Data Arguments
- `--parts-dir`: Source rendered parts (default: ~/brickscan/ml/data/test_renders)
- `--output-dir`: All results output (default: ./yolo_detector)
- `--image-size`: Training image size (default: 640)
- `--num-train`: Training samples (default: 5000)
- `--num-val`: Validation samples (default: 1000)

#### Training Arguments
- `--epochs`: Total epochs (default: 100)
- `--batch-size`: Batch size (default: 32)
- `--patience`: Early stop patience (default: 15)
- `--device`: CUDA device ID (default: '0')

#### Pipeline Control
- `--no-generate-data`: Skip data generation
- `--no-train`: Skip training
- `--no-export`: Skip ONNX export

All user paths expanded with `os.path.expanduser()` to support `~` notation.

## Data Flow

```
Parts Images (PNG with transparency)
        ↓
SyntheticDataGenerator
├── Background Generation
│   └── Random color/gradient/noise/texture
├── Part Compositing
│   ├── Random selection
│   ├── Transform (rotate, scale)
│   ├── Random position
│   └── Alpha composite onto background
├── Bbox Extraction
│   └── From alpha mask to YOLO format
└── Save Results
    ├── JPG image
    └── TXT annotation (YOLO format)
        ↓
YOLO Dataset Structure
├── images/train/ (5000 JPGs)
├── labels/train/ (5000 TXTs)
├── images/val/ (1000 JPGs)
├── labels/val/ (1000 TXTs)
└── data.yaml
        ↓
YOLOv8m Training
└── output/yolo_detector/
    ├── weights/best.pt
    ├── weights/last.pt
    ├── results.csv
    └── plots/
        ↓
ONNX Export
└── models/best.onnx
```

## Key Implementation Details

### Bounding Box Extraction
- Uses alpha channel threshold (127) for edge detection
- `np.where()` finds all non-transparent pixels
- Gets min/max coordinates automatically
- Accounts for part position offset

### YOLO Format Normalization
```python
center_x = (x_min + x_max) / 2 / image_width
center_y = (y_min + y_max) / 2 / image_height
width = (x_max - x_min) / image_width
height = (y_max - y_min) / image_height
```
Values always 0-1 after clipping.

### Image Quality Tradeoffs
- Training images: JPG quality=95 (good balance)
- Format: JPG (smaller than PNG, sufficient for detection)
- Annotations: TXT (minimal, human-readable)

### Augmentation Strategy
**Data Generation:**
- Part variations (rotation, scale)
- Background diversity (4 types)
- Overlapping pieces (realistic)
- Random composition (each image unique)

**Training:**
- Mosaic (4-image tiles)
- Mixup (pixel-level blending)
- HSV shift (color robustness)
- Geometric (rotation, flip, scale)

### Memory Efficiency
- Parts loaded on-demand per image
- Background generated per image
- No dataset pre-caching
- Scales well with number of parts

### Robustness Features
- Bbox clipping to valid range
- Empty annotation filtering
- Error handling for malformed images
- Logging at each stage
- Path expansion support

## Performance Characteristics

### Data Generation
- Per-image time: ~100-200ms
- I/O bound (PIL operations)
- Single-threaded (could parallelize)
- 5000 training images: ~8-16 minutes

### Training
- Batch time: ~500ms @ batch_size=32
- 100 epochs × ~156 batches = ~13,000 batches
- Estimated: 6,500 seconds ≈ 1.8 hours
- With validation overhead: 3-4 hours typical

### GPU Utilization
- YOLOv8m uses ~5-8GB VRAM @ batch_size=32
- Can increase batch_size to 64+ on GB10
- Better GPU utilization = faster training

## Common Modifications

### Increase Part Density
```python
# In generate_image()
num_parts = random.randint(10, 30)  # More crowded scenes
```

### Restrict Part Sizes
```python
# In generate_image()
scale = random.uniform(0.15, max_scale)  # Larger minimum
```

### Add Custom Backgrounds
```python
# In BackgroundGenerator class
@staticmethod
def custom_pattern(size):
    # Implementation here
    return img

# In generate() method
method = random.choice([..., cls.custom_pattern])
```

### Adjust Training Hyperparameters
```python
# In train_yolo_model()
model.train(
    epochs=200,
    patience=20,
    momentum=0.95,
    weight_decay=0.001,
)
```

## Debugging & Validation

### Check Generated Data
```bash
# Visually inspect training images
ls -lh yolo_detector/dataset/images/train/ | head -20
file yolo_detector/dataset/images/train/img_00000.jpg

# Check annotations format
head -3 yolo_detector/dataset/labels/train/img_00000.txt
```

### Verify YOLO Format
```bash
# Count images vs labels
ls -1 yolo_detector/dataset/images/train/*.jpg | wc -l
ls -1 yolo_detector/dataset/labels/train/*.txt | wc -l
# Should match
```

### Monitor Training
```bash
# Follow live training output
tail -f yolo_detector/training/yolo_detector/results.csv

# Check best model
ls -lh yolo_detector/training/yolo_detector/weights/best.pt
```

## Dependencies

**Required (pre-installed in venv):**
- torch
- torchvision
- PIL/Pillow
- numpy
- pandas
- tqdm
- cv2 (optional, imported but not essential)

**Auto-installed:**
- ultralytics

**Python Version:** 3.8+

## Error Handling

Script handles:
- Missing parts directory → sys.exit(1)
- No parts found → ValueError with details
- Missing dataset.yaml during training → sys.exit(1)
- Missing best.pt during export → sys.exit(1)
- Keyboard interrupt (Ctrl+C) → graceful exit
- All exceptions → logged with traceback

## Future Enhancements

Potential improvements:
1. **Parallel data generation** with multiprocessing
2. **Augmentation variations** per dataset split
3. **Custom class labels** beyond single-class detection
4. **Real image mixing** (combine synthetic + real data)
5. **Interactive data exploration** with visualization tools
6. **Model ensemble** training
7. **Distributed training** on multiple GPUs
8. **Quantized export** (int8 ONNX)
