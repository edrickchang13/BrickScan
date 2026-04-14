# Sample LEGO Piece Images for Testing

This directory contains sample LEGO piece images used for testing the BrickScan vision system and scan prediction endpoints.

## Overview

Test images are **not committed to Git** because they are large binary files. Instead, we provide:
- A download script to fetch sample images from public LEGO datasets
- Instructions for organizing images for use in tests
- Guidelines for adding custom test images

## Quick Start

### 1. Download Sample Images

Run the download script to fetch 10 sample LEGO piece photos:

```bash
cd backend/data_pipeline/sample_images
python download_samples.py
```

This script will:
- Download 10 representative LEGO piece photos from Rebrickable's public image CDN
- Organize them by part number
- Create a manifest file for test reference
- Verify image integrity

### 2. Verify Downloaded Images

```bash
python verify_images.py
```

This will:
- Check image formats (JPEG/PNG)
- Verify all images are readable
- Generate metadata for test fixtures
- Report any corrupted images

### 3. Use in Tests

Images are automatically discovered by the test suite:

```python
from pathlib import Path
import json

# Load image manifest
manifest_path = Path(__file__).parent / "manifest.json"
with open(manifest_path) as f:
    images = json.load(f)

# Use in tests
for image in images:
    print(f"Testing {image['part_num']}: {image['path']}")
```

## Image Organization

After downloading, the directory structure looks like:

```
sample_images/
├── README.md                 # This file
├── manifest.json            # Index of all sample images
├── download_samples.py      # Download script
├── verify_images.py         # Verification script
└── images/
    ├── 3001.jpg            # Part 2x4 brick
    ├── 3002.jpg            # Part 2x2 brick
    ├── 3003.jpg            # Part 1x2 brick
    ├── 3069b.jpg           # Part 1x2 tile with groove
    ├── 3626bp00.jpg        # Minifigure head
    ├── 3815.jpg            # Minifigure torso
    ├── 2780c01.jpg         # Wheel
    ├── 3039.jpg            # Slope 45° 2x2
    ├── 32316.jpg           # Technic beam
    └── 6587.jpg            # Technic axle
```

## Manifest File Format

The `manifest.json` file contains metadata for all images:

```json
{
  "generated": "2026-04-11T12:34:56.789Z",
  "total_images": 10,
  "images": [
    {
      "part_num": "3001",
      "name": "Brick 2 x 4",
      "category": "Bricks",
      "path": "images/3001.jpg",
      "size_bytes": 45632,
      "width": 512,
      "height": 512,
      "format": "JPEG",
      "url_source": "https://cdn.rebrickable.com/media/parts/3001.jpg"
    },
    ...
  ]
}
```

## Using Images in Tests

### Example: Scan Prediction Test

```python
import pytest
from pathlib import Path
import json
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_scan_prediction_with_sample_images(client, auth_headers):
    """Test scan prediction using sample LEGO piece images."""
    
    manifest_path = Path(__file__).parent.parent / "sample_images" / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Test with first sample image
    image_info = manifest["images"][0]
    image_path = Path(__file__).parent.parent / image_info["path"]
    
    # Read image as base64
    with open(image_path, "rb") as img_file:
        image_data = img_file.read()
    
    import base64
    base64_image = base64.b64encode(image_data).decode()
    
    # Call scan endpoint
    resp = await client.post(
        "/api/v1/scan/predict",
        json={"image_data": f"data:image/jpeg;base64,{base64_image}"},
        headers=auth_headers
    )
    
    assert resp.status_code == 200
    prediction = resp.json()
    assert "part_num" in prediction
    # May or may not match the actual part due to model accuracy
```

### Example: Batch Testing

```python
@pytest.mark.parametrize("image_info", manifest["images"][:3])
async def test_scan_multiple_parts(client, auth_headers, image_info):
    """Test scanning multiple different parts."""
    image_path = Path(__file__).parent.parent / image_info["path"]
    
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    base64_image = base64.b64encode(image_data).decode()
    
    resp = await client.post(
        "/api/v1/scan/predict",
        json={"image_data": f"data:image/jpeg;base64,{base64_image}"},
        headers=auth_headers
    )
    
    assert resp.status_code == 200
```

## Adding Custom Images

### 1. Find Images

LEGO images can be sourced from:
- **Rebrickable CDN**: https://cdn.rebrickable.com/media/parts/
- **BrickLink**: https://www.bricklink.com/
- **LEGO Official**: https://www.lego.com/
- **Your own photos**: High-quality photos of LEGO pieces work well

### 2. Prepare Images

Requirements:
- Format: JPEG or PNG
- Size: 300-1000px (smaller is faster for tests)
- Compression: ~50-200KB per image
- Background: Neutral (white or light colored preferred)
- Lighting: Clear, well-lit images work best

### 3. Add to Directory

```bash
# Copy your image
cp my_lego_piece.jpg backend/data_pipeline/sample_images/images/

# Update manifest
python backend/data_pipeline/sample_images/update_manifest.py
```

### 4. Update Tests

Reference new images in test fixtures:

```python
@pytest.fixture
def sample_test_images():
    """Get all sample images for testing."""
    manifest_path = Path(__file__).parent.parent / "sample_images" / "manifest.json"
    with open(manifest_path) as f:
        return json.load(f)["images"]
```

## Testing Strategy

### Unit Tests
- Small, focused tests on individual parts
- Fast execution (< 1 second per test)
- Use diverse set of part types

### Integration Tests
- Multi-part scanning workflows
- Set completion checks
- Inventory management with scanned pieces

### Performance Tests
- Large image batches
- Concurrent scan requests
- Memory usage profiling

## Troubleshooting

### Images Won't Download
```bash
# Check internet connection
python -c "import urllib.request; urllib.request.urlopen('https://cdn.rebrickable.com')"

# Retry with verbose output
python download_samples.py --verbose
```

### Corrupted Images
```bash
# Verify and repair
python verify_images.py --repair

# Or re-download specific part
python download_samples.py --part 3001
```

### Manifest Out of Sync
```bash
# Regenerate manifest
python update_manifest.py --full
```

## Performance Notes

- Sample images range from 40KB to 150KB each
- Total directory size: ~800KB (uncompressed)
- Download time: 30-60 seconds depending on connection
- Test execution: ~2-5 seconds per image with full ML pipeline

## Git Ignore

This directory is excluded from Git to avoid committing large binaries:

```gitignore
# In .gitignore at project root
backend/data_pipeline/sample_images/images/
backend/data_pipeline/sample_images/manifest.json
```

The scripts and this README remain in version control for reproducibility.

## CI/CD Integration

In GitHub Actions or other CI environments:

1. **Setup Step**: Download images before running tests
   ```yaml
   - name: Prepare test images
     run: cd backend/data_pipeline/sample_images && python download_samples.py
   ```

2. **Caching**: Speed up repeated runs
   ```yaml
   - uses: actions/cache@v3
     with:
       path: backend/data_pipeline/sample_images/images
       key: lego-test-images-${{ hashFiles('backend/data_pipeline/sample_images/manifest.json') }}
   ```

3. **Cleanup**: Remove images after tests if needed
   ```yaml
   - name: Cleanup
     if: always()
     run: rm -rf backend/data_pipeline/sample_images/images
   ```

## Additional Resources

- **Rebrickable API**: https://rebrickable.com/api/
- **BrickLink Catalog**: https://www.bricklink.com/catalog/
- **LEGO Part Numbers**: https://brickset.com/
- **Image Processing**: See `backend/services/image_processor.py`
