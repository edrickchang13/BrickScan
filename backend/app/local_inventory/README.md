# BrickScan Local Inventory System

A production-quality device-local LEGO brick scanning and inventory management system. Users can scan their physical LEGO bricks with a mobile phone camera, build a local SQLite database of every part they own, and handle uncertain predictions with confidence-based user confirmation.

## Features

- **Offline-First**: All data stored locally on device (SQLite at `~/brickscan_inventory.db`)
- **Confidence-Aware**: Distinguishes "known" predictions (≥80%) from "uncertain" (<80%)
- **Image Storage**: Saves uncertain scans to `~/brickscan_images/` for later review/retraining
- **No Authentication**: Device-local, no user accounts or API keys
- **Session Management**: Group scans into named sessions (e.g., "Technic 42145")
- **CSV Export**: Export full inventory as CSV for backup or analysis
- **Correction Support**: Let users fix incorrect predictions
- **Production-Ready**: Error handling, logging, validation, unit tests

## Architecture

```
Mobile Camera Image (JPEG/PNG)
  ↓ [validate & decode base64]
  ↓ [resize 224×224 & normalize]
  ↓ [run ONNX inference]
  ↓ [return top-3 predictions]
  ↓ [determine status: "known" ≥80% or "uncertain" <80%]
  ↓ [save image if uncertain]
  → Response with predictions + status
```

## File Structure

```
backend/app/local_inventory/
├── __init__.py           Module docstring
├── models.py             SQLAlchemy ORM models
├── database.py           SQLite setup + session mgmt
├── schemas.py            Pydantic request/response models
├── image_processor.py    Image validation + preprocessing
├── routes.py             FastAPI endpoints (main file)
├── constants.py          Configuration constants
├── utils.py              Helper functions
├── test_routes.py        Unit tests
├── DESIGN.md             Technical design document
├── CLIENT_USAGE.md       Mobile/web client integration guide
└── README.md             This file
```

## Quick Start

### Installation

```bash
# Install dependencies (already in requirements.txt)
pip install fastapi sqlalchemy pillow onnxruntime

# Database is auto-created at ~/brickscan_inventory.db on first use
```

### Integration

The routes are auto-registered in `backend/main.py`:

```python
from app.local_inventory import routes as local_inventory_routes

# In FastAPI app setup:
app.include_router(local_inventory_routes.router)
```

### Basic Usage

```python
import requests
import base64

# Scan a brick
with open("brick_photo.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8000/api/local-inventory/scan",
    json={"image_base64": image_b64}
)
result = response.json()

# Result: status="known" or "uncertain", predictions with top-3

# Add to inventory
requests.post(
    "http://localhost:8000/api/local-inventory/inventory/add",
    json={
        "part_num": result["primary_prediction"]["part_num"],
        "color_id": result["primary_prediction"]["color_id"],
        "color_name": result["primary_prediction"]["color_name"],
        "quantity": 1
    }
)

# View inventory
response = requests.get("http://localhost:8000/api/local-inventory/inventory")
print(response.json())
```

See `CLIENT_USAGE.md` for detailed examples with error handling.

## API Endpoints

All endpoints are prefixed with `/api/local-inventory`:

### Scanning
- `POST /scan` — Scan image, return prediction + confidence
- `GET /inventory/stats` — Get aggregate inventory statistics

### Inventory Management
- `POST /inventory/add` — Add confirmed part to inventory
- `GET /inventory` — List all parts
- `PUT /inventory/{id}` — Update part quantity
- `POST /inventory/{id}/correct` — Correct a mislabeled part
- `DELETE /inventory/{id}` — Remove part from inventory
- `GET /inventory/export` — Export as CSV

### Session Management
- `POST /scan-session/start` — Start a named session
- `GET /scan-session` — List sessions (with `completed_only` filter)
- `POST /scan-session/{id}/complete` — Mark session done
- `DELETE /scan-session/{id}` — Delete session

See `DESIGN.md` for full endpoint documentation with request/response examples.

## Database Schema

### LocalInventoryPart
```sql
CREATE TABLE local_inventory_parts (
    id UUID PRIMARY KEY,
    part_num VARCHAR(50) NOT NULL,
    color_id INTEGER,
    color_name VARCHAR(100),
    quantity INTEGER NOT NULL DEFAULT 1,
    confidence FLOAT NOT NULL DEFAULT 0.0,
    user_confirmed BOOLEAN NOT NULL DEFAULT FALSE,
    image_path TEXT,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    UNIQUE(part_num, color_id)
);
```

### ScanSession
```sql
CREATE TABLE scan_sessions (
    id UUID PRIMARY KEY,
    set_name VARCHAR(255) NOT NULL,
    completed BOOLEAN NOT NULL DEFAULT FALSE,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
);
```

## Configuration

Key settings (in `app/core/config.py` or environment):

```python
CONFIDENCE_THRESHOLD = 0.80  # >= 80% is "known"
ML_MODEL_PATH = "/app/models/lego_detector.onnx"
```

Local paths (auto-created):
```
~/brickscan_inventory.db        # SQLite database
~/brickscan_images/             # Scanned images (low-confidence only)
```

## Image Processing

Images are:
1. **Validated**: Base64 decoded, format checked, size <10MB
2. **Preprocessed**: Resized to 224×224, converted to CHW, normalized (ImageNet)
3. **Saved** (if uncertain): PNG to `~/brickscan_images/`
4. **Inferred**: ONNX model returns top-3 predictions with confidence scores

## Confidence Thresholds

- **≥ 80%** → "known" status
  - Model is confident
  - User can quick-add or review
  - Image NOT saved (reduces storage)

- **< 80%** → "uncertain" status
  - Model is uncertain
  - User shown top-3 candidates to pick from
  - Image saved for retraining
  - Marked `user_confirmed=false` until user selects

## Error Handling

Comprehensive error responses:

| Status | Scenario | Message |
|--------|----------|---------|
| 400 | Invalid base64 | "Invalid base64 encoding" |
| 400 | Image too large | "Image too large: X bytes (max 10MB)" |
| 400 | Corrupted image | "Invalid or corrupted image" |
| 404 | Part not found | "Inventory part not found" |
| 404 | Session not found | "Scan session not found" |
| 500 | Model not found | "Model inference failed" |
| 500 | Inference error | "Model inference failed" |

All errors include descriptive detail messages.

## Logging

Structured logging with logger name `app.local_inventory.*`:

```python
logger.info("Processing scan request")
logger.info(f"Prediction: {part_num} ({confidence:.1%}) - {status}")
logger.info(f"Saved uncertain scan image: {image_path}")
logger.info(f"Added to inventory: {part_num} qty={quantity}")
logger.error(f"Image validation failed: {error}")
```

## Testing

Run unit tests:

```bash
pytest app/local_inventory/test_routes.py -v
```

Test coverage includes:
- Image validation and preprocessing
- Inventory CRUD operations
- Confidence thresholds
- Session management
- Statistics calculation

Create test images:
```python
from PIL import Image
import base64
import io

img = Image.new("RGB", (224, 224), color="white")
buffer = io.BytesIO()
img.save(buffer, format="JPEG")
test_base64 = base64.b64encode(buffer.getvalue()).decode()
```

## Performance

- **Image processing**: ~100ms (resize + normalize in-memory)
- **ONNX inference**: ~200-500ms (depends on model size, CPU/GPU)
- **Database**: Indexed on part_num, color_id, created_at
- **Image storage**: Only for uncertain predictions (<80%), PNG format

## Security

- **No authentication**: Device-local, no API tokens
- **Input validation**: Base64 size, image format, part_num length
- **Local storage**: Images at `~/brickscan_images/`, user-owned
- **No encryption**: Not needed (local device, not sensitive)

Images can be manually deleted:
```bash
rm ~/brickscan_images/*.png
```

## Future Enhancements

1. **Cloud Sync**: Optional sync to backend PostgreSQL with user accounts
2. **Retraining**: Periodic retraining on user-corrected predictions
3. **Set Recognition**: OCR to identify set from box/photos
4. **Batch Scanning**: Camera stream mode for rapid scanning
5. **Part Search**: Search/filter by part number, color, or description
6. **Value Estimation**: BrickLink pricing integration
7. **Advanced Export**: LDraw, Stud.io, BrickLink formats
8. **Analytics**: Color distribution, most common parts, value totals

## Troubleshooting

### Database Issues

**"local_inventory database initialization: [error]"**
- Check `~/` directory is writable
- Verify Python has file permissions
- SQLite driver installed: `pip install sqlalchemy`

### Image Processing

**"Failed to process image: [error]"**
- Ensure image is valid JPEG or PNG
- Try a different image
- Check Pillow installed: `pip install pillow`

### Model Inference

**"Model inference failed"**
- ONNX model not found at configured path
- Check `ML_MODEL_PATH` in config
- ONNX runtime installed: `pip install onnxruntime`

### API Errors

**"Image too large: X bytes"**
- Reduce image resolution before scanning
- JPEG instead of PNG saves space

**"Inventory part not found"**
- Item may have been deleted
- Check inventory listing first

## Contributing

Adding features:
1. Update models in `models.py` if schema changes
2. Add Pydantic schema in `schemas.py`
3. Implement endpoint in `routes.py`
4. Add tests in `test_routes.py`
5. Update `DESIGN.md` with endpoint docs

## References

- **DESIGN.md** — Full technical design, API reference, architecture
- **CLIENT_USAGE.md** — Mobile/web integration examples (Python, React, Flutter)
- **test_routes.py** — Unit test examples
- **routes.py** — Endpoint implementation (commented)

## License

Proprietary - BrickScan Project

## Support

For issues or questions:
1. Check logs: `~/brickscan_inventory.db` and `~/brickscan_images/`
2. Review DESIGN.md for endpoint details
3. See CLIENT_USAGE.md for integration examples
4. Run tests: `pytest app/local_inventory/test_routes.py -v`
