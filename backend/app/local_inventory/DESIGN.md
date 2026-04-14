# Local Inventory System Design

## Overview

The local inventory system enables users to scan their physical LEGO bricks with a mobile phone camera, build a device-local database of parts they own, and handle uncertain predictions with user confirmation.

Key design principles:
- **Offline-first**: All data stored locally on device (SQLite at `~/brickscan_inventory.db`)
- **No authentication**: Local device, no user accounts needed
- **Confidence-aware**: Distinguishes "known" predictions (>80%) from "uncertain" (<80%)
- **User-centric**: Allows users to confirm/correct uncertain predictions
- **Image persistence**: Saves low-confidence scans for later retraining

## Architecture

### Database Schema

#### LocalInventoryPart
Represents a single LEGO part in the user's inventory.

```
id                  UUID primary key
part_num            LEGO part number (e.g., "3001")
color_id            Rebrickable color ID (int, nullable)
color_name          Human-readable color (e.g., "White", "Bright Red")
quantity            Number of pieces owned
confidence          Model prediction confidence (0.0-1.0)
user_confirmed      True if user verified/corrected this entry
image_path          Path to original scanned image (for retraining)
created_at          Timestamp (UTC) when first added
updated_at          Timestamp (UTC) of last change
```

Unique constraint on (part_num, color_id) ensures no duplicates.

#### ScanSession
Groups multiple scans into a named session for organization.

```
id                  UUID primary key
set_name            Human-friendly name (e.g., "Technic 42145")
completed           Boolean: is this session finished?
created_at          Timestamp (UTC) when session started
updated_at          Timestamp (UTC) of last update
```

### Image Storage

Scanned images saved to `~/brickscan_images/`:
- Named: `{part_num}_{confidence}_{timestamp}.png`
- Saved for low-confidence predictions (<80%)
- Used for manual review and retraining

### Processing Pipeline

```
Mobile Camera Image (JPEG/PNG)
    ↓
[Validate & Decode] (base64 to bytes)
    ↓
[Preprocess] (224×224, normalize to [0,1], ImageNet norm)
    ↓
[ONNX Inference] (EfficientNet-B3, dual-head)
    ↓
[Top-3 Predictions] with confidence scores
    ↓
[Determine Status]
    ├─ confidence >= 80% → "known" (user can quick-add)
    └─ confidence < 80%  → "uncertain" (user selects from 3 options)
    ↓
[Save Image] (if uncertain, for retraining)
    ↓
[Return Response] with predictions + status
```

## API Endpoints

### POST /api/local-inventory/scan
Scan a LEGO brick image and predict part + color.

**Request:**
```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUg...",
  "session_id": "optional-uuid"
}
```

**Response (known):**
```json
{
  "status": "known",
  "predictions": [
    {
      "part_num": "3001",
      "part_name": "Brick 2x4",
      "confidence": 0.92,
      "color_id": 1,
      "color_name": "White",
      "color_hex": "F2F3F2"
    },
    ...
  ],
  "primary_prediction": { /* same as predictions[0] */ },
  "save_image": false
}
```

**Response (uncertain):**
```json
{
  "status": "uncertain",
  "predictions": [
    {
      "part_num": "3001",
      "confidence": 0.68,
      ...
    },
    {
      "part_num": "3002",
      "confidence": 0.15,
      ...
    },
    {
      "part_num": "3003",
      "confidence": 0.12,
      ...
    }
  ],
  "primary_prediction": { /* top candidate */ },
  "save_image": true
}
```

### POST /api/local-inventory/inventory/add
Add a confirmed part to inventory.

**Request:**
```json
{
  "part_num": "3001",
  "color_id": 1,
  "color_name": "White",
  "quantity": 5
}
```

**Response:**
```json
{
  "id": "uuid",
  "part_num": "3001",
  "color_id": 1,
  "color_name": "White",
  "quantity": 5,
  "confidence": 1.0,
  "user_confirmed": true,
  "image_path": null,
  "created_at": "2024-04-12T14:30:22.123Z",
  "updated_at": "2024-04-12T14:30:22.123Z"
}
```

### GET /api/local-inventory/inventory
List all parts in inventory.

**Response:**
```json
[
  {
    "id": "uuid",
    "part_num": "3001",
    "color_id": 1,
    "color_name": "White",
    "quantity": 5,
    "confidence": 1.0,
    "user_confirmed": true,
    ...
  },
  ...
]
```

### GET /api/local-inventory/inventory/stats
Get aggregate inventory statistics.

**Response:**
```json
{
  "total_parts": 42,
  "total_quantity": 850,
  "user_confirmed": 40,
  "uncertain_parts": 2
}
```

### PUT /api/local-inventory/inventory/{id}
Update part quantity.

**Request:**
```json
{
  "quantity": 10
}
```

**Response:** Updated LocalInventoryPartSchema

### POST /api/local-inventory/inventory/{id}/correct
Correct a mispredicted part.

**Request:**
```json
{
  "correct_part_num": "3002",
  "correct_color_id": 1,
  "correct_color_name": "White"
}
```

**Response:** Updated LocalInventoryPartSchema with user_confirmed=true

### DELETE /api/local-inventory/inventory/{id}
Remove a part from inventory.

**Response:**
```json
{
  "message": "Item deleted"
}
```

### GET /api/local-inventory/inventory/export
Export inventory as CSV.

**Response:** Plain text CSV with columns:
- Part Number
- Color
- Hex
- Quantity
- Confidence
- User Confirmed
- Created

### POST /api/local-inventory/scan-session/start
Start a new scanning session.

**Request:**
```json
{
  "set_name": "Technic 42145"
}
```

**Response:**
```json
{
  "id": "uuid",
  "set_name": "Technic 42145",
  "completed": false,
  "created_at": "2024-04-12T14:00:00Z",
  "updated_at": "2024-04-12T14:00:00Z"
}
```

### GET /api/local-inventory/scan-session
List all scan sessions.

**Query params:**
- `completed_only` (bool): Filter to completed sessions only

**Response:** List of ScanSessionSchema

### POST /api/local-inventory/scan-session/{id}/complete
Mark a session complete.

**Response:** Updated ScanSessionSchema

### DELETE /api/local-inventory/scan-session/{id}
Delete a scan session.

**Response:**
```json
{
  "message": "Session deleted"
}
```

## Confidence Handling

### "Known" Predictions (≥80%)
- Model is confident about the part
- User sees primary prediction
- Can quick-add with one tap
- May still see top-3 for comparison
- Image not saved (reduces storage)

### "Uncertain" Predictions (<80%)
- Model has low confidence
- User shown top-3 predictions
- Must pick the correct part
- Image saved to `~/brickscan_images/`
- Marked as `user_confirmed=false` until user selects

### User Workflow
1. Scan brick → Get prediction
2. If "known" (≥80%): Quick-add or review top-3
3. If "uncertain" (<80%): Pick correct part from 3 options
4. If all wrong: User can manually enter part_num
5. Uncertain scans saved for later retraining

## File Organization

```
backend/app/local_inventory/
├── __init__.py           # Module docstring
├── models.py             # SQLAlchemy ORM models
├── database.py           # SQLite setup, session mgmt
├── schemas.py            # Pydantic request/response models
├── image_processor.py    # Image validation, preprocessing, saving
├── routes.py             # FastAPI endpoints
├── constants.py          # Config constants
├── utils.py              # Helper functions
└── DESIGN.md            # This file
```

### External Storage
```
~/brickscan_inventory.db       # SQLite database
~/brickscan_images/            # Scanned images (for retraining)
    ├── 3001_0_92_20240412_143022.png
    ├── 3002_0_68_20240412_143145.png
    └── ...
```

## Error Handling

### Image Validation
- Base64 decode error → 400 "Invalid base64 encoding"
- Image too large (>10MB) → 400 "Image too large"
- Corrupted image → 400 "Invalid or corrupted image"
- Unsupported format → 400 "Unsupported image format"

### ML Inference
- Model not found → Returns empty predictions, logs warning
- Inference error → 500 "Model inference failed"
- No predictions returned → 500 "Model returned no predictions"

### Database
- Part not found in inventory → 404 "Inventory part not found"
- Session not found → 404 "Scan session not found"
- Database error → 500 with generic message

## Performance Considerations

### Image Preprocessing
- LANCZOS resampling: High quality
- ImageNet normalization: Matches training pipeline
- In-memory only (no disk writes except for uncertain predictions)

### Database
- Indexed on: part_num, color_id, created_at, updated_at
- Unique constraint on (part_num, color_id)
- Small local DB, no concurrency issues

### Image Storage
- Only save for uncertain predictions (<80% confidence)
- Named with timestamp to avoid collisions
- Can be manually purged from `~/brickscan_images/`

## Security

### No Authentication
- Local device only
- No user accounts
- No API keys or tokens
- Data isolation by filesystem permissions

### Input Validation
- Base64 image size limit: 10 MB
- Part number: Alphanumeric, max 50 chars
- Color ID: Integer validation
- Session name: String, max 255 chars

### Image Storage
- Saved to user's home directory
- Original format preserved (PNG)
- No encryption (local device, not sensitive)
- Can be deleted manually

## Testing

Example request with curl:
```bash
# Scan an image (base64 encoded)
curl -X POST http://localhost:8000/api/local-inventory/scan \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "iVBORw0KGgoAAAANSUhEUg...",
    "session_id": null
  }'

# Add to inventory
curl -X POST http://localhost:8000/api/local-inventory/inventory/add \
  -H "Content-Type: application/json" \
  -d '{
    "part_num": "3001",
    "color_id": 1,
    "color_name": "White",
    "quantity": 5
  }'

# Get inventory
curl http://localhost:8000/api/local-inventory/inventory

# Export CSV
curl http://localhost:8000/api/local-inventory/inventory/export > inventory.csv
```

## Future Enhancements

1. **Sync to Cloud**: Optional sync to backend PostgreSQL (with user accounts)
2. **Retraining**: Periodic retraining with user-corrected predictions
3. **Set Recognition**: OCR to identify set from box/manual entry
4. **Batch Scanning**: Camera stream for rapid scanning
5. **Part Search**: Search/filter by part number or color
6. **Value Estimation**: BrickLink pricing integration
7. **Export Formats**: BrickLink CSV, LDraw, Stud.io format
8. **Analytics**: Brick color distribution, most common parts, etc.
