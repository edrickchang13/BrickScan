# Local Inventory System - File Index & Quick Reference

## Overview Files

| File | Purpose | Location |
|------|---------|----------|
| **LOCAL_INVENTORY_OVERVIEW.md** | Executive summary & checklist | backend/ |
| **LOCAL_INVENTORY_INDEX.md** | This file | backend/ |

## Module Files (`backend/app/local_inventory/`)

### Core Implementation
| File | Lines | Purpose |
|------|-------|---------|
| **routes.py** | 551 | 12 FastAPI endpoints (scan, inventory CRUD, sessions) |
| **models.py** | 107 | 2 SQLAlchemy ORM models (LocalInventoryPart, ScanSession) |
| **database.py** | 82 | SQLite setup, session management, auto-initialization |
| **image_processor.py** | 170 | Image validation, preprocessing, saving for retraining |
| **schemas.py** | 139 | 10 Pydantic request/response models with validation |
| **utils.py** | 159 | Helper functions, ConfidenceAnalysis class |
| **constants.py** | 24 | Configuration constants, thresholds, paths |
| **__init__.py** | 13 | Module docstring |

### Testing
| File | Lines | Purpose |
|------|-------|---------|
| **test_routes.py** | 304 | 20+ unit tests (image, CRUD, confidence, sessions) |

### Documentation
| File | Lines | Purpose | Audience |
|------|-------|---------|----------|
| **README.md** | 339 | Quick start, features, API summary, troubleshooting | Everyone |
| **DESIGN.md** | 429 | Technical reference, API specs, architecture | Backend/DevOps |
| **CLIENT_USAGE.md** | 420 | Integration examples (Python, React, Flutter) | Frontend/Mobile |
| **INTEGRATION_SUMMARY.md** | 403 | Deployment, architecture diagrams, checklist | DevOps/Architect |

## Quick Navigation

### "I want to..."

**...understand what was built**
→ Read: **LOCAL_INVENTORY_OVERVIEW.md** (5 min)

**...integrate this into my mobile app**
→ Read: **CLIENT_USAGE.md** (20 min)
- Python examples
- React Native examples
- Flutter examples
- Error handling patterns

**...see the API reference**
→ Read: **DESIGN.md** (30 min)
- All 12 endpoints documented
- Request/response examples
- Error codes and messages
- Architecture diagram

**...deploy or set up locally**
→ Read: **README.md** quick start (10 min)
→ Then: **INTEGRATION_SUMMARY.md** deployment section (10 min)

**...understand the code**
→ Start with: **routes.py** (main endpoints)
→ Then: **models.py** (data structures)
→ Then: **image_processor.py** (image handling)

**...run tests**
→ See: **test_routes.py** (copy test setup from fixtures)
→ Run: `pytest app/local_inventory/test_routes.py -v`

**...fix something**
→ Check: **README.md** troubleshooting section
→ Then: Appropriate module file (routes, database, image_processor)

## Endpoint Quick Reference

### POST /api/local-inventory/scan
**What**: Scan a LEGO brick image
**Input**: base64-encoded image
**Output**: Top-3 predictions + status ("known" or "uncertain")
**See**: DESIGN.md section "POST /api/local-inventory/scan"

### POST /api/local-inventory/inventory/add
**What**: Add a confirmed part to inventory
**Input**: part_num, color_id, color_name, quantity
**Output**: Updated inventory item
**See**: DESIGN.md section "POST /api/local-inventory/inventory/add"

### GET /api/local-inventory/inventory
**What**: List all parts in inventory
**Output**: Array of inventory items
**See**: DESIGN.md section "GET /api/local-inventory/inventory"

### GET /api/local-inventory/inventory/stats
**What**: Get aggregate statistics
**Output**: total_parts, total_quantity, confirmed, uncertain
**See**: routes.py line ~300

### PUT /api/local-inventory/inventory/{id}
**What**: Update part quantity
**Input**: new quantity
**Output**: Updated inventory item
**See**: DESIGN.md section "PUT /api/local-inventory/inventory/{id}"

### POST /api/local-inventory/inventory/{id}/correct
**What**: Fix a mislabeled part
**Input**: correct part_num, color_id, color_name
**Output**: Updated item with user_confirmed=true
**See**: routes.py line ~350

### DELETE /api/local-inventory/inventory/{id}
**What**: Remove a part from inventory
**Output**: Confirmation message
**See**: routes.py line ~400

### GET /api/local-inventory/inventory/export
**What**: Export inventory as CSV
**Output**: CSV text (part_num, color, quantity, confidence, etc.)
**See**: routes.py line ~420

### POST /api/local-inventory/scan-session/start
**What**: Start a named scanning session
**Input**: set_name (e.g., "Technic 42145")
**Output**: Session object with UUID
**See**: routes.py line ~450

### GET /api/local-inventory/scan-session
**What**: List all scan sessions
**Query**: completed_only (optional boolean)
**Output**: Array of sessions
**See**: routes.py line ~475

### POST /api/local-inventory/scan-session/{id}/complete
**What**: Mark a session as complete
**Output**: Updated session object
**See**: routes.py line ~495

### DELETE /api/local-inventory/scan-session/{id}
**What**: Delete a scan session
**Output**: Confirmation message
**See**: routes.py line ~515

## Database Schema Quick Reference

### local_inventory_parts table
```
id              UUID primary key
part_num        LEGO part number (indexed)
color_id        Rebrickable color ID (nullable, indexed)
color_name      Human-readable color name
quantity        Number of pieces
confidence      Model confidence (0.0-1.0)
user_confirmed  Boolean: user verified this?
image_path      Path to original scanned image (nullable)
created_at      Timestamp (UTC, indexed)
updated_at      Timestamp (UTC)
```

### scan_sessions table
```
id              UUID primary key
set_name        Session name (indexed)
completed       Boolean: is this done? (indexed)
created_at      Timestamp (UTC, indexed)
updated_at      Timestamp (UTC)
```

## Configuration & Paths

### Database
- **Location**: `~/brickscan_inventory.db`
- **Type**: SQLite
- **Auto-created**: Yes, on first startup

### Image Storage
- **Location**: `~/brickscan_images/`
- **Format**: PNG
- **Saved when**: Confidence < 80% (uncertain predictions)
- **Naming**: `{part_num}_{confidence}_{timestamp}.png`

### Configuration
- **Confidence threshold**: 0.80 (80%)
- **Image size limit**: 10 MB
- **Model input**: 224×224 pixels
- **Normalization**: ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Key Concepts

### Confidence Status
- **"known"** (≥80%): Model is confident → user can quick-add
- **"uncertain"** (<80%): Model uncertain → user picks from 3 options (image saved)

### User Confirmation
- **user_confirmed=true**: User has verified/corrected this entry
- **user_confirmed=false**: Uncertain prediction, not yet approved

### Confidence Score
- Stored with each part (0.0 to 1.0)
- For "known" predictions: Not saved (efficiency)
- For "uncertain" predictions: Saved for later analysis
- For manual additions: Set to 1.0 (user-confirmed)

## Error Codes

| Code | Scenario | Example |
|------|----------|---------|
| 400 | Invalid input | "Invalid base64 encoding" |
| 404 | Not found | "Inventory part not found" |
| 500 | Server error | "Model inference failed" |

See DESIGN.md section "Error Handling" for complete list.

## Testing Quick Start

```bash
# Run all tests
pytest app/local_inventory/test_routes.py -v

# Run specific test class
pytest app/local_inventory/test_routes.py::TestImageProcessing -v

# Run with coverage
pytest app/local_inventory/test_routes.py --cov=app.local_inventory
```

## Integration Points

- **main.py**: Router imported and registered (2 line changes)
- **app/core/database.py**: Not affected (separate SQLite DB)
- **app/services/ml_inference.py**: Used for ONNX inference
- **app/core/config.py**: Settings (ML_MODEL_PATH, CONFIDENCE_THRESHOLD)

## Production Checklist

- [x] Code implemented and tested
- [x] Documentation complete
- [x] Error handling comprehensive
- [x] Database auto-initialization
- [x] Logging configured
- [x] Type hints throughout
- [x] No external dependencies (all in requirements.txt)
- [x] Integration with main.py
- [x] Unit tests included
- [x] README & API docs

## Support

**Quick questions**: README.md
**API details**: DESIGN.md
**How to integrate**: CLIENT_USAGE.md
**Deployment**: INTEGRATION_SUMMARY.md
**Code details**: Module docstrings and comments

---

**Total Implementation**: 3,140 lines (code + tests + docs)
**Status**: ✅ Production-ready
**Location**: `/sessions/adoring-clever-goodall/mnt/Lego/brickscan/backend/app/local_inventory/`
