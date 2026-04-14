# BrickScan Backend API

A complete FastAPI-based backend for BrickScan - a LEGO piece scanning iOS app. The API handles user authentication, LEGO part recognition, inventory management, set building comparisons, and BrickLink integration.

## Features

- **User Authentication**: JWT-based auth with bcrypt password hashing
- **LEGO Part Recognition**: ML model inference with Gemini Vision API fallback
- **Inventory Management**: Track personal LEGO piece collections
- **Set Building**: Compare inventory against LEGO sets and identify missing pieces
- **BrickLink Integration**: Generate XML wanted lists for BrickLink marketplace
- **Redis Caching**: High-performance caching for API responses
- **Async Database**: Full async/await with SQLAlchemy 2.0 and PostgreSQL

## Tech Stack

- **Framework**: FastAPI with async support
- **Database**: PostgreSQL with asyncpg driver
- **ORM**: SQLAlchemy 2.0 (async)
- **Auth**: python-jose with JWT and bcrypt
- **Caching**: Redis with async client
- **ML**: ONNX Runtime (optional) + Gemini Vision API
- **Image Processing**: Pillow
- **HTTP Client**: httpx for async requests

## Project Structure

```
app/
├── core/
│   ├── config.py         # Settings from environment
│   ├── database.py       # SQLAlchemy async setup
│   └── security.py       # JWT and password utilities
├── models/
│   ├── user.py           # User model
│   ├── part.py           # Part, Color, PartCategory models
│   ├── lego_set.py       # LegoSet, Theme, SetPart models
│   └── inventory.py      # InventoryItem, ScanLog models
├── schemas/
│   ├── auth.py           # Auth request/response schemas
│   ├── part.py           # Part-related schemas
│   ├── lego_set.py       # Set-related schemas
│   ├── inventory.py      # Inventory schemas
│   └── scan.py           # Scan request/response schemas
├── api/
│   ├── auth.py           # /auth endpoints
│   ├── parts.py          # /parts endpoints
│   ├── sets.py           # /sets endpoints
│   ├── inventory.py      # /inventory endpoints
│   ├── scan.py           # /scan endpoints
│   └── bricklink.py      # /bricklink endpoints
└── services/
    ├── rebrickable.py    # Rebrickable API client
    ├── bricklink_service.py   # BrickLink utilities
    ├── ml_inference.py   # ML model inference
    ├── gemini_service.py # Gemini Vision API
    └── build_check.py    # Set comparison logic

main.py                    # FastAPI app entry point
requirements.txt           # Python dependencies
.env.example               # Environment variables template
Dockerfile                 # Docker build config
```

## Setup

### 1. Prerequisites

- Python 3.11+
- PostgreSQL 13+
- Redis 6.0+
- Docker & Docker Compose (optional)

### 2. Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env` with your values:
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `SECRET_KEY`: Secure random key for JWT signing
- `REBRICKABLE_API_KEY`: API key from rebrickable.com
- `GEMINI_API_KEY`: Google Gemini API key
- `BRICKLINK_*`: BrickLink OAuth credentials
- `AWS_*`: S3 bucket for image storage

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run with Docker Compose

```bash
docker-compose up -d
```

This starts PostgreSQL, Redis, and the FastAPI app.

### 5. Run Locally

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Authentication

```
POST   /auth/register      # Create new user account
POST   /auth/login         # Login and get JWT token
GET    /auth/me            # Get current user (protected)
```

### LEGO Parts

```
GET    /parts?search=&limit=20&offset=0    # Search parts with pagination
GET    /parts/{part_num}                    # Get part details
```

### LEGO Sets

```
GET    /sets?search=&theme=&year=           # Search sets
GET    /sets/{set_num}                      # Get set details
GET    /sets/{set_num}/parts                # Get all parts in set
```

### User Inventory (Protected)

```
GET    /inventory                           # Get user's inventory
POST   /inventory                           # Add part to inventory
PUT    /inventory/{item_id}                 # Update quantity
DELETE /inventory/{item_id}                 # Remove item
GET    /inventory/export                    # Export as CSV
```

### LEGO Scanning

```
POST   /scan                                # Identify piece from image
POST   /scan/confirm                        # Confirm scan result
```

### BrickLink Integration

```
POST   /bricklink/wanted-list/{set_num}    # Generate wanted list XML
GET    /bricklink/colors                    # Get color mappings
```

### Health

```
GET    /health                              # Health check
```

## Database Models

### Users
- `id` (UUID): Primary key
- `email` (String, unique): User email
- `hashed_password` (String): Bcrypt hash
- `is_active` (Boolean): Account status
- `created_at`, `updated_at` (DateTime): Timestamps

### Parts
- `id` (UUID): Primary key
- `part_num` (String, unique): BrickLink part number
- `name` (String): Official LEGO name
- `part_cat_id` (FK): Category reference
- `year_from`, `year_to` (Integer): Production years
- `image_url` (String): Picture URL

### Colors
- `id` (UUID): Primary key
- `rebrickable_id` (Integer, unique): Rebrickable color ID
- `name` (String): Official LEGO color name
- `hex_code` (String): RGB hex code
- `is_transparent` (Boolean): Transparency flag

### LEGO Sets
- `id` (UUID): Primary key
- `set_num` (String, unique): Official set number
- `name` (String): Set name
- `year` (Integer): Release year
- `theme_id` (FK): Theme reference
- `num_parts` (Integer): Part count
- `img_url` (String): Set image URL

### SetParts
- Associates parts with sets
- Tracks quantities and spare parts
- Unique constraint on (set_id, part_id, color_id)

### InventoryItems
- User's collected pieces
- Unique constraint on (user_id, part_id, color_id)
- Tracks quantity and timestamps

### ScanLogs
- Records all ML/Gemini identifications
- Tracks predicted and confirmed part numbers
- Stores confidence scores

## Authentication

The API uses JWT (JSON Web Tokens) with Bearer scheme:

```bash
curl -H "Authorization: Bearer <token>" http://localhost:8000/auth/me
```

Tokens expire after 30 minutes by default (configurable via `ACCESS_TOKEN_EXPIRE_MINUTES`).

## Caching Strategy

- **Parts Search**: 1 hour TTL
- **Set Data**: 24 hours TTL
- **Part Details**: 24 hours TTL
- All caches stored in Redis with keys like `parts:search:{query}`

## ML Model Integration

### Local Model (Optional)

Place an ONNX model at `ML_MODEL_PATH`. The service:
1. Loads the model on startup
2. Preprocesses images (resize to 224x224, normalize)
3. Returns top 3 predictions with confidence scores
4. Falls back to Gemini if confidence < threshold

### Gemini Vision API (Fallback)

If no local model or confidence is low:
1. Sends base64-encoded image to Gemini 2.0 Flash
2. Gets JSON response with part predictions
3. Cross-references against database
4. Returns top 3 results with confidence

## BrickLink Integration

The `/bricklink/wanted-list/{set_num}` endpoint:
1. Gets user's inventory for the set
2. Identifies missing pieces
3. Maps Rebrickable IDs to BrickLink equivalents
4. Generates BrickLink-compatible XML format
5. Returns XML string ready to import

Example XML output:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<INVENTORY>
    <ITEM>
        <ITEMID>3001</ITEMID>
        <ITEMTYPE>P</ITEMTYPE>
        <COLOR>11</COLOR>
        <MINQTY>5</MINQTY>
        <CONDITION>N</CONDITION>
    </ITEM>
</INVENTORY>
```

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200 OK`: Successful request
- `201 Created`: Resource created
- `204 No Content`: Successful deletion
- `400 Bad Request`: Invalid input
- `401 Unauthorized`: Missing/invalid auth
- `404 Not Found`: Resource not found
- `409 Conflict`: Duplicate email, etc.
- `500 Internal Server Error`: Server error

Error responses include descriptive messages:

```json
{
  "detail": "Invalid email or password"
}
```

## Development

### Run Tests

```bash
pytest
```

### Format Code

```bash
black app/ main.py
```

### Lint

```bash
flake8 app/ main.py
```

### Type Check

```bash
mypy app/ main.py
```

## Deployment

### Docker Build

```bash
docker build -t brickscan-api:latest .
```

### Environment in Production

Set these environment variables in your production environment:
- Change `SECRET_KEY` to a strong random value
- Use secure PostgreSQL with SSL
- Use Redis with authentication
- Set proper CORS origins
- Enable HTTPS

## Performance Considerations

1. **Database Indexing**: All commonly searched columns are indexed
2. **Connection Pooling**: SQLAlchemy with pool_size=20, max_overflow=10
3. **Async/Await**: All I/O operations are non-blocking
4. **Redis Caching**: Frequently accessed data cached with TTL
5. **Pagination**: List endpoints support offset/limit
6. **Lazy Loading**: Related objects loaded on demand

## Future Enhancements

- [ ] WebSocket support for real-time inventory updates
- [ ] Batch import from BrickLink/CSV
- [ ] Advanced set comparison with price estimates
- [ ] Image upload directly (not base64)
- [ ] Duplicate detection in inventory
- [ ] Building progress notifications
- [ ] Social features (collection sharing)
- [ ] Mobile push notifications

## License

Proprietary - BrickScan Team

## Support

For issues or feature requests, contact the development team.
