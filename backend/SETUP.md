# BrickScan Backend Setup Guide

## Quick Start with Docker Compose

The easiest way to get the API running locally is with Docker Compose:

```bash
# 1. Clone/navigate to the project
cd /sessions/adoring-clever-goodall/mnt/Lego/brickscan/backend

# 2. Create .env file from template
cp .env.example .env

# 3. Update .env with your API keys (minimum: GEMINI_API_KEY, REBRICKABLE_API_KEY)
# Edit .env file...

# 4. Start all services
docker-compose up -d

# 5. Check logs
docker-compose logs -f api
```

The API will be available at `http://localhost:8000`

## Manual Setup (Local Development)

### Prerequisites

- Python 3.11+
- PostgreSQL 13+ (or use Docker: `docker run -d -e POSTGRES_PASSWORD=password postgres:16-alpine`)
- Redis 6.0+ (or use Docker: `docker run -d -p 6379:6379 redis:7-alpine`)
- Git

### Installation Steps

```bash
# 1. Clone repository
git clone <repo-url>
cd backend

# 2. Create Python virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file
cp .env.example .env

# 5. Configure environment variables
# Edit .env with your values - at minimum set:
# - DATABASE_URL (PostgreSQL connection)
# - REDIS_URL (Redis connection)
# - SECRET_KEY (generate: python -c "import secrets; print(secrets.token_urlsafe(32))")
# - GEMINI_API_KEY
# - REBRICKABLE_API_KEY

# 6. Start the API server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## Environment Variables Explained

### Database & Cache
- `DATABASE_URL`: PostgreSQL connection string
  - Format: `postgresql+asyncpg://user:password@host:port/dbname`
  - Example: `postgresql+asyncpg://brickscan:password@localhost:5432/brickscan`

- `REDIS_URL`: Redis connection string
  - Format: `redis://host:port/db`
  - Example: `redis://localhost:6379/0`

### Authentication
- `SECRET_KEY`: JWT signing key (must be kept secret in production)
  - Generate: `python -c "import secrets; print(secrets.token_urlsafe(32))"`

- `ALGORITHM`: JWT algorithm (default: HS256)

- `ACCESS_TOKEN_EXPIRE_MINUTES`: Token expiration time (default: 30)

### External APIs
- `REBRICKABLE_API_KEY`: Get from https://rebrickable.com/api
  - Required for part and set data

- `GEMINI_API_KEY`: Get from Google AI Studio
  - Required for vision-based piece identification
  - Get it from: https://aistudio.google.com

- `BRICKLINK_*`: BrickLink OAuth credentials
  - Only needed if integrating with BrickLink marketplace
  - Get from: https://www.bricklink.com/v2/api/register_consumer.page

### ML & Storage
- `ML_MODEL_PATH`: Path to ONNX model file (optional)
  - Default: `/app/models/lego_detector.onnx`
  - Leave as-is if you don't have a trained model

- `CONFIDENCE_THRESHOLD`: Minimum confidence for ML predictions
  - Default: 0.75 (75%)
  - If below this, falls back to Gemini API

- `S3_BUCKET`: AWS S3 bucket for image storage
  - Required for production image uploads

- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`: AWS credentials
  - Only needed if using S3 storage

## API Testing

### Using curl

```bash
# Register user
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password123"}'

# Login and get token
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password123"}'

# Use token in subsequent requests
TOKEN="your_token_here"
curl -X GET http://localhost:8000/auth/me \
  -H "Authorization: Bearer $TOKEN"

# Health check
curl http://localhost:8000/health
```

### Using Python

```python
import httpx
import json

BASE_URL = "http://localhost:8000"

# Register
response = httpx.post(
    f"{BASE_URL}/auth/register",
    json={"email": "test@example.com", "password": "password123"}
)
token = response.json()["access_token"]

# Get current user
response = httpx.get(
    f"{BASE_URL}/auth/me",
    headers={"Authorization": f"Bearer {token}"}
)
print(response.json())

# Search parts
response = httpx.get(
    f"{BASE_URL}/parts?search=brick&limit=10"
)
print(response.json())
```

## Database Schema

The application automatically creates all tables on first run. Tables include:

- `users`: User accounts
- `colors`: LEGO color definitions
- `part_categories`: Part categories
- `parts`: LEGO parts catalog
- `themes`: LEGO themes/categories
- `lego_sets`: LEGO sets
- `set_parts`: Parts in sets
- `inventory_items`: User's collection
- `scan_logs`: ML/vision identification history

## Common Issues

### Issue: "connection refused" on database
**Solution**: Ensure PostgreSQL is running:
```bash
# Using Docker
docker run -d -p 5432:5432 \
  -e POSTGRES_USER=brickscan_user \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=brickscan \
  postgres:16-alpine

# Update DATABASE_URL in .env
DATABASE_URL=postgresql+asyncpg://brickscan_user:password@localhost:5432/brickscan
```

### Issue: "redis connection refused"
**Solution**: Start Redis:
```bash
# Using Docker
docker run -d -p 6379:6379 redis:7-alpine

# Update REDIS_URL in .env
REDIS_URL=redis://localhost:6379/0
```

### Issue: "GEMINI_API_KEY not set"
**Solution**: Add your Gemini API key to .env:
```bash
GEMINI_API_KEY=your_key_here
```

### Issue: "ModuleNotFoundError: No module named 'sqlalchemy'"
**Solution**: Reinstall dependencies:
```bash
pip install -r requirements.txt
```

## Development Workflow

### Running with auto-reload
```bash
uvicorn main:app --reload
```

### Accessing API documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Debugging
Set environment variable for more verbose logging:
```bash
export SQLALCHEMY_ECHO=true
```

### Running specific module tests
```bash
pytest app/tests/test_auth.py -v
```

## Deployment Checklist

Before deploying to production:

- [ ] Change `SECRET_KEY` to a strong random value
- [ ] Set `DEBUG = false` (if applicable)
- [ ] Use strong database password
- [ ] Enable Redis authentication
- [ ] Configure proper CORS origins
- [ ] Use HTTPS/TLS
- [ ] Set up proper logging
- [ ] Configure monitoring/alerting
- [ ] Test database backups
- [ ] Set up rate limiting
- [ ] Enable API key rotation for external services
- [ ] Configure environment-specific settings

## Docker Deployment

### Build image
```bash
docker build -t brickscan-api:latest .
```

### Run container
```bash
docker run -d \
  -e DATABASE_URL="postgresql+asyncpg://..." \
  -e REDIS_URL="redis://..." \
  -e SECRET_KEY="..." \
  -e GEMINI_API_KEY="..." \
  -p 8000:8000 \
  brickscan-api:latest
```

### Using with docker-compose for production
See `docker-compose.yml` - update the image reference and remove `--reload` flag.

## Performance Optimization

### For production deployment:

1. **Database**: Enable connection pooling (already configured)
2. **Redis**: Consider using Redis Cluster
3. **API**: Deploy multiple instances with load balancer
4. **Caching**: Increase TTL for stable data
5. **Logging**: Use structured logging

## Getting Help

### Check logs
```bash
# Docker Compose
docker-compose logs api

# Local
# Check terminal output where you ran uvicorn
```

### API Documentation
- Swagger UI: http://localhost:8000/docs
- Interactive API testing available there

### Common API responses
- `200`: Success
- `400`: Bad request (invalid input)
- `401`: Unauthorized (missing/invalid token)
- `404`: Not found
- `500`: Server error (check logs)

## Next Steps

1. Set up your LEGO parts database (import from Rebrickable)
2. Upload your local inventory
3. Integrate with iOS app
4. Test piece scanning with Gemini Vision API
5. Configure BrickLink integration if needed

For more information, see README.md
