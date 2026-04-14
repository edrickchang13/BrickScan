# BrickScan - LEGO Piece Scanning iOS App

BrickScan is a comprehensive iOS application that helps LEGO enthusiasts scan and manage their brick collections. Using advanced computer vision and machine learning, users can quickly identify LEGO pieces by scanning them with their phone camera, then manage a complete inventory of their collection.

## Features

- **Smart Piece Scanning**: Real-time LEGO brick detection using on-device ML models
- **Inventory Management**: Track your complete LEGO collection with quantities and colors
- **Build Planning**: Check if you have all the pieces for any LEGO set
- **BrickLink Integration**: Export missing pieces to BrickLink for purchasing
- **Offline Support**: Scan and manage inventory without constant internet connection
- **Cloud Sync**: Optional cloud backup and sync across devices

## Tech Stack

### Backend
- **Framework**: FastAPI (Python 3.11+)
- **Database**: PostgreSQL 16
- **Cache**: Redis 7
- **Migrations**: Alembic
- **API**: RESTful with async/await support
- **Authentication**: JWT tokens

### Mobile
- **Framework**: React Native with Expo
- **Language**: TypeScript
- **ML**: TensorFlow Lite (on-device inference)
- **State Management**: Redux Toolkit
- **Testing**: Jest + React Native Testing Library

### Infrastructure
- **Containerization**: Docker & Docker Compose
- **CI/CD**: GitHub Actions
- **Database**: PostgreSQL with pgAdmin (Adminer)
- **Cache**: Redis for caching and session management

## Prerequisites

### System Requirements
- **Node.js**: 20.x LTS
- **Python**: 3.11+
- **Docker**: Latest version with Docker Compose
- **Xcode**: 15+ (for iOS development)
- **Expo CLI**: `npm install -g expo-cli`

### API Keys & Credentials
- **Rebrickable API**: Free account at https://rebrickable.com
- **BrickLink API**: Account at https://www.bricklink.com
- **Google Gemini API**: For advanced image recognition (optional)

## Quick Start

### 1. Clone Repository
```bash
git clone <repository-url>
cd brickscan
```

### 2. Setup Environment
```bash
# Copy environment template and fill in your API keys
cp backend/.env.example backend/.env

# Edit backend/.env with:
# - REBRICKABLE_API_KEY=your_key
# - BRICKLINK_API_KEY=your_key
# - GEMINI_API_KEY=your_key (optional)
# - JWT_SECRET_KEY=your_secret (generate with: openssl rand -hex 32)
```

### 3. Start Docker Services
```bash
make up
```

This starts:
- PostgreSQL database (port 5432)
- Redis cache (port 6379)
- FastAPI backend (port 8000)
- Adminer DB UI (port 8080)

### 4. Initialize Database
```bash
# Run migrations
make migrate
```

### 5. Import Rebrickable Data
```bash
# Download Rebrickable CSV dumps
cd data_pipeline
./download_rebrickable.sh ./rebrickable_data
cd ..

# Import into database
make import-data

# Verify import success
make verify-data
```

### 6. Start Mobile Development
```bash
# Install dependencies
make install-mobile

# Start iOS dev server (requires Xcode simulator)
make dev-mobile
```

## API Documentation

Once the backend is running, interactive API documentation is available:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

#### Authentication
- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - Login and get JWT token
- `GET /api/v1/auth/me` - Get current user profile
- `POST /api/v1/auth/refresh` - Refresh access token

#### Inventory Management
- `GET /api/v1/inventory/items` - List user's inventory
- `POST /api/v1/inventory/items` - Add piece to inventory
- `PATCH /api/v1/inventory/items/{id}` - Update quantity
- `DELETE /api/v1/inventory/items/{id}` - Remove piece

#### Build Checking
- `GET /api/v1/builds/{set_id}/check` - Check if user has all pieces for a set
- `GET /api/v1/builds/{set_id}/missing` - Get list of missing pieces
- `GET /api/v1/builds/{set_id}/bricklink-xml` - Export missing pieces as BrickLink XML

#### LEGO Data
- `GET /api/v1/parts` - Search parts database
- `GET /api/v1/sets` - Search LEGO sets
- `GET /api/v1/colors` - List available colors
- `GET /api/v1/themes` - List LEGO themes

## Development Workflow

### Running Tests

```bash
# Backend tests
make test-backend

# Mobile tests
make test-mobile
```

### Code Quality

```bash
# Run linters and formatters
make lint-backend

# Format code
docker-compose exec backend black app/ tests/
```

### Database Management

```bash
# Access PostgreSQL directly
make db-shell

# View database with web UI
# Visit http://localhost:8080
# Server: db
# User: brickscan_user
# Password: brickscan_password
# Database: brickscan

# Create new migration
docker-compose exec backend alembic revision -m "description"

# Rollback migrations
docker-compose exec backend alembic downgrade -1
```

### Backend Development

```bash
# Enter backend shell
make backend-shell

# Run single test file
pytest tests/test_auth.py -v

# Debug with print statements
pytest tests/test_auth.py -v -s
```

### Mobile Development

```bash
# Install dependencies
make install-mobile

# Start development server
make dev-mobile

# Run type checking
cd mobile && npx tsc --noEmit

# Format code
cd mobile && npx prettier --write src/
```

## Project Structure

```
brickscan/
├── backend/                      # FastAPI backend
│   ├── app/
│   │   ├── main.py             # FastAPI app initialization
│   │   ├── database.py          # Database configuration
│   │   ├── models/              # SQLAlchemy ORM models
│   │   ├── schemas/             # Pydantic request/response schemas
│   │   ├── api/                 # API route handlers
│   │   ├── services/            # Business logic
│   │   ├── core/                # Core utilities (auth, config, etc)
│   │   └── ml/                  # ML model loading & inference
│   ├── alembic/                 # Database migrations
│   ├── tests/                   # Backend unit & integration tests
│   ├── Dockerfile
│   ├── requirements.txt         # Python dependencies
│   ├── requirements-dev.txt     # Development dependencies
│   └── alembic.ini
│
├── mobile/                       # React Native/Expo app
│   ├── src/
│   │   ├── screens/             # Screen components
│   │   ├── components/          # Reusable components
│   │   ├── services/            # API clients & utilities
│   │   ├── store/               # Redux state management
│   │   ├── models/              # ML inference
│   │   └── App.tsx
│   ├── package.json
│   ├── tsconfig.json
│   └── app.json
│
├── data_pipeline/               # Data import utilities
│   ├── rebrickable_import.py   # CSV import script
│   ├── verify_import.py         # Data validation
│   └── download_rebrickable.sh  # Download CSV dumps
│
├── .github/
│   └── workflows/               # CI/CD workflows
│       ├── backend_ci.yml
│       └── mobile_ci.yml
│
├── docker-compose.yml           # Local development stack
├── Makefile                     # Development commands
├── .gitignore
└── README.md
```

## Database Schema

### Core Tables

**users** - User accounts and authentication
- id (UUID)
- email (unique)
- hashed_password
- full_name
- is_active
- created_at, updated_at

**colors** - LEGO color definitions
- id (from Rebrickable)
- name
- rgb (hex color code)
- is_transparent

**part_categories** - Part type categories (Brick, Plate, etc)
- id (from Rebrickable)
- name

**parts** - LEGO piece definitions
- id (UUID)
- part_num (unique, from Rebrickable)
- name
- part_category_id
- material
- image_url

**themes** - LEGO themes (Star Wars, City, etc)
- id (from Rebrickable)
- name
- parent_id (for theme hierarchy)

**lego_sets** - LEGO set definitions
- id (UUID)
- set_num (unique, from Rebrickable)
- name
- year
- theme_id
- num_parts
- image_url

**set_parts** - Junction table mapping sets to their pieces
- id (UUID)
- set_id
- part_id
- color_id
- quantity
- is_spare

**inventory_items** - User's collected pieces
- id (UUID)
- user_id
- part_id
- color_id
- quantity
- Unique constraint: (user_id, part_id, color_id)

**scan_logs** - Record of piece scans for training
- id (UUID)
- user_id
- part_id (nullable)
- color_id (nullable)
- quantity
- confidence (ML model confidence)
- image_path
- status (success, unknown, error)
- error_message

## Getting API Keys

### Rebrickable
1. Visit https://rebrickable.com/api/
2. Create free account
3. Generate API key on account page
4. Add to `backend/.env`: `REBRICKABLE_API_KEY=your_key`

### BrickLink
1. Visit https://www.bricklink.com
2. Go to Account > Preferences > API > Registration
3. Create new API key
4. Add to `backend/.env`: `BRICKLINK_API_KEY=your_key`

### Google Gemini (Optional)
1. Visit https://ai.google.dev
2. Create project and enable Gemini API
3. Generate API key
4. Add to `backend/.env`: `GEMINI_API_KEY=your_key`

## ML Model Training

The app includes on-device ML models for piece classification. To update models:

```bash
# In future: train pipeline instructions
# For now, pre-trained TensorFlow Lite models are included in mobile/models/
```

## Common Commands Reference

```bash
# Start development environment
make up

# Stop services
make down

# View logs
make logs

# Run migrations
make migrate

# Import Rebrickable data
make import-data

# Verify data import
make verify-data

# Run backend tests
make test-backend

# Run mobile tests
make test-mobile

# Start mobile dev server
make dev-mobile

# Run linters
make lint-backend

# Access database shell
make db-shell

# Access backend shell
make backend-shell

# Clean up everything
make clean
```

## Contributing

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes and write tests
3. Run `make lint-backend` and `make test-backend`
4. Commit with clear messages
5. Push to branch and create Pull Request

## CI/CD

GitHub Actions automatically runs:
- Python linting (ruff, black)
- Python type checking (mypy)
- Backend pytest suite with PostgreSQL test database
- TypeScript type checking
- Jest mobile tests
- Code coverage reports

## Troubleshooting

### Docker Issues
```bash
# Rebuild containers
docker-compose down && docker-compose up -d --build

# Check service health
docker-compose ps

# View service logs
docker-compose logs backend
docker-compose logs db
```

### Database Issues
```bash
# Reset database (destroys data!)
docker-compose down -v

# Connect to database
make db-shell

# Check migration status
docker-compose exec backend alembic current
```

### Mobile Development
```bash
# Clear Expo cache
cd mobile && npx expo start --clear

# Clear npm cache
npm cache clean --force

# Reinstall dependencies
rm -rf node_modules package-lock.json && npm install
```

## Performance Tips

- Use Redis caching for frequently accessed LEGO data
- Index inventory queries by user_id for fast retrieval
- Lazy load images with progressive JPEG
- Batch ML inference on multiple scans
- Use database connection pooling

## License

This project is licensed under the MIT License.

## Support

For issues, feature requests, or questions:
- Open GitHub Issue
- Contact: support@brickscan.dev

---

Built with passion for LEGO enthusiasts everywhere!
