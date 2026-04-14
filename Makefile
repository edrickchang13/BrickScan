.PHONY: help up down backend-shell db-shell migrate import-data verify-data test-backend test-mobile lint-backend install-mobile dev-mobile logs clean

help:
	@echo "BrickScan Development Commands"
	@echo "=============================="
	@echo ""
	@echo "Docker & Setup:"
	@echo "  make up                - Start all Docker services"
	@echo "  make down              - Stop all Docker services"
	@echo "  make logs              - Follow logs from all services"
	@echo ""
	@echo "Database:"
	@echo "  make migrate           - Run database migrations"
	@echo "  make db-shell          - Access PostgreSQL shell"
	@echo "  make import-data       - Import Rebrickable CSV data"
	@echo "  make verify-data       - Verify data import success"
	@echo ""
	@echo "Backend:"
	@echo "  make backend-shell     - Access backend container shell"
	@echo "  make test-backend      - Run backend tests"
	@echo "  make lint-backend      - Run linters (ruff, black, mypy)"
	@echo ""
	@echo "Mobile:"
	@echo "  make install-mobile    - Install mobile dependencies"
	@echo "  make test-mobile       - Run mobile tests"
	@echo "  make dev-mobile        - Start mobile dev server (iOS)"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean             - Remove all containers and volumes"
	@echo "  make help              - Show this help message"

up:
	@echo "Starting BrickScan services..."
	docker-compose up -d
	@echo ""
	@echo "Services started! Access:"
	@echo "  Backend API: http://localhost:8000"
	@echo "  API Docs:    http://localhost:8000/docs"
	@echo "  Adminer DB:  http://localhost:8080"
	@echo "  Redis:       localhost:6379"

down:
	@echo "Stopping BrickScan services..."
	docker-compose down

logs:
	@echo "Following logs from all services (Ctrl+C to exit)..."
	docker-compose logs -f

backend-shell:
	@echo "Entering backend container shell..."
	docker-compose exec backend bash

db-shell:
	@echo "Connecting to PostgreSQL database..."
	docker-compose exec db psql -U brickscan_user -d brickscan

migrate:
	@echo "Running database migrations..."
	docker-compose exec backend alembic upgrade head
	@echo "Migrations complete!"

import-data:
	@echo "Importing Rebrickable data..."
	@echo "Make sure you have downloaded Rebrickable CSVs to ./backend/data_pipeline/rebrickable_data"
	docker-compose exec backend python /app/data_pipeline/rebrickable_import.py /app/data_pipeline/rebrickable_data
	@echo "Import complete!"

verify-data:
	@echo "Verifying data import..."
	docker-compose exec backend python /app/data_pipeline/verify_import.py

test-backend:
	@echo "Running backend tests..."
	docker-compose exec backend pytest tests/ -v --tb=short
	@echo "Tests complete!"

test-mobile:
	@echo "Running mobile tests..."
	cd mobile && npm test

lint-backend:
	@echo "Running backend linters..."
	@echo ""
	@echo "Ruff check:"
	docker-compose exec backend ruff check app/ tests/
	@echo ""
	@echo "Black formatting check:"
	docker-compose exec backend black --check app/ tests/
	@echo ""
	@echo "Linting complete!"

install-mobile:
	@echo "Installing mobile dependencies..."
	cd mobile && npm install
	@echo "Dependencies installed!"

dev-mobile:
	@echo "Starting mobile dev server (iOS)..."
	@echo "Make sure you have Xcode simulator running"
	cd mobile && npx expo start --ios

clean:
	@echo "Removing all containers, volumes, and images..."
	docker-compose down -v
	docker system prune -f
	@echo "Clean complete!"

.DEFAULT_GOAL := help
