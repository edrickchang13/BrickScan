#!/bin/bash
# Start BrickScan Vision Server on DGX Spark
#
# This script starts the FastAPI vision inference server that provides
# local LEGO piece identification via Ollama models.
#
# Usage:
#   ./start_server.sh                    # Default on port 8001
#   PORT=9000 ./start_server.sh         # Custom port
#
# The server will be available at: http://<dgx-ip>:8001

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
PORT="${PORT:-8001}"
HOST="${HOST:-0.0.0.0}"
WORKERS="${WORKERS:-1}"
VENV_PATH="${VENV_PATH:-$HOME/brickscan-env}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "=========================================="
echo "BrickScan Vision Server"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo -e "${RED}ERROR: Virtual environment not found at $VENV_PATH${NC}"
    echo "Run setup/install_dependencies.sh first"
    exit 1
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Check if uvicorn is installed
if ! python -c "import uvicorn" 2>/dev/null; then
    echo -e "${RED}ERROR: uvicorn not installed in virtual environment${NC}"
    echo "Run: pip install fastapi uvicorn[standard]"
    exit 1
fi

# Check if Ollama is running
echo "Checking Ollama status..."
if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
    echo -e "${YELLOW}WARNING: Ollama doesn't appear to be running${NC}"
    echo "Start Ollama with: sudo systemctl start ollama"
    echo "You can continue, but the server will fail when you try to identify pieces."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for available models
echo "Checking available models..."
MODELS=$(curl -s http://localhost:11434/api/tags 2>/dev/null | grep -o '"name":"[^"]*"' | cut -d'"' -f4 | head -5)
if [ -z "$MODELS" ]; then
    echo -e "${YELLOW}WARNING: No Ollama models found${NC}"
    echo "Pull models with:"
    echo "  ollama pull llava:13b"
    echo "  ollama pull moondream"
else
    echo "Found models: $(echo $MODELS | tr '\n' ', ')"
fi
echo ""

# Get local IP
LOCAL_IP=$(hostname -I | awk '{print $1}')
if [ -z "$LOCAL_IP" ]; then
    LOCAL_IP="localhost"
fi

# Display startup info
echo -e "${GREEN}Starting Vision Server${NC}"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  URL: http://$LOCAL_IP:$PORT"
echo "  API Docs: http://$LOCAL_IP:$PORT/docs"
echo "  Workers: $WORKERS"
echo ""
echo "Check health: curl http://$LOCAL_IP:$PORT/health"
echo ""
echo "To use from backend, set:"
echo "  DGX_VISION_URL=http://$LOCAL_IP:$PORT"
echo "  VISION_BACKEND=dgx"
echo ""
echo -e "${GREEN}Server starting...${NC}"
echo ""

# Start the server
uvicorn server:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level info

# If we get here, server stopped
echo ""
echo -e "${YELLOW}Server stopped${NC}"
