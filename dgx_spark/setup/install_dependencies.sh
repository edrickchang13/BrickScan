#!/bin/bash
# BrickScan DGX Spark Dependency Installation
# Run on DGX Spark with Ubuntu 22.04+ and JetPack
# Usage: bash install_dependencies.sh

set -e

echo "=========================================="
echo "BrickScan DGX Spark Setup"
echo "=========================================="
echo ""

# Check if running on DGX/Jetson (ARM64)
if ! uname -m | grep -q "aarch64"; then
    echo "WARNING: This script is designed for ARM64 (DGX Spark/Jetson)"
    echo "You are running on: $(uname -m)"
    echo "Some components may not work correctly."
fi

# Update system
echo "[1/8] Updating system packages..."
sudo apt-get update -qq
sudo apt-get upgrade -y -qq

# Install essential tools
echo "[2/8] Installing essential development tools..."
sudo apt-get install -y -qq \
    curl \
    wget \
    git \
    build-essential \
    python3-dev \
    python3-pip \
    python3-venv \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev

# Install Ollama for local LLM/vision models
echo "[3/8] Installing Ollama (local LLM runtime)..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.ai/install.sh 2>/dev/null | sh
else
    echo "    Ollama already installed"
fi

# Start and enable Ollama
echo "[4/8] Starting Ollama service..."
sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama

# Wait for Ollama to be ready
echo "    Waiting for Ollama to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        echo "    Ollama is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "    WARNING: Ollama didn't start. Try: sudo systemctl status ollama"
    fi
    sleep 1
done

# Download vision models
echo "[5/8] Pulling vision models from Ollama registry..."
echo "    This may take 5-10 minutes (models are ~10GB total)..."

# LLaVA 13B - better accuracy for LEGO piece identification
echo "    Pulling llava:13b..."
ollama pull llava:13b 2>&1 | grep -E "pulling|verifying|writing|SUCCESS|Error" || true

# Moondream - faster fallback model
echo "    Pulling moondream..."
ollama pull moondream 2>&1 | grep -E "pulling|verifying|writing|SUCCESS|Error" || true

echo "    Vision models ready!"

# Create Python virtual environment for vision server
echo "[6/8] Setting up Python virtual environment..."
python3 -m venv ~/brickscan-env
source ~/brickscan-env/bin/activate

# Install FastAPI server dependencies
echo "[7/8] Installing Python dependencies for vision server..."
pip install --upgrade pip -q
pip install -q \
    fastapi \
    uvicorn[standard] \
    httpx \
    pillow \
    numpy \
    pydantic

# Install Blender headless (for rendering synthetic training data)
echo "[8/8] Installing Blender (this may take 2-3 minutes)..."

BLENDER_VERSION="4.1.0"
BLENDER_URL="https://mirrors.dotsrc.org/blender/blender-release/Blender4.1/blender-${BLENDER_VERSION}-linux-arm64.tar.xz"
BLENDER_DIR="/opt/blender-${BLENDER_VERSION}"

if [ ! -d "$BLENDER_DIR" ]; then
    echo "    Downloading Blender ${BLENDER_VERSION}..."
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"

    # Try primary URL first, fall back to alternative
    if ! wget -q "$BLENDER_URL" -O blender.tar.xz 2>/dev/null; then
        echo "    Primary mirror unavailable, trying alternative..."
        wget -q "https://www.blender.org/download/release/Blender4.1/blender-4.1.0-linux-arm64.tar.xz" -O blender.tar.xz || {
            echo "    WARNING: Could not download Blender. Check internet connection."
            echo "    You can install manually: https://www.blender.org/download/"
        }
    fi

    if [ -f "blender.tar.xz" ]; then
        echo "    Extracting Blender..."
        tar -xf blender.tar.xz
        sudo mkdir -p "$(dirname "$BLENDER_DIR")"
        sudo mv "blender-${BLENDER_VERSION}-linux-arm64" "$BLENDER_DIR"
        sudo ln -sf "$BLENDER_DIR/blender" /usr/local/bin/blender
        echo "    Blender installed to $BLENDER_DIR"
    fi
    cd ~
    rm -rf "$TEMP_DIR"
else
    echo "    Blender ${BLENDER_VERSION} already installed"
fi

# Create directories for project structure
echo ""
echo "[Setup] Creating project directories..."
mkdir -p ~/brickscan-training/data/{train,synthetic,val}
mkdir -p ~/brickscan-training/checkpoints
mkdir -p ~/brickscan-training/exports
mkdir -p ~/ldraw

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Verify Ollama models:"
echo "   ollama list"
echo ""
echo "2. Start the vision server (copy first):"
echo "   cp -r ../vision_server ~/brickscan-vision-server"
echo "   cd ~/brickscan-vision-server"
echo "   source ~/brickscan-env/bin/activate"
echo "   uvicorn server:app --host 0.0.0.0 --port 8001"
echo ""
echo "3. Test from Mac:"
echo "   curl http://$(hostname -I | awk '{print $1}'):8001/health"
echo ""
echo "4. Install as system service (optional):"
echo "   cd ~/brickscan-vision-server"
echo "   sudo bash systemd_service.sh"
echo ""
echo "5. Download LDraw parts library for rendering:"
echo "   cd ~/ldraw"
echo "   wget https://library.ldraw.org/latest-parts.html -O parts.zip"
echo "   unzip parts.zip"
echo ""
echo "For ML training setup, run:"
echo "   bash training/setup_ml_env.sh"
echo ""
