#!/bin/bash
# BrickScan PyTorch ML Environment Setup on DGX Spark
#
# Sets up a Python virtual environment with:
# - PyTorch with CUDA support (ARM64 optimized)
# - Training utilities (timm, albumentations, wandb, etc.)
# - Export tools (coremltools, onnx)
#
# This uses NVIDIA's official PyTorch wheels for Jetson/JetPack
# which are ARM64-optimized and include CUDA support.
#
# Usage:
#   bash setup_ml_env.sh

set -e

echo "=========================================="
echo "BrickScan ML Environment Setup"
echo "=========================================="
echo ""

# Check if running on ARM64
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo "WARNING: This script is designed for ARM64 (DGX Spark)"
    echo "You are on: $ARCH"
    echo ""
fi

# Create virtual environment
VENV_PATH="$HOME/brickscan-ml-env"

if [ -d "$VENV_PATH" ]; then
    echo "Virtual environment already exists at $VENV_PATH"
    read -p "Delete and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_PATH"
    else
        echo "Using existing environment"
        source "$VENV_PATH/bin/activate"
        python --version
        exit 0
    fi
fi

echo "Creating Python virtual environment at $VENV_PATH..."
python3 -m venv "$VENV_PATH"

echo "Activating environment..."
source "$VENV_PATH/bin/activate"

echo ""
echo "Upgrading pip..."
pip install --upgrade pip -q

# Install PyTorch from NVIDIA's official wheels (ARM64 optimized)
echo ""
echo "Installing PyTorch with CUDA support (this may take 5-10 minutes)..."
echo "  Downloading PyTorch wheels for ARM64..."

# NVIDIA PyTorch index for Jetson/JetPack
# For JetPack 6.x with Python 3.10+
pip install --upgrade --quiet \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url https://pypi.ngc.nvidia.com || {

    echo ""
    echo "WARNING: Failed to install from NVIDIA PyTorch index"
    echo "Trying alternative method..."

    # Fallback to pip.org
    pip install --upgrade --quiet torch torchvision --no-cache-dir
}

# Verify PyTorch installation
echo ""
echo "Verifying PyTorch..."
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"

if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    CUDA_OK=1
else
    CUDA_OK=0
    echo "  WARNING: CUDA not available yet (may need reboot)"
fi

# Install training dependencies
echo ""
echo "Installing training dependencies..."
pip install --quiet \
    timm==0.9.12 \
    albumentations==1.3.1 \
    wandb==0.16.0 \
    tqdm==4.66.1 \
    pandas==2.1.3 \
    scikit-learn==1.3.2 \
    matplotlib==3.8.2 \
    seaborn==0.13.0 \
    pyyaml==6.0.1

# Install data/image processing
echo "Installing image processing libraries..."
pip install --quiet \
    opencv-python-headless==4.8.1 \
    Pillow==10.1.0 \
    numpy==1.24.3

# Install export tools
echo "Installing model export tools..."
pip install --quiet \
    coremltools==7.1 \
    onnx==1.15.0 \
    onnxruntime-gpu==1.17.0 || {

    echo "  Note: onnxruntime-gpu install optional (CPU fallback available)"
    pip install --quiet onnxruntime==1.17.0
}

# Install optional utilities
echo "Installing optional utilities..."
pip install --quiet \
    httpx==0.25.2 \
    requests==2.31.0 \
    python-dotenv==1.0.0

# Create directories
echo ""
echo "Creating project directories..."
mkdir -p $HOME/brickscan-training/{data,checkpoints,exports,training_logs}
mkdir -p $HOME/brickscan-training/data/{train,val,test,synthetic}

echo ""
echo "=========================================="
echo "ML Environment Setup Complete!"
echo "=========================================="
echo ""
echo "Environment: $VENV_PATH"
echo ""
echo "To use this environment:"
echo "  source $VENV_PATH/bin/activate"
echo ""
echo "Verify installation:"
python << 'VERIFY'
import sys
print("Python packages:")
packages = [
    ("torch", "PyTorch"),
    ("torchvision", "TorchVision"),
    ("timm", "Timm"),
    ("albumentations", "Albumentations"),
    ("sklearn", "Scikit-learn"),
    ("cv2", "OpenCV"),
    ("PIL", "Pillow"),
    ("onnx", "ONNX"),
    ("coremltools", "CoreMLTools"),
]

for pkg, name in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, "__version__", "unknown")
        print(f"  {name}: {version} ✓")
    except ImportError:
        print(f"  {name}: NOT INSTALLED")

print()
print("PyTorch Details:")
import torch
print(f"  CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print(f"  Device: CPU (training will be slow)")

print()
VERIFY

echo "Next steps:"
echo ""
echo "1. Prepare training data:"
echo "   - Organize into data/train/{class_id}/{images}"
echo "   - Or generate synthetic: bash rendering/blender_dgx_render.sh"
echo ""
echo "2. Start training:"
echo "   source ~/brickscan-ml-env/bin/activate"
echo "   bash training/train_on_dgx.sh"
echo ""
echo "3. Export trained model:"
echo "   python export/to_coreml.py --checkpoint checkpoints/best.pt"
echo "   python export/to_onnx.py --checkpoint checkpoints/best.pt"
echo ""
