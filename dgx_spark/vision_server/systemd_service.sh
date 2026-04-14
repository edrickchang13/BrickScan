#!/bin/bash
# Install BrickScan Vision Server as systemd service
#
# This makes the vision server start automatically on DGX Spark boot
# and manages its lifecycle (restart on failure, etc.)
#
# Usage: sudo bash systemd_service.sh
#
# After installation:
#   sudo systemctl status brickscan-vision
#   sudo systemctl start/stop/restart brickscan-vision
#   sudo systemctl enable/disable brickscan-vision
#   sudo journalctl -u brickscan-vision -f

set -e

# Require root
if [ "$EUID" -ne 0 ]; then
    echo "This script must be run as root (sudo)"
    exit 1
fi

echo "=========================================="
echo "BrickScan Vision Server - Systemd Setup"
echo "=========================================="
echo ""

# Get the actual user running sudo (not root)
ACTUAL_USER="${SUDO_USER:-$(whoami)}"
if [ "$ACTUAL_USER" = "root" ]; then
    echo "ERROR: Could not determine actual user. Run with sudo."
    exit 1
fi

ACTUAL_HOME=$(eval echo ~$ACTUAL_USER)
INSTALL_DIR="$ACTUAL_HOME/brickscan-vision-server"
VENV_PATH="$ACTUAL_HOME/brickscan-env"

echo "Configuration:"
echo "  User: $ACTUAL_USER"
echo "  Home: $ACTUAL_HOME"
echo "  Install Dir: $INSTALL_DIR"
echo "  Venv: $VENV_PATH"
echo ""

# Check if server files exist
if [ ! -f "$INSTALL_DIR/server.py" ]; then
    echo "ERROR: server.py not found at $INSTALL_DIR/server.py"
    echo ""
    echo "Setup first:"
    echo "  cd ~/brickscan-vision-server"
    echo "  sudo bash systemd_service.sh"
    exit 1
fi

# Check if venv exists
if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    echo "Run setup/install_dependencies.sh first"
    exit 1
fi

# Create systemd service file
SERVICE_FILE="/etc/systemd/system/brickscan-vision.service"

echo "Creating systemd service file: $SERVICE_FILE"

cat > "$SERVICE_FILE" << 'EOF'
[Unit]
Description=BrickScan Local Vision Inference Server
Documentation=https://github.com/yourusername/brickscan
After=network.target ollama.service
Requires=ollama.service
Wants=network-online.target
ConditionPathExists=%h/brickscan-env/bin/python

[Service]
Type=notify
User=%i
Group=%i
WorkingDirectory=%h/brickscan-vision-server
Environment="PATH=%h/brickscan-env/bin:/usr/local/sbin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="PYTHONUNBUFFERED=1"
ExecStart=%h/brickscan-env/bin/uvicorn server:app --host 0.0.0.0 --port 8001 --workers 1

# Restart policy
Restart=always
RestartSec=5
StartLimitInterval=60
StartLimitBurst=5

# Resource limits (optional)
MemoryLimit=2G
CPUQuota=80%

# Security
NoNewPrivileges=true
PrivateTmp=true

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=brickscan-vision

[Install]
WantedBy=multi-user.target
EOF

# Replace %i with actual user and %h with home directory
sed -i "s|%i|$ACTUAL_USER|g" "$SERVICE_FILE"
sed -i "s|%h|$ACTUAL_HOME|g" "$SERVICE_FILE"

# Set permissions
chmod 644 "$SERVICE_FILE"

echo "Service file created successfully"
echo ""

# Reload systemd
echo "Reloading systemd daemon..."
systemctl daemon-reload

# Enable service (auto-start on boot)
echo "Enabling service to start on boot..."
systemctl enable brickscan-vision

# Start the service
echo "Starting service..."
systemctl start brickscan-vision

# Wait for it to start
sleep 2

# Check status
echo ""
echo "=========================================="
echo "Service Installation Complete"
echo "=========================================="
echo ""

if systemctl is-active --quiet brickscan-vision; then
    echo "Status: RUNNING ✓"
else
    echo "Status: FAILED ✗"
    echo ""
    echo "Check logs:"
    echo "  sudo journalctl -u brickscan-vision -n 50"
    exit 1
fi

echo ""
echo "Usage:"
echo "  sudo systemctl status brickscan-vision      # Check status"
echo "  sudo systemctl stop brickscan-vision        # Stop service"
echo "  sudo systemctl restart brickscan-vision     # Restart service"
echo "  sudo systemctl disable brickscan-vision     # Stop auto-start on boot"
echo ""
echo "Logs:"
echo "  sudo journalctl -u brickscan-vision -f      # Follow logs"
echo "  sudo journalctl -u brickscan-vision -n 100  # Last 100 lines"
echo ""
echo "Test from Mac:"
DGX_IP=$(hostname -I | awk '{print $1}')
echo "  curl http://$DGX_IP:8001/health"
echo ""
echo "Update backend .env:"
echo "  DGX_VISION_URL=http://$DGX_IP:8001"
echo "  VISION_BACKEND=dgx"
echo ""
