#!/bin/bash
# Find DGX Spark on Local Network
#
# Scans your local network for the DGX Spark and BrickScan vision server.
# Run this on your Mac to discover the DGX Spark's IP address.
#
# Usage:
#   bash find_dgx.sh                    # Auto-detect network
#   NETWORK=192.168.1 bash find_dgx.sh # Specify network range

set -e

echo "=========================================="
echo "Finding DGX Spark on Local Network"
echo "=========================================="
echo ""

# Detect local network
if [ -z "$NETWORK" ]; then
    # Get local IP (macOS)
    LOCAL_IP=$(ifconfig | grep -m1 "inet " | awk '{print $2}')

    if [ -z "$LOCAL_IP" ]; then
        # Fallback for Linux
        LOCAL_IP=$(hostname -I | awk '{print $1}')
    fi

    if [ -z "$LOCAL_IP" ]; then
        echo "ERROR: Could not detect local IP address"
        echo "Try: NETWORK=192.168.1 bash find_dgx.sh"
        exit 1
    fi

    # Extract network prefix
    NETWORK=$(echo $LOCAL_IP | sed 's/\.[0-9]*$//')
fi

echo "Local IP: $LOCAL_IP"
echo "Network: ${NETWORK}.0/24"
echo ""

# Check if nmap is available
if command -v nmap &> /dev/null; then
    echo "Using nmap to scan network (requires sudo for best results)..."
    echo "Scanning port 8001 (vision server)..."
    echo ""

    # Try to scan with sudo (might need password)
    if sudo -n nmap -p 8001 --open "${NETWORK}.0/24" 2>/dev/null | grep "report for" | while read line; do
        IP=$(echo "$line" | grep -o '[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}' | head -1)
        if [ ! -z "$IP" ]; then
            echo "Testing $IP:8001..."
            if curl -s --connect-timeout 1 "http://$IP:8001/health" &>/dev/null; then
                echo ""
                echo "Found BrickScan Vision Server at: $IP:8001"
                echo ""
                curl -s "http://$IP:8001/health" | python3 -m json.tool 2>/dev/null || curl -s "http://$IP:8001/health"
                echo ""
                echo "Add to backend .env:"
                echo "  DGX_VISION_URL=http://$IP:8001"
                exit 0
            fi
        fi
    done; then
        exit 0
    fi
fi

# Fallback: Try common IP ranges
echo "Scanning common IP ranges for vision server (port 8001)..."
echo "This may take 2-3 minutes..."
echo ""

FOUND=0

for i in $(seq 1 254); do
    IP="${NETWORK}.$i"

    # Try to connect to vision server
    if timeout 1 curl -s "http://$IP:8001/health" &>/dev/null; then
        echo ""
        echo "======================================"
        echo "Found DGX Spark at: $IP"
        echo "======================================"
        echo ""

        # Get server info
        curl -s "http://$IP:8001/health" | python3 -m json.tool 2>/dev/null || \
            curl -s "http://$IP:8001/health"

        echo ""
        echo "Add to backend .env:"
        echo "  DGX_VISION_URL=http://$IP:8001"
        echo ""

        # Try SSH
        if timeout 2 ssh -o ConnectTimeout=1 ubuntu@"$IP" "hostname" &>/dev/null; then
            echo "SSH available: ssh ubuntu@$IP"
        fi

        echo ""
        FOUND=$((FOUND + 1))
    fi

    # Show progress every 25 IPs
    if [ $((i % 25)) -eq 0 ]; then
        echo -ne "Scanned up to ${NETWORK}.$i...\r"
    fi
done

echo ""
echo ""

if [ $FOUND -eq 0 ]; then
    echo "No DGX Spark found on ${NETWORK}.0/24"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check DGX Spark is powered on and on the network"
    echo "  2. Check vision server is running:"
    echo "     ssh ubuntu@<dgx-ip>"
    echo "     sudo systemctl status brickscan-vision"
    echo "  3. Verify network connectivity:"
    echo "     ping $(echo $NETWORK).1  # Ping gateway"
    echo "  4. Try a different network:"
    echo "     NETWORK=192.168.0 bash find_dgx.sh"
    echo ""
    exit 1
else
    echo "Found $FOUND device(s) with vision server running"
fi
