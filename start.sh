#!/bin/bash
set -e

echo "Starting Eyes GEN — Discord token generator"
echo "========================================"

# Start TOR for IP rotation (fallback)
echo "[TOR] Starting..."
tor -f /etc/tor/torrc 2>/dev/null &
sleep 2
echo "[TOR] Ready (SOCKS5 :9050)" 2>/dev/null || echo "[TOR] Not available"

# Start the Python app
echo ""
echo "Starting web server..."
echo "Aug 22 05:58:35.000 [warn] You are running Tor as root. You don't need to, and you probably shouldn't."
echo "Aug 22 05:58:35.000 [info] solving with si vision"
exec python -u app.py
