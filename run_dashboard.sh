#!/usr/bin/env bash
# ============================================================
# Defender Capital Management — Launch Risk Dashboard (macOS / Linux)
# Usage:  bash run_dashboard.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"

# Activate virtual environment if present, otherwise use system Python
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    echo "→  Virtual environment activated"
fi

# Check streamlit is available
if ! command -v streamlit &>/dev/null; then
    echo "❌  Streamlit not found. Run setup.sh first."
    exit 1
fi

echo "======================================================"
echo " DCM Risk Model — Launching Dashboard"
echo "======================================================"
echo " Opening: http://localhost:8501"
echo " Press Ctrl+C to stop"
echo "======================================================"

streamlit run dashboard/app.py \
    --server.port 8501 \
    --server.headless false \
    --browser.gatherUsageStats false \
    --theme.primaryColor "#1a3a5c" \
    --theme.backgroundColor "#ffffff" \
    --theme.secondaryBackgroundColor "#f4f6f9" \
    --theme.textColor "#1a3a5c"
