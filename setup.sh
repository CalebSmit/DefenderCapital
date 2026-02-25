#!/usr/bin/env bash
# ============================================================
# Defender Capital Management — Risk Model Setup (macOS / Linux)
# Run once to install dependencies and create the environment.
# Usage:  bash setup.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================"
echo " DCM Risk Model — Setup"
echo "======================================================"

# ── 1. Check Python version ──────────────────────────────────
MIN_PYTHON="3.10"
PYTHON_CMD=""
for cmd in python3 python3.12 python3.11 python3.10; do
    if command -v "$cmd" &>/dev/null; then
        VERSION=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" 2>/dev/null; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "❌  Python 3.10+ is required. Please install it from https://python.org"
    exit 1
fi
echo "✅  Python: $($PYTHON_CMD --version)"

# ── 2. Create virtual environment ───────────────────────────
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "→  Creating virtual environment at .venv/"
    "$PYTHON_CMD" -m venv "$VENV_DIR"
else
    echo "→  Virtual environment already exists at .venv/"
fi

# Activate
source "$VENV_DIR/bin/activate"
echo "✅  Virtual environment activated"

# ── 3. Upgrade pip and install packages ─────────────────────
echo "→  Installing packages (this may take 1–2 minutes)…"
pip install --quiet --upgrade pip
pip install --quiet \
    openpyxl \
    pandas \
    numpy \
    yfinance \
    scipy \
    matplotlib \
    seaborn \
    plotly \
    kaleido \
    streamlit \
    requests \
    python-dateutil \
    jinja2 \
    pytest

echo "✅  All packages installed"

# ── 4. Create necessary folders ──────────────────────────────
mkdir -p data/exports data/cache
echo "✅  Directories ready"

# ── 5. Quick smoke-test ───────────────────────────────────────
echo "→  Running quick smoke-test…"
python3 -c "
import sys
sys.path.insert(0, '.')
from engine.data_loader import load_portfolio
r = load_portfolio()
print(f'   Portfolio loaded: {len(r.holdings)} holdings')
"
echo "✅  Smoke-test passed"

echo ""
echo "======================================================"
echo " Setup complete!  Next steps:"
echo "   1. Edit data/portfolio_holdings.xlsx with your holdings"
echo "   2. Run the dashboard:  bash run_dashboard.sh"
echo "   3. Or generate a report: bash update_model.sh"
echo "======================================================"
