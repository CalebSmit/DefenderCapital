@echo off
REM ============================================================
REM  Defender Capital Management — Risk Model Setup (Windows)
REM  Run once to install dependencies and create the environment.
REM  Usage:  Double-click or run from Command Prompt
REM ============================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ======================================================
echo  DCM Risk Model — Setup
echo ======================================================

REM ── 1. Check Python ──────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+ from https://python.org
    echo         Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)
echo [OK] Python found:
python --version

REM ── 2. Create virtual environment ────────────────────────
if not exist ".venv" (
    echo [->] Creating virtual environment at .venv\
    python -m venv .venv
) else (
    echo [->] Virtual environment already exists at .venv\
)

REM ── 3. Install packages ───────────────────────────────────
echo [->] Installing packages (this may take 1-2 minutes)...
call .venv\Scripts\activate.bat

python -m pip install --quiet --upgrade pip
python -m pip install --quiet ^
    openpyxl ^
    pandas ^
    numpy ^
    yfinance ^
    scipy ^
    matplotlib ^
    seaborn ^
    plotly ^
    kaleido ^
    streamlit ^
    requests ^
    python-dateutil ^
    jinja2 ^
    pytest

echo [OK] All packages installed

REM ── 4. Create directories ─────────────────────────────────
if not exist "data\exports" mkdir data\exports
if not exist "data\cache"   mkdir data\cache
echo [OK] Directories ready

REM ── 5. Smoke-test ─────────────────────────────────────────
echo [->] Running quick smoke-test...
python -c "import sys; sys.path.insert(0,'.'); from engine.data_loader import load_portfolio; r=load_portfolio(); print(f'   Portfolio loaded: {len(r.holdings)} holdings')"
if errorlevel 1 (
    echo [ERROR] Smoke-test failed. Check the error above.
    pause
    exit /b 1
)
echo [OK] Smoke-test passed

echo.
echo ======================================================
echo  Setup complete!  Next steps:
echo    1. Edit data\portfolio_holdings.xlsx with your holdings
echo    2. Run the dashboard:  run_dashboard.bat
echo    3. Or generate a report: update_model.bat
echo ======================================================
pause
