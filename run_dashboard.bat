@echo off
REM ============================================================
REM  Defender Capital Management — Launch Risk Dashboard (Windows)
REM  Usage:  Double-click or run from Command Prompt
REM ============================================================

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo [->] Virtual environment activated
)

where streamlit >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Streamlit not found. Run setup.bat first.
    pause
    exit /b 1
)

echo ======================================================
echo  DCM Risk Model — Launching Dashboard
echo ======================================================
echo  Opening: http://localhost:8501
echo  Press Ctrl+C to stop
echo ======================================================

streamlit run dashboard\app.py ^
    --server.port 8501 ^
    --server.headless false ^
    --browser.gatherUsageStats false ^
    --theme.primaryColor "#1a3a5c" ^
    --theme.backgroundColor "#ffffff" ^
    --theme.secondaryBackgroundColor "#f4f6f9" ^
    --theme.textColor "#1a3a5c"
