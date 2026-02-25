@echo off
REM ============================================================
REM  Defender Capital Management — Update Model (Windows)
REM  Fetches latest prices, recomputes risk metrics, generates report.
REM  Usage:  Double-click or run from Command Prompt
REM ============================================================

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

echo ======================================================
echo  DCM Risk Model — Refreshing Data & Generating Report
echo ======================================================

python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
from engine.data_loader import load_portfolio
from engine.synthetic_data import get_market_data
from engine.risk_metrics import compute_all_metrics
from engine.monte_carlo import run_simulation
from engine.stress_testing import run_all_stress_tests
from engine.report_generator import generate_html_report, write_results_to_excel
from engine.utils import get_portfolio_path

print('Loading portfolio...')
lr = load_portfolio()
print(f'  {len(lr.holdings)} holdings loaded')
print('Fetching market data...')
md = get_market_data(lr)
print(f'  {len(md.quality.valid_tickers)}/{len(lr.holdings)} tickers valid')
print('Computing risk metrics...')
metrics = compute_all_metrics(md)
print(f'  VaR(95%): \${metrics.var_95.parametric_var:,.0f}')
print('Running Monte Carlo (10,000 paths)...')
sim = run_simulation(md, n_paths=10000, n_days=252, seed=42)
print(f'  Median 1Y terminal value: \${sim.median_terminal:,.0f}')
print('Running stress tests...')
stress = run_all_stress_tests(md)
print(f'  {len(stress.historical) + len(stress.hypothetical)} scenarios complete')
print('Generating HTML report...')
report = generate_html_report(md, metrics, sim, stress)
print(f'  Saved: {report}')
print('Writing results to Excel...')
write_results_to_excel(get_portfolio_path(), md, metrics, sim, stress)
print(f'  Updated: {get_portfolio_path().name}')
print()
print('Update complete! Report saved to:', report)
"

if errorlevel 1 (
    echo [ERROR] Update failed. See error above.
    pause
    exit /b 1
)

echo ======================================================
echo  Update complete!
echo ======================================================
pause
