#!/usr/bin/env bash
# ============================================================
# Defender Capital Management — Update Model & Generate Report
# Fetches latest prices, recomputes all risk metrics, and
# generates a fresh HTML report + Excel write-back.
# Usage:  bash update_model.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

echo "======================================================"
echo " DCM Risk Model — Refreshing Data & Generating Report"
echo "======================================================"

python3 - <<'PYEOF'
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

print("→  Loading portfolio…")
lr = load_portfolio()
print(f"   {len(lr.holdings)} holdings loaded")

print("→  Fetching market data…")
md = get_market_data(lr)
print(f"   Data ready: {len(md.quality.valid_tickers)}/{len(lr.holdings)} tickers valid")

print("→  Computing risk metrics…")
metrics = compute_all_metrics(md)
print(f"   VaR(95%): ${metrics.var_95.parametric_var:,.0f}  Vol: {metrics.annualized_vol:.1%}  Sharpe: {metrics.sharpe:.2f}")

print("→  Running Monte Carlo (10,000 paths)…")
sim = run_simulation(md, n_paths=10_000, n_days=252, seed=42)
print(f"   Median 1Y: ${sim.p05_terminal:,.0f}–${sim.p95_terminal:,.0f} (P5–P95)")

print("→  Running stress tests…")
stress = run_all_stress_tests(md)
print(f"   {len(stress.historical)} historical + {len(stress.hypothetical)} hypothetical scenarios")

print("→  Generating HTML report…")
report = generate_html_report(md, metrics, sim, stress)
print(f"   Saved: {report}")

print("→  Writing results to Excel…")
write_results_to_excel(get_portfolio_path(), md, metrics, sim, stress)
print(f"   Updated: {get_portfolio_path().name}")

print()
print("====================================================")
print(" Update complete!")
print(f" Report: {report}")
print("====================================================")
PYEOF
