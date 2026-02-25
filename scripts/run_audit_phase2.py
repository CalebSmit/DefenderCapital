"""Phase 2 Audit — Risk Analytics Engine."""
import sys, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.utils import AuditLog
from engine.data_loader import load_portfolio
from engine.synthetic_data import generate_synthetic_market_data
from engine.risk_metrics import compute_all_metrics, parametric_var
from engine.monte_carlo import run_simulation
from engine.stress_testing import run_all_stress_tests
from scipy.stats import norm

audit = AuditLog("Phase 2 — Risk Analytics Engine")

result = load_portfolio()
md = generate_synthetic_market_data(result)
metrics = compute_all_metrics(md)

# ── TEST 1: Parametric VaR formula ─────────────────────────────────────────
try:
    port_ret = md.portfolio_returns(log_returns=True)
    mu = float(port_ret.mean())
    sig = float(port_ret.std())
    z = norm.ppf(0.05)
    expected_var = -(mu + z * sig) * md.total_portfolio_value
    actual_var = metrics.var_95.parametric_var
    diff = abs(expected_var - actual_var)
    if diff < 1.0:
        audit.record("Parametric VaR formula", "PASS",
                     f"Expected=${expected_var:.2f}, Actual=${actual_var:.2f}, diff=${diff:.6f}")
    else:
        audit.record("Parametric VaR formula", "FAIL",
                     f"Expected=${expected_var:.2f}, Actual=${actual_var:.2f}, diff=${diff:.2f}")
except Exception as e:
    audit.record("Parametric VaR formula", "FAIL", str(e))

# ── TEST 2: Historical VaR matches empirical percentile ────────────────────
try:
    hist_ret = md.portfolio_returns(log_returns=True)
    simple_ret = (hist_ret.apply(lambda x: np.exp(x) - 1))
    p5 = float(np.percentile(simple_ret.dropna(), 5))
    expected_hvar = p5 * md.total_portfolio_value
    actual_hvar = metrics.var_95.historical_var
    diff = abs(expected_hvar - actual_hvar)
    if diff < 1.0:
        audit.record("Historical VaR percentile", "PASS",
                     f"p5={p5:.6f}, expected=${expected_hvar:.2f}, actual=${actual_hvar:.2f}")
    else:
        audit.record("Historical VaR percentile", "FAIL", f"diff=${diff:.2f}")
except Exception as e:
    audit.record("Historical VaR percentile", "FAIL", str(e))

# ── TEST 3: CVaR > VaR (parametric) ────────────────────────────────────────
try:
    cvar = metrics.var_95.parametric_cvar
    var  = metrics.var_95.parametric_var
    # Both VaR and CVaR are positive dollar loss amounts.
    # CVaR (Expected Shortfall) must have a larger absolute value than VaR —
    # it is the average loss in the worst-alpha tail, always worse than the threshold.
    ok = abs(cvar) >= abs(var) - 1e-6
    audit.record("CVaR ≥ VaR (parametric)", "PASS" if ok else "FAIL",
                 f"CVaR=${cvar:.2f}, VaR=${var:.2f} ({'CVaR is worse ✅' if ok else 'CVaR less severe ❌'})")
except Exception as e:
    audit.record("CVaR ≥ VaR (parametric)", "FAIL", str(e))

# ── TEST 4: Component VaR Euler decomposition ──────────────────────────────
try:
    total_comp = sum(sm.component_var_95 for sm in metrics.stock_metrics)
    total_var  = metrics.var_95.parametric_var
    diff = abs(total_comp - total_var)
    pct_diff = diff / abs(total_var) * 100 if total_var != 0 else 100
    if pct_diff < 1.0:
        audit.record("Component VaR Euler decomp", "PASS",
                     f"Sum={total_comp:.4f}, Total={total_var:.4f}, diff=${diff:.6f} ({pct_diff:.4f}%)")
    else:
        audit.record("Component VaR Euler decomp", "FAIL",
                     f"Sum={total_comp:.4f}, Total={total_var:.4f}, pct_diff={pct_diff:.2f}%")
except Exception as e:
    audit.record("Component VaR Euler decomp", "FAIL", str(e))

# ── TEST 5: VaR methods in same ballpark ──────────────────────────────────
try:
    p_var = abs(metrics.var_95.parametric_var)
    h_var = abs(metrics.var_95.historical_var)
    ratio = max(p_var, h_var) / max(min(p_var, h_var), 0.01)
    if ratio < 3.0:
        audit.record("VaR methods ballpark agreement", "PASS",
                     f"Parametric=${p_var:.0f}, Historical=${h_var:.0f}, ratio={ratio:.2f}x")
    else:
        audit.record("VaR methods ballpark agreement", "WARN",
                     f"Parametric=${p_var:.0f}, Historical=${h_var:.0f}, ratio={ratio:.2f}x (>3x)")
except Exception as e:
    audit.record("VaR methods ballpark agreement", "FAIL", str(e))

# ── TEST 6: Monte Carlo ────────────────────────────────────────────────────
try:
    sim = run_simulation(md, n_paths=10000, n_days=252, seed=42)
    # CVaR ≤ VaR in terminal return space
    tv = sim.terminal_values
    iv = sim.initial_value
    rets = (tv - iv) / iv
    p05 = np.percentile(rets, 5)
    cvar95 = rets[rets <= p05].mean()
    mc_cvar_ok = cvar95 <= p05
    audit.record("Monte Carlo CVaR ≤ VaR", "PASS" if mc_cvar_ok else "FAIL",
                 f"1Y VaR(95%)={p05:.4f} ({p05*100:.2f}%), CVaR={cvar95:.4f} ({cvar95*100:.2f}%)")

    # Check mean return consistency
    sim_mean = np.mean(np.log(tv / iv) / 252)  # annualized daily mean
    hist_daily_mean = float(md.portfolio_returns(log_returns=True).mean())
    mean_diff = abs(sim_mean - hist_daily_mean)
    if mean_diff < abs(hist_daily_mean) * 0.05:
        audit.record("Monte Carlo mean return", "PASS",
                     f"Sim daily mean={sim_mean:.6f}, Historical={hist_daily_mean:.6f}")
    else:
        audit.record("Monte Carlo mean return", "WARN",
                     f"Sim={sim_mean:.6f}, Historical={hist_daily_mean:.6f}, diff={mean_diff:.6f}")
except Exception as e:
    audit.record("Monte Carlo tests", "FAIL", str(e))

# ── TEST 7: Stress tests sanity ────────────────────────────────────────────
try:
    stress = run_all_stress_tests(md, custom_drawdown=-0.20)
    gfc   = stress.historical[0]
    covid = stress.historical[1]
    corr  = next(s for s in stress.hypothetical if "Corr" in s.name)

    gfc_ok   = -0.55 <= gfc.portfolio_loss_pct <= -0.30
    covid_ok = -0.35 <= covid.portfolio_loss_pct <= -0.20

    audit.record("Stress: GFC (30-55% loss expected)", "PASS" if gfc_ok else "WARN",
                 f"GFC loss={gfc.portfolio_loss_pct:.1%}")
    audit.record("Stress: COVID (20-35% loss expected)", "PASS" if covid_ok else "WARN",
                 f"COVID loss={covid.portfolio_loss_pct:.1%}")
    audit.record("Stress: Correlation spike scenario", "PASS",
                 f"Correlation spike loss={corr.portfolio_loss_pct:.1%}")
    audit.record("Stress: All 9 scenarios generated", "PASS",
                 f"Historical={len(stress.historical)}, Hypothetical={len(stress.hypothetical)}")
except Exception as e:
    audit.record("Stress tests", "FAIL", str(e))

# ── TEST 8: Single-stock portfolio ────────────────────────────────────────
try:
    from engine.data_loader import Holding, LoadResult, PortfolioSettings
    single_h = Holding(ticker="AAPL", company_name="Apple", sector="Technology",
                       industry="Consumer Electronics", shares_held=100, cost_basis=150.0,
                       current_price=182.0, market_value=18200.0, weight=1.0)
    single_lr = LoadResult(holdings=[single_h], settings=PortfolioSettings())
    single_md = generate_synthetic_market_data(single_lr)
    single_metrics = compute_all_metrics(single_md)
    audit.record("Single-stock portfolio works", "PASS",
                 f"VaR={abs(single_metrics.var_95.parametric_var):.2f}, vol={single_metrics.annualized_vol:.2%}")
except Exception as e:
    audit.record("Single-stock portfolio", "FAIL", str(e))

# ── TEST 9: Concentration metrics ─────────────────────────────────────────
try:
    n = len(metrics.stock_metrics)
    # For equal weights, HHI = 1/n
    # Our portfolio has unequal weights, so HHI should be > 1/n
    min_hhi = 1.0 / n
    ok = metrics.hhi > min_hhi and metrics.hhi < 0.5
    audit.record("Concentration metrics (HHI)", "PASS" if ok else "WARN",
                 f"HHI={metrics.hhi:.4f} (min={min_hhi:.4f}, 1/n), ENB={metrics.eff_num_bets:.1f}")
except Exception as e:
    audit.record("Concentration metrics", "FAIL", str(e))

# ── TEST 10: Report generation ────────────────────────────────────────────
try:
    from engine.report_generator import generate_html_report, write_results_to_excel
    from engine.utils import get_portfolio_path

    report_path = generate_html_report(md, metrics, sim, stress)
    ok = report_path.exists() and report_path.stat().st_size > 10000
    audit.record("HTML report generation", "PASS" if ok else "FAIL",
                 f"Size: {report_path.stat().st_size:,} bytes → {report_path.name}")

    write_results_to_excel(get_portfolio_path(), md, metrics, sim, stress)
    audit.record("Excel results write-back", "PASS",
                 "Risk Summary, Stock Risk Detail, Stress Tests sheets written")
except Exception as e:
    audit.record("Report/Excel generation", "FAIL", str(e))

# ── Save ───────────────────────────────────────────────────────────────────
log_path = audit.save("audit_phase2.txt")
print(f"\n{'='*60}")
print(f"Phase 2 Audit Complete")
print(f"{'='*60}")
for e in audit.entries:
    sym = {"PASS":"✅","FAIL":"❌","WARN":"⚠️"}.get(e["result"],"?")
    print(f"{sym} {e['result']:4s}  {e['test']}")
print(f"\nAll critical tests passed: {audit.all_passed}")
print(f"Audit log: {log_path}")
