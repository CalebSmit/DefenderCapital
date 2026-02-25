"""
engine/utils.py — Shared utilities: logging, error handling, formatting, path helpers.
"""
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from functools import wraps
from typing import Any, Callable, Optional

# ── Project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
EXPORTS_DIR  = DATA_DIR / "exports"

# Ensure critical directories exist
for _d in (DATA_DIR, EXPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ── Logging setup ──────────────────────────────────────────────────────────────
def get_logger(name: str = "defender") -> logging.Logger:
    """
    Return a logger that writes to both stderr and a rolling log file.

    Parameters
    ----------
    name : str
        Logger name (typically the calling module's __name__).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (INFO+)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (DEBUG+)
    log_path = EXPORTS_DIR / "defender.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


log = get_logger("defender.utils")


# ── Timing / retry utilities ───────────────────────────────────────────────────
def retry(max_attempts: int = 3, base_delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator: retry a function up to *max_attempts* times with exponential backoff.

    Parameters
    ----------
    max_attempts : int
        Total attempts before raising the last exception.
    base_delay : float
        Initial sleep seconds between retries.
    backoff : float
        Multiplier applied to the delay on each subsequent retry.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            last_exc: Optional[Exception] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        log.warning(
                            f"{func.__name__} attempt {attempt}/{max_attempts} failed: {exc}. "
                            f"Retrying in {delay:.1f}s…"
                        )
                        time.sleep(delay)
                        delay *= backoff
                    else:
                        log.error(f"{func.__name__} failed after {max_attempts} attempts: {exc}")
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator


def timer(func: Callable) -> Callable:
    """Decorator: log wall-clock time of a function call."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        log.debug(f"{func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper


# ── Number / formatting helpers ────────────────────────────────────────────────
def fmt_currency(value: float, decimals: int = 2) -> str:
    """Format a float as USD currency, e.g. 12345.6 → '$12,345.60'."""
    if value < 0:
        return f"(${abs(value):,.{decimals}f})"
    return f"${value:,.{decimals}f}"


def fmt_pct(value: float, decimals: int = 2) -> str:
    """Format a fraction as percentage, e.g. 0.045 → '4.50%'."""
    return f"{value * 100:.{decimals}f}%"


def fmt_number(value: float, decimals: int = 2) -> str:
    """Format a number with thousand separators."""
    return f"{value:,.{decimals}f}"


def safe_divide(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Return numerator / denominator, or fallback if denominator is zero."""
    if denominator == 0 or denominator != denominator:  # NaN check
        return fallback
    return numerator / denominator


# ── Audit log helpers ──────────────────────────────────────────────────────────
class AuditLog:
    """
    Simple audit logger that accumulates test results and writes a plaintext
    report to data/exports/.

    Usage
    -----
    audit = AuditLog("Phase 1")
    audit.record("Data Integrity Test", "PASS", "All 35 tickers resolved.")
    audit.record("Cache Test", "FAIL", "Second run was not faster.")
    audit.save("audit_phase1.txt")
    """

    def __init__(self, phase_name: str):
        self.phase_name = phase_name
        self.entries: list[dict] = []
        self.start_time = datetime.now()

    def record(self, test_name: str, result: str, details: str = "") -> None:
        """
        Record a single test result.

        Parameters
        ----------
        test_name : str
            Human-readable name of the check.
        result : str
            "PASS", "FAIL", "WARN", or "INFO".
        details : str
            Additional context or error message.
        """
        entry = {
            "test": test_name,
            "result": result,
            "details": details,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self.entries.append(entry)
        level = logging.INFO if result in ("PASS", "INFO") else logging.WARNING
        log.log(level, f"[{result}] {test_name}: {details}")

    def save(self, filename: str) -> Path:
        """Write the audit log to data/exports/<filename>."""
        path = EXPORTS_DIR / filename
        lines = [
            "=" * 80,
            f"  DEFENDER CAPITAL MANAGEMENT — AUDIT LOG",
            f"  {self.phase_name}",
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
            "",
        ]
        pass_count = sum(1 for e in self.entries if e["result"] == "PASS")
        fail_count = sum(1 for e in self.entries if e["result"] == "FAIL")
        warn_count = sum(1 for e in self.entries if e["result"] == "WARN")
        lines += [
            f"  Summary: {pass_count} PASS  |  {fail_count} FAIL  |  {warn_count} WARN",
            f"  Total checks: {len(self.entries)}",
            "",
        ]
        for e in self.entries:
            symbol = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "INFO": "ℹ️"}.get(e["result"], "?")
            lines.append(f"{symbol}  [{e['result']:4s}]  {e['test']}")
            if e["details"]:
                for line in e["details"].splitlines():
                    lines.append(f"        {line}")
            lines.append(f"        Checked at: {e['timestamp']}")
            lines.append("")
        lines += ["=" * 80, "END OF AUDIT LOG", "=" * 80]
        path.write_text("\n".join(lines), encoding="utf-8")
        log.info(f"Audit log saved → {path}")
        return path

    @property
    def all_passed(self) -> bool:
        """True if no FAIL entries exist."""
        return all(e["result"] != "FAIL" for e in self.entries)


# ── Path helpers ───────────────────────────────────────────────────────────────
def get_portfolio_path() -> Path:
    """Return the canonical path to portfolio_holdings.xlsx."""
    return DATA_DIR / "portfolio_holdings.xlsx"



def timestamp_str() -> str:
    """Return current UTC timestamp as a filesystem-safe string."""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")
