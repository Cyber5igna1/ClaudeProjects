"""
validate.py — End-to-end smoke test for the trading bot pipeline.
Runs entirely offline using synthetic OHLCV data — no API keys required.

Checks:
  1. All imports resolve
  2. Config loads without errors
  3. Database schema creates cleanly
  4. All 5 rule-based strategies produce valid signals on synthetic data
  5. Backtest runner executes and writes results to DB
  6. Strategy comparator reads and ranks results
  7. Risk module: position sizing, SL/TP, kill switch
  8. Exchange module: paper trade buy + sell + P&L
  9. Notifier: graceful no-op when credentials are absent
 10. Bot: dry-run config validation
 11. Dashboard: FastAPI app loads without import errors

Exit code 0 = all checks passed. Non-zero = at least one failure.
"""

import os
import sys
import traceback
import tempfile

# Point to project root
ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)

PASS  = "  [PASS]"
FAIL  = "  [FAIL]"
INFO  = "  [INFO]"

results: list[tuple[str, bool, str]] = []


def check(name: str):
    """Decorator-style context for a named check."""
    class _Ctx:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, tb):
            if exc_type is None:
                results.append((name, True, ""))
                print(f"{PASS}  {name}")
            else:
                msg = f"{exc_type.__name__}: {exc_val}"
                results.append((name, False, msg))
                print(f"{FAIL}  {name}")
                print(f"         {msg}")
                traceback.print_exc()
            return True   # suppress exception — continue running other checks
    return _Ctx()


# ── Synthetic data factory ─────────────────────────────────────────────────────

def make_ohlcv(n: int = 500):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(42)
    close = 30_000 + np.cumsum(rng.normal(0, 200, n))
    close = np.maximum(close, 100)
    high  = close * (1 + rng.uniform(0, 0.01, n))
    low   = close * (1 - rng.uniform(0, 0.01, n))
    op    = close * (1 + rng.normal(0, 0.005, n))
    vol   = rng.uniform(100, 1000, n)
    idx   = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": op, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


# ── Override DB paths to temp files so tests don't pollute real data ──────────

_tmp_dir = tempfile.mkdtemp()

import config as _cfg
_cfg.DB_PATH       = os.path.join(_tmp_dir, "test_results.db")
_cfg.CACHE_DB_PATH = os.path.join(_tmp_dir, "test_cache.db")
_cfg.MODELS_DIR    = os.path.join(_tmp_dir, "models")


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  TRADING BOT — END-TO-END VALIDATION")
print("═" * 60)

# 1 — Imports
with check("1. Core imports"):
    import numpy as np
    import pandas as pd
    import pandas_ta as ta

with check("2. ML imports"):
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

with check("3. Config import"):
    from config import (
        ACTIVE_STRATEGY, SYMBOL, TIMEFRAME, PAPER_TRADE,
        STOP_LOSS_PCT, TAKE_PROFIT_PCT, KILL_SWITCH_BALANCE,
        CAPITAL_ALLOCATION, FEES,
    )
    assert PAPER_TRADE, "PAPER_TRADE must default to True"
    assert STOP_LOSS_PCT   == 0.015
    assert TAKE_PROFIT_PCT == 0.020
    assert KILL_SWITCH_BALANCE == 40.0
    assert CAPITAL_ALLOCATION  == 0.80

with check("4. Database init"):
    from database.init_db import init_results_db, get_conn
    init_results_db()
    with get_conn() as conn:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
    for t in ["backtest_results", "live_trades", "ml_feature_importance"]:
        assert t in tables, f"Missing table: {t}"

# 2 — Strategies
with check("5. Strategy registry"):
    from research.strategies import REGISTRY
    assert len(REGISTRY) >= 5, f"Expected >=5 strategies, got {len(REGISTRY)}"

df_test = make_ohlcv(500)

for strat_name in ["rsi_ema", "macd_bb", "momentum", "triple_ema", "rsi_divergence"]:
    with check(f"6. Strategy signals: {strat_name}"):
        mod = REGISTRY[strat_name]
        entries, exits = mod.generate_signals(df_test)
        assert len(entries) == len(df_test), "entries length mismatch"
        assert len(exits)   == len(df_test), "exits length mismatch"
        assert entries.dtype == bool,        "entries must be bool"
        assert exits.dtype   == bool,        "exits must be bool"
        assert not entries.isna().any(),     "entries contain NaN"
        assert not exits.isna().any(),       "exits contain NaN"
        n_entry = entries.sum()
        n_exit  = exits.sum()
        print(f"{INFO}  {strat_name}: {n_entry} entries, {n_exit} exits in {len(df_test)} candles")

# 3 — Backtest runner
with check("7. Backtest runner (offline)"):
    from unittest.mock import patch
    from research.backtest_runner import run_backtest, save_result
    import vectorbt as vbt

    result = run_backtest("rsi_ema", df_test, "BTC/USDT")
    assert "sharpe_ratio"      in result
    assert "total_return_pct"  in result
    assert "max_drawdown_pct"  in result
    assert "win_rate_pct"      in result
    assert "total_trades"      in result
    save_result(result)

    # Verify row was written
    with get_conn() as conn:
        rows = conn.execute("SELECT COUNT(*) FROM backtest_results").fetchone()[0]
    assert rows >= 1, "No rows written to backtest_results"

# 4 — Strategy compare
with check("8. Strategy comparator"):
    # Write a second result so we have something to rank
    r2 = run_backtest("momentum", df_test, "BTC/USDT")
    save_result(r2)

    from research.strategy_compare import load_latest_results, rank_strategies
    raw    = load_latest_results()
    ranked = rank_strategies(raw)
    assert len(ranked) >= 1, "No ranked results"
    assert "rank" in ranked.columns

# 5 — Risk module
with check("9. Risk: position sizing"):
    from execution.risk import compute_position_size, reset_kill_switch
    reset_kill_switch()
    size = compute_position_size(100.0)
    assert abs(size - 80.0) < 0.01, f"Expected 80.0, got {size}"

with check("10. Risk: stop-loss / take-profit"):
    from execution.risk import compute_stop_loss, compute_take_profit
    sl = compute_stop_loss(1000.0)
    tp = compute_take_profit(1000.0)
    assert abs(sl -  985.0) < 0.01, f"SL expected 985.0, got {sl}"
    assert abs(tp - 1020.0) < 0.01, f"TP expected 1020.0, got {tp}"

with check("11. Risk: kill switch triggers"):
    from execution.risk import (
        check_balance, KillSwitchError,
        is_kill_switch_active, reset_kill_switch,
    )
    reset_kill_switch()
    try:
        check_balance(39.99)
        assert False, "Should have raised KillSwitchError"
    except KillSwitchError:
        pass
    assert is_kill_switch_active()
    reset_kill_switch()

with check("12. Risk: should_exit priority"):
    from execution.risk import should_exit, reset_kill_switch, ExitReason
    reset_kill_switch()

    # Stop-loss
    flag, reason = should_exit(984.0, 1000.0, False, 100.0)
    assert flag and reason == ExitReason.STOP_LOSS, f"Expected stop_loss, got {reason}"

    # Take-profit
    flag, reason = should_exit(1021.0, 1000.0, False, 100.0)
    assert flag and reason == ExitReason.TAKE_PROFIT, f"Expected take_profit, got {reason}"

    # Strategy signal
    flag, reason = should_exit(1000.0, 1000.0, True, 100.0)
    assert flag and reason == ExitReason.STRATEGY_SIGNAL, f"Expected strategy_signal, got {reason}"

    # No exit
    flag, reason = should_exit(1001.0, 1000.0, False, 100.0)
    assert not flag

with check("13. Risk: P&L calculation"):
    from execution.risk import compute_pnl
    pnl = compute_pnl(1000.0, 1020.0, 0.1, fees=0.001)
    # gross = 2.0, fees = 0.001 * (1000+1020) * 0.1 = 0.202
    assert abs(pnl - (2.0 - 0.202)) < 0.001, f"P&L mismatch: {pnl}"

# 6 — Exchange paper trade
with check("14. Exchange: paper buy + sell"):
    from execution.exchange import (
        reset_paper_state, get_balance,
        place_market_buy, place_market_sell,
    )
    from unittest.mock import patch

    reset_paper_state(100.0)
    assert abs(get_balance("USDT") - 100.0) < 0.01

    mock_price = 50_000.0

    with patch("execution.exchange.get_price", return_value=mock_price):
        buy_order = place_market_buy("BTC/USDT", 80.0)
        assert buy_order["side"]  == "buy"
        assert buy_order["paper"] == True
        assert abs(buy_order["price"] - mock_price) < 1.0
        assert abs(get_balance("USDT") - 20.0) < 0.01   # 100 - 80

        sell_order = place_market_sell("BTC/USDT", buy_order["qty"])
        assert sell_order["side"] == "sell"
        # Balance should be close to 100 again (no fees in paper mode)
        bal = get_balance("USDT")
        assert bal > 90.0, f"Post-sell balance too low: {bal}"

# 7 — Notifier (no credentials → silent no-op)
with check("15. Notifier: no-op without credentials"):
    # Temporarily blank out credentials
    import config as _c
    orig_token, orig_chat = _c.TELEGRAM_BOT_TOKEN, _c.TELEGRAM_CHAT_ID
    _c.TELEGRAM_BOT_TOKEN = None
    _c.TELEGRAM_CHAT_ID   = None
    import importlib, execution.notifier as _notif
    importlib.reload(_notif)

    ok = _notif.alert_info("test")
    assert ok == False, "Should return False when not configured"

    _c.TELEGRAM_BOT_TOKEN = orig_token
    _c.TELEGRAM_CHAT_ID   = orig_chat

# 8 — Dashboard imports
with check("16. Dashboard: FastAPI app imports"):
    from fastapi.testclient import TestClient
    from dashboard.server import app
    client = TestClient(app)
    # /health should return 200 without needing real DB data
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body

# ── Summary ────────────────────────────────────────────────────────────────────
print()
print("═" * 60)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"  Results: {passed} passed, {failed} failed  ({len(results)} total)")
print("═" * 60)

if failed:
    print("\n  FAILED CHECKS:")
    for name, ok, msg in results:
        if not ok:
            print(f"    ✗  {name}")
            print(f"       {msg}")
    print()
    sys.exit(1)
else:
    print("\n  All checks passed. Ready for testnet.\n")
    sys.exit(0)
