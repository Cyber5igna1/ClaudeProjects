"""
research/backtest_runner.py — Loads all strategies and runs vectorbt backtests.
Writes performance metrics to results.db (table: backtest_results).

Usage:
    python research/backtest_runner.py
    python research/backtest_runner.py --strategy rsi_ema --symbol BTC/USDT
"""

import json
import logging
import argparse
from datetime import datetime, timezone

import numpy as np
import vectorbt as vbt
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import SYMBOLS, TIMEFRAME, STOP_LOSS_PCT, TAKE_PROFIT_PCT, FEES
from data.fetcher import load_cached
from database.init_db import init_results_db, get_conn
from research.strategies import REGISTRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Backtest a single strategy ─────────────────────────────────────────────────

def run_backtest(
    strategy_name: str,
    df: pd.DataFrame,
    symbol: str,
) -> dict:
    """
    Run a vectorbt backtest for one strategy on one symbol.

    Returns a dict of performance metrics.
    """
    module = REGISTRY[strategy_name]
    entries, exits = module.generate_signals(df)
    params = module.PARAMS

    close = df["close"]

    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        sl_stop=STOP_LOSS_PCT,
        tp_stop=TAKE_PROFIT_PCT,
        fees=FEES,
        freq="1h",
        init_cash=100.0,        # normalised — metrics are % based
        size=1.0,               # invest full allocation each signal
        size_type="percent",
    )

    stats = portfolio.stats()

    total_trades = int(stats.get("Total Trades", 0))
    win_rate = float(stats.get("Win Rate [%]", 0.0) or 0.0)
    sharpe = float(stats.get("Sharpe Ratio", 0.0) or 0.0)
    max_dd = float(stats.get("Max Drawdown [%]", 0.0) or 0.0)
    total_return = float(stats.get("Total Return [%]", 0.0) or 0.0)

    # Average trade duration in hours
    trades_df = portfolio.trades.records_readable
    if not trades_df.empty and "Duration" in trades_df.columns:
        avg_duration_h = trades_df["Duration"].dt.total_seconds().mean() / 3600
    else:
        avg_duration_h = 0.0

    result = {
        "strategy_name": strategy_name,
        "symbol": symbol,
        "run_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "total_return_pct": round(total_return, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd, 4),
        "win_rate_pct": round(win_rate, 4),
        "total_trades": total_trades,
        "avg_trade_duration_h": round(avg_duration_h, 2),
        "params_json": json.dumps(params),
    }

    log.info(
        "%-20s %-10s | return=%+.1f%%  sharpe=%.2f  dd=%.1f%%  wr=%.1f%%  trades=%d",
        strategy_name, symbol,
        total_return, sharpe, max_dd, win_rate, total_trades,
    )

    return result


# ── Persist results ────────────────────────────────────────────────────────────

def save_result(result: dict):
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO backtest_results
                (strategy_name, symbol, run_date, total_return_pct, sharpe_ratio,
                 max_drawdown_pct, win_rate_pct, total_trades, avg_trade_duration_h, params_json)
            VALUES
                (:strategy_name, :symbol, :run_date, :total_return_pct, :sharpe_ratio,
                 :max_drawdown_pct, :win_rate_pct, :total_trades, :avg_trade_duration_h, :params_json)
            """,
            result,
        )


# ── Print summary table ────────────────────────────────────────────────────────

def print_summary(results: list[dict]):
    if not results:
        print("No results.")
        return

    df = pd.DataFrame(results).sort_values("sharpe_ratio", ascending=False)
    df = df[[
        "strategy_name", "symbol", "total_return_pct",
        "sharpe_ratio", "max_drawdown_pct", "win_rate_pct", "total_trades",
    ]]
    df.columns = ["Strategy", "Symbol", "Return %", "Sharpe", "MaxDD %", "WinRate %", "Trades"]

    print("\n" + "=" * 80)
    print("BACKTEST RESULTS — ranked by Sharpe Ratio")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")


# ── Main entry point ───────────────────────────────────────────────────────────

def run_all(strategies: list[str] | None = None, symbols: list[str] | None = None):
    """
    Run backtests for all (or specified) strategies across all (or specified) symbols.
    Results are saved to results.db and printed as a summary table.
    """
    init_results_db()

    target_strategies = strategies or list(REGISTRY.keys())
    target_symbols = symbols or SYMBOLS

    results = []

    for symbol in target_symbols:
        log.info("Loading data for %s ...", symbol)
        try:
            df = load_cached(symbol, TIMEFRAME)
        except RuntimeError as e:
            log.error("%s — skipping. Run data/fetcher.py first.", e)
            continue

        log.info("  %d candles loaded (%s → %s)", len(df), df.index[0].date(), df.index[-1].date())

        for name in target_strategies:
            if name not in REGISTRY:
                log.warning("Unknown strategy '%s' — skipping.", name)
                continue
            try:
                result = run_backtest(name, df, symbol)
                save_result(result)
                results.append(result)
            except Exception as e:
                log.error("Error running %s on %s: %s", name, symbol, e)

    print_summary(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run strategy backtests")
    parser.add_argument("--strategy", type=str, help="Run a single strategy by name")
    parser.add_argument("--symbol",   type=str, help="Run on a single symbol, e.g. BTC/USDT")
    args = parser.parse_args()

    strategies = [args.strategy] if args.strategy else None
    symbols    = [args.symbol]   if args.symbol   else None

    run_all(strategies=strategies, symbols=symbols)
