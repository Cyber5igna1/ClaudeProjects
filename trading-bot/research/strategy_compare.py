"""
research/strategy_compare.py — Reads backtest_results from SQLite and generates
a ranked comparison report sorted by Sharpe Ratio.

Prints a formatted table to stdout and returns structured data for the dashboard.

Usage:
    python research/strategy_compare.py
    python research/strategy_compare.py --symbol ETH/USDT
    python research/strategy_compare.py --top 3
"""

import os
import sys
import argparse
import logging
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SYMBOLS
from database.init_db import get_conn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Minimum trades for a result to be considered valid
MIN_TRADES = 5


# ── Data loading ───────────────────────────────────────────────────────────────

def load_latest_results(symbol: str | None = None) -> pd.DataFrame:
    """
    Load the most recent backtest result per strategy (and symbol).
    If symbol is given, filter to that symbol only.
    """
    with get_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT
                r.strategy_name,
                r.symbol,
                r.run_date,
                r.total_return_pct,
                r.sharpe_ratio,
                r.max_drawdown_pct,
                r.win_rate_pct,
                r.total_trades,
                r.avg_trade_duration_h,
                r.params_json
            FROM backtest_results r
            INNER JOIN (
                SELECT strategy_name, symbol, MAX(run_date) AS latest
                FROM backtest_results
                GROUP BY strategy_name, symbol
            ) latest
            ON  r.strategy_name = latest.strategy_name
            AND r.symbol        = latest.symbol
            AND r.run_date      = latest.latest
            ORDER BY r.sharpe_ratio DESC
            """,
            conn,
        )

    if symbol:
        df = df[df["symbol"] == symbol]

    return df


# ── Scoring ────────────────────────────────────────────────────────────────────

def _composite_score(row: pd.Series) -> float:
    """
    Composite score that balances return, risk-adjusted return, and drawdown.
    Used as a secondary sort when Sharpe ratios are similar.

    Score = (sharpe * 0.5) + (total_return_pct / 100 * 0.3) - (max_drawdown_pct / 100 * 0.2)
    """
    sharpe   = row["sharpe_ratio"]       or 0.0
    ret      = row["total_return_pct"]   or 0.0
    drawdown = row["max_drawdown_pct"]   or 0.0
    return (sharpe * 0.5) + (ret / 100 * 0.3) - (drawdown / 100 * 0.2)


def rank_strategies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply ranking logic:
    1. Filter out strategies with fewer than MIN_TRADES trades.
    2. Compute composite score.
    3. Sort by Sharpe descending, then composite score descending.
    4. Add rank column.
    """
    df = df[df["total_trades"] >= MIN_TRADES].copy()

    if df.empty:
        return df

    df["composite_score"] = df.apply(_composite_score, axis=1)
    df = df.sort_values(
        ["sharpe_ratio", "composite_score"],
        ascending=[False, False],
    ).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


# ── Formatting ─────────────────────────────────────────────────────────────────

def _bar(value: float, max_val: float, width: int = 20, fill: str = "█") -> str:
    """Render a simple ASCII bar scaled to max_val."""
    if max_val == 0:
        return ""
    filled = int(round(abs(value) / max_val * width))
    return fill * filled


def print_report(df: pd.DataFrame, top: int | None = None):
    """Print a formatted comparison table to stdout."""
    if df.empty:
        print("\nNo backtest results found. Run research/backtest_runner.py first.\n")
        return

    display = df.head(top) if top else df

    print()
    print("╔" + "═" * 98 + "╗")
    print("║  STRATEGY COMPARISON REPORT" + " " * 71 + "║")
    print("║  Ranked by Sharpe Ratio" + " " * 75 + "║")
    print("║  Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M") + " " * 62 + "║")
    print("╠" + "═" * 98 + "╣")
    print(
        f"║  {'#':>2}  {'Strategy':<20} {'Symbol':<10} "
        f"{'Return %':>9} {'Sharpe':>7} {'MaxDD %':>8} {'WinRate %':>10} {'Trades':>7} "
        f"{'AvgDur(h)':>10}  ║"
    )
    print("╠" + "═" * 98 + "╣")

    max_return = display["total_return_pct"].abs().max() or 1

    for _, row in display.iterrows():
        ret_bar = _bar(row["total_return_pct"], max_return, width=8)
        ret_sign = "+" if (row["total_return_pct"] or 0) >= 0 else ""
        print(
            f"║  {int(row['rank']):>2}  {row['strategy_name']:<20} {row['symbol']:<10} "
            f"{ret_sign}{row['total_return_pct']:>7.2f}% "
            f"{row['sharpe_ratio']:>7.3f} "
            f"{row['max_drawdown_pct']:>7.2f}% "
            f"{row['win_rate_pct']:>9.1f}% "
            f"{int(row['total_trades']):>7} "
            f"{row['avg_trade_duration_h']:>10.1f}  ║"
        )

    print("╠" + "═" * 98 + "╣")

    # Best-in-class summary
    best_sharpe  = df.loc[df["sharpe_ratio"].idxmax()]
    best_return  = df.loc[df["total_return_pct"].idxmax()]
    least_dd     = df.loc[df["max_drawdown_pct"].idxmin()]

    print(f"║  Best Sharpe  : {best_sharpe['strategy_name']} ({best_sharpe['symbol']}) "
          f"→ {best_sharpe['sharpe_ratio']:.3f}"
          + " " * max(0, 51 - len(best_sharpe['strategy_name']) - len(best_sharpe['symbol'])) + "║")
    print(f"║  Best Return  : {best_return['strategy_name']} ({best_return['symbol']}) "
          f"→ +{best_return['total_return_pct']:.2f}%"
          + " " * max(0, 49 - len(best_return['strategy_name']) - len(best_return['symbol'])) + "║")
    print(f"║  Least Drawdown: {least_dd['strategy_name']} ({least_dd['symbol']}) "
          f"→ -{least_dd['max_drawdown_pct']:.2f}%"
          + " " * max(0, 48 - len(least_dd['strategy_name']) - len(least_dd['symbol'])) + "║")
    print("╚" + "═" * 98 + "╝")
    print()


# ── Recommendation ─────────────────────────────────────────────────────────────

def recommend_best(df: pd.DataFrame) -> dict | None:
    """
    Return the top-ranked strategy as a dict.
    Returns None if no valid results exist.
    Used by the execution bot to validate the configured ACTIVE_STRATEGY.
    """
    ranked = rank_strategies(df)
    if ranked.empty:
        return None

    best = ranked.iloc[0]
    return {
        "strategy_name":     best["strategy_name"],
        "symbol":            best["symbol"],
        "sharpe_ratio":      best["sharpe_ratio"],
        "total_return_pct":  best["total_return_pct"],
        "max_drawdown_pct":  best["max_drawdown_pct"],
        "win_rate_pct":      best["win_rate_pct"],
        "total_trades":      int(best["total_trades"]),
        "run_date":          best["run_date"],
    }


# ── Public entry point ─────────────────────────────────────────────────────────

def compare(symbol: str | None = None, top: int | None = None) -> pd.DataFrame:
    """
    Load latest results, rank them, print the report, and return the ranked DataFrame.
    """
    raw = load_latest_results(symbol=symbol)
    ranked = rank_strategies(raw)
    print_report(ranked, top=top)

    best = recommend_best(ranked)
    if best:
        log.info(
            "Recommended strategy: %s on %s (Sharpe=%.3f, Return=+%.2f%%)",
            best["strategy_name"], best["symbol"],
            best["sharpe_ratio"], best["total_return_pct"],
        )

    return ranked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare strategy backtest results")
    parser.add_argument("--symbol", type=str,  help="Filter to a single symbol")
    parser.add_argument("--top",    type=int,  help="Show only top N strategies")
    args = parser.parse_args()

    compare(symbol=args.symbol, top=args.top)
