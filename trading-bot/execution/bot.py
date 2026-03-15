"""
execution/bot.py — Main trading loop.
Runs every hour on candle close, evaluates the active strategy,
places orders via ccxt, enforces risk rules, logs trades, sends alerts.

Usage:
    python execution/bot.py
    python execution/bot.py --symbol ETH/USDT
    python execution/bot.py --dry-run      # validate config only, no loop
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    ACTIVE_STRATEGY, SYMBOL, TIMEFRAME,
    PAPER_TRADE, CAPITAL_ALLOCATION,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT, KILL_SWITCH_BALANCE,
)
from data.fetcher import load_cached, fetch_ohlcv
from database.init_db import init_results_db, get_conn
from execution.exchange import (
    get_balance, get_price, get_ohlcv,
    place_market_buy, place_market_sell,
    get_open_position,
)
from execution.risk import (
    can_enter, should_exit, compute_position_size,
    compute_pnl, status_summary,
    KillSwitchError, ExitReason,
)
from execution.notifier import (
    alert_buy, alert_sell, alert_kill_switch,
    alert_error, alert_start, alert_info,
)
from research.strategies import REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# Seconds to sleep between candle-close checks
_POLL_INTERVAL_S = 60       # check every 60s; act only on new closed candles
_OHLCV_LOOKBACK  = 300      # candles to fetch for indicator warm-up


# ── Trade logging ──────────────────────────────────────────────────────────────

def _log_trade_open(strategy: str, symbol: str, order: dict) -> int:
    """Insert an open trade row and return its row id."""
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO live_trades
                (strategy_name, symbol, side, entry_price, quantity, entry_time, paper_trade)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                strategy, symbol, order["side"],
                order["price"], order["qty"],
                order["timestamp"],
                1 if order["paper"] else 0,
            ),
        )
        return cur.lastrowid


def _log_trade_close(trade_id: int, exit_price: float, pnl_usd: float, exit_time: str):
    """Update the open trade row with exit data."""
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE live_trades
               SET exit_price=?, pnl_usd=?, exit_time=?
             WHERE id=?
            """,
            (exit_price, pnl_usd, exit_time, trade_id),
        )


# ── OHLCV helpers ──────────────────────────────────────────────────────────────

def _fetch_live_df(symbol: str, timeframe: str, limit: int = _OHLCV_LOOKBACK) -> pd.DataFrame:
    """
    Fetch recent candles from the exchange and return a DataFrame.
    The last candle is dropped — it is still open and would introduce lookahead bias.
    """
    raw = get_ohlcv(symbol, timeframe, limit=limit + 1)
    df  = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df.iloc[:-1]          # drop the current (unclosed) candle


def _last_closed_candle_ts(symbol: str, timeframe: str) -> pd.Timestamp:
    df = _fetch_live_df(symbol, timeframe, limit=2)
    return df.index[-1]


# ── Single-candle decision ─────────────────────────────────────────────────────

def _run_tick(
    symbol:         str,
    strategy_name:  str,
    open_trade:     dict | None,   # {id, entry_price, qty, entry_time} or None
    last_candle_ts: pd.Timestamp,
) -> tuple[dict | None, pd.Timestamp]:
    """
    Evaluate one candle and execute any entry/exit.

    Returns (updated open_trade, last_candle_ts).
    """
    df = _fetch_live_df(symbol, TIMEFRAME)
    current_ts = df.index[-1]

    # Skip if we already processed this candle
    if current_ts == last_candle_ts:
        return open_trade, last_candle_ts

    log.info("── New candle: %s ──", current_ts)
    last_candle_ts = current_ts
    current_price  = float(df["close"].iloc[-1])
    balance_usdt   = get_balance("USDT")

    module  = REGISTRY[strategy_name]
    entries, exits = module.generate_signals(df)
    strategy_entry = bool(entries.iloc[-1])
    strategy_exit  = bool(exits.iloc[-1])

    log.info(
        "Price=%.4f  Balance=%.2f USDT  Signal: entry=%s exit=%s",
        current_price, balance_usdt, strategy_entry, strategy_exit,
    )

    # ── EXIT path ──────────────────────────────────────────────────────────────
    if open_trade:
        entry_price = open_trade["entry_price"]
        qty         = open_trade["qty"]

        exit_flag, reason = should_exit(
            current_price  = current_price,
            entry_price    = entry_price,
            strategy_exit  = strategy_exit,
            balance_usdt   = balance_usdt,
        )

        if exit_flag:
            try:
                order    = place_market_sell(symbol, qty)
                pnl      = compute_pnl(entry_price, order["price"], qty)
                exit_ts  = order["timestamp"]

                _log_trade_close(open_trade["id"], order["price"], pnl, exit_ts)
                alert_sell(symbol, order["price"], qty, pnl, reason, entry_price)

                log.info(
                    "SOLD %s qty=%.6f @ %.4f  pnl=%.4f USD  reason=%s",
                    symbol, qty, order["price"], pnl, reason,
                )

                if reason == ExitReason.KILL_SWITCH:
                    new_balance = get_balance("USDT")
                    alert_kill_switch(reason, new_balance)
                    raise KillSwitchError(reason)

                return None, last_candle_ts

            except KillSwitchError:
                raise
            except Exception as e:
                alert_error("place_market_sell", e)
                log.exception("Failed to place sell order")
                return open_trade, last_candle_ts

    # ── ENTRY path ─────────────────────────────────────────────────────────────
    else:
        ok, reason = can_enter(
            has_open_position = open_trade is not None,
            balance_usdt      = balance_usdt,
            strategy_entry    = strategy_entry,
        )

        if ok:
            usdt_amount = compute_position_size(balance_usdt)
            try:
                order = place_market_buy(symbol, usdt_amount)
                trade_id = _log_trade_open(strategy_name, symbol, order)
                alert_buy(symbol, order["price"], order["qty"], usdt_amount)

                log.info(
                    "BOUGHT %s qty=%.6f @ %.4f  spent=%.2f USDT",
                    symbol, order["qty"], order["price"], usdt_amount,
                )

                return {
                    "id":          trade_id,
                    "entry_price": order["price"],
                    "qty":         order["qty"],
                    "entry_time":  order["timestamp"],
                }, last_candle_ts

            except Exception as e:
                alert_error("place_market_buy", e)
                log.exception("Failed to place buy order")
                return None, last_candle_ts

        elif reason not in ("no_signal", "already_in_position"):
            log.info("Entry blocked: %s", reason)

    return open_trade, last_candle_ts


# ── Main loop ──────────────────────────────────────────────────────────────────

def run(symbol: str = SYMBOL, strategy_name: str = ACTIVE_STRATEGY):
    """
    Main bot loop. Runs indefinitely, processing one candle per hour.
    Exits cleanly on KillSwitchError or KeyboardInterrupt.
    """
    if strategy_name not in REGISTRY:
        log.error("Unknown strategy '%s'. Available: %s", strategy_name, list(REGISTRY.keys()))
        sys.exit(1)

    init_results_db()

    mode = "PAPER TRADE" if PAPER_TRADE else "LIVE TRADING"
    log.info("=" * 60)
    log.info("Bot starting — %s", mode)
    log.info("Strategy : %s", strategy_name)
    log.info("Symbol   : %s", symbol)
    log.info("SL: %.1f%%  TP: %.1f%%  Kill: $%.0f  Alloc: %.0f%%",
             STOP_LOSS_PCT * 100, TAKE_PROFIT_PCT * 100,
             KILL_SWITCH_BALANCE, CAPITAL_ALLOCATION * 100)
    log.info("=" * 60)

    balance = get_balance("USDT")
    alert_start(symbol, balance)

    # Restore open position from DB if bot was restarted mid-trade
    open_trade     = _restore_open_trade(strategy_name, symbol)
    last_candle_ts = pd.Timestamp("1970-01-01", tz="UTC")

    if open_trade:
        log.info(
            "Restored open trade from DB: id=%d entry=%.4f qty=%.6f",
            open_trade["id"], open_trade["entry_price"], open_trade["qty"],
        )

    try:
        while True:
            try:
                open_trade, last_candle_ts = _run_tick(
                    symbol, strategy_name, open_trade, last_candle_ts
                )
            except KillSwitchError as e:
                log.critical("Kill switch — bot stopped: %s", e)
                break
            except Exception as e:
                alert_error("main loop tick", e)
                log.exception("Unhandled error in tick — sleeping 120s before retry")
                time.sleep(120)

            time.sleep(_POLL_INTERVAL_S)

    except KeyboardInterrupt:
        log.info("Bot stopped by user (KeyboardInterrupt).")
        alert_info(f"Bot stopped manually. Last symbol: {symbol}")

    log.info("Bot exited.")


# ── Restore open trade after restart ──────────────────────────────────────────

def _restore_open_trade(strategy_name: str, symbol: str) -> dict | None:
    """
    Check DB for a trade that was opened but never closed (bot crashed/restarted).
    Returns the trade dict if found, else None.
    """
    with get_conn() as conn:
        import sqlite3
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT id, entry_price, quantity, entry_time
            FROM live_trades
            WHERE strategy_name=? AND symbol=?
              AND (exit_time IS NULL OR exit_time='')
            ORDER BY entry_time DESC LIMIT 1
            """,
            (strategy_name, symbol),
        ).fetchone()

    if row:
        return {
            "id":          row["id"],
            "entry_price": row["entry_price"],
            "qty":         row["quantity"],
            "entry_time":  row["entry_time"],
        }
    return None


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the trading bot")
    parser.add_argument("--symbol",   type=str, default=SYMBOL,          help="Trading pair")
    parser.add_argument("--strategy", type=str, default=ACTIVE_STRATEGY, help="Strategy name")
    parser.add_argument("--dry-run",  action="store_true",               help="Validate config and exit")
    args = parser.parse_args()

    if args.dry_run:
        log.info("Dry-run: config OK — strategy=%s  symbol=%s  paper=%s",
                 args.strategy, args.symbol, PAPER_TRADE)
        sys.exit(0)

    run(symbol=args.symbol, strategy_name=args.strategy)
