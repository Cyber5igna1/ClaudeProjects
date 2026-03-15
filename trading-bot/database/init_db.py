"""
database/init_db.py — Creates all SQLite tables on first run.
Tables: backtest_results, live_trades, ml_feature_importance.
Safe to re-run (uses CREATE TABLE IF NOT EXISTS).
"""

import sqlite3
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DB_PATH


def get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_results_db():
    """Create all application tables if they don't exist."""
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS backtest_results (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name     TEXT    NOT NULL,
                symbol            TEXT    NOT NULL,
                run_date          TEXT    NOT NULL,
                total_return_pct  REAL,
                sharpe_ratio      REAL,
                max_drawdown_pct  REAL,
                win_rate_pct      REAL,
                total_trades      INTEGER,
                avg_trade_duration_h REAL,
                params_json       TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS live_trades (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name   TEXT    NOT NULL,
                symbol          TEXT    NOT NULL,
                side            TEXT    NOT NULL,
                entry_price     REAL,
                exit_price      REAL,
                quantity        REAL,
                pnl_usd         REAL,
                entry_time      TEXT,
                exit_time       TEXT,
                paper_trade     INTEGER NOT NULL DEFAULT 1
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_feature_importance (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name   TEXT NOT NULL,
                run_date     TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                importance   REAL NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_backtest ON backtest_results (strategy_name, symbol, run_date)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_live_trades ON live_trades (strategy_name, entry_time)"
        )
    print(f"Results DB ready: {DB_PATH}")


if __name__ == "__main__":
    init_results_db()
