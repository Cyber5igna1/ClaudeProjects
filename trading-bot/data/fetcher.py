"""
data/fetcher.py — Downloads OHLCV data via ccxt and caches it in SQLite.
Supports incremental refresh to avoid re-downloading existing candles.
"""

import sqlite3
import time
import logging
from datetime import datetime, timezone, timedelta

import ccxt
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    EXCHANGE, API_KEY, API_SECRET,
    SYMBOLS, TIMEFRAME,
    OHLCV_HISTORY_DAYS, OHLCV_LIMIT_PER_REQUEST,
    CACHE_DB_PATH, USE_TESTNET, BINANCE_TESTNET_URL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Database setup ─────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(CACHE_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(CACHE_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_cache_db():
    """Create the ohlcv_cache table if it doesn't exist."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_cache (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol    TEXT    NOT NULL,
                timeframe TEXT    NOT NULL,
                timestamp INTEGER NOT NULL,
                open      REAL    NOT NULL,
                high      REAL    NOT NULL,
                low       REAL    NOT NULL,
                close     REAL    NOT NULL,
                volume    REAL    NOT NULL,
                UNIQUE(symbol, timeframe, timestamp)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ohlcv ON ohlcv_cache (symbol, timeframe, timestamp)"
        )
    log.info("Cache DB ready: %s", CACHE_DB_PATH)


# ── Exchange factory ───────────────────────────────────────────────────────────

_PLACEHOLDER_KEYS = {None, "", "your_binance_api_key", "your_binance_api_secret"}


def _build_exchange() -> ccxt.Exchange:
    exchange_class = getattr(ccxt, EXCHANGE)
    has_keys = API_KEY not in _PLACEHOLDER_KEYS and API_SECRET not in _PLACEHOLDER_KEYS
    params = {
        "enableRateLimit": True,
        "options": {
            "fetchCurrencies": False,
            "defaultType": "spot",
            "fetchMarkets": {"types": ["spot"]},  # skip margin/futures private endpoints
        },
    }
    if has_keys:
        params["apiKey"] = API_KEY
        params["secret"] = API_SECRET
    if USE_TESTNET and EXCHANGE == "binance":
        params["urls"] = {"api": {"public": BINANCE_TESTNET_URL, "private": BINANCE_TESTNET_URL}}
    return exchange_class(params)


# ── Fetch helpers ──────────────────────────────────────────────────────────────

def _latest_cached_timestamp(conn: sqlite3.Connection, symbol: str, timeframe: str) -> int | None:
    """Return the most recent cached candle timestamp (ms) or None if empty."""
    row = conn.execute(
        "SELECT MAX(timestamp) FROM ohlcv_cache WHERE symbol=? AND timeframe=?",
        (symbol, timeframe),
    ).fetchone()
    return row[0] if row and row[0] is not None else None


def _insert_candles(conn: sqlite3.Connection, symbol: str, timeframe: str, candles: list):
    """Bulk-insert candles, ignoring duplicates."""
    conn.executemany(
        """
        INSERT OR IGNORE INTO ohlcv_cache
            (symbol, timeframe, timestamp, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [(symbol, timeframe, c[0], c[1], c[2], c[3], c[4], c[5]) for c in candles],
    )


def _timeframe_ms(timeframe: str) -> int:
    """Convert a ccxt timeframe string to milliseconds."""
    units = {"m": 60, "h": 3600, "d": 86400, "w": 604800}
    return int(timeframe[:-1]) * units[timeframe[-1]] * 1000


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, timeframe: str = TIMEFRAME, days: int = OHLCV_HISTORY_DAYS) -> pd.DataFrame:
    """
    Fetch OHLCV data for a symbol, using cached data where available.
    Downloads only the missing candles (incremental refresh).

    Returns a DataFrame with columns: timestamp, open, high, low, close, volume
    Index is a UTC DatetimeIndex.
    """
    init_cache_db()
    exchange = _build_exchange()

    since_ms = int(
        (datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000
    )
    tf_ms = _timeframe_ms(timeframe)

    with _get_conn() as conn:
        latest = _latest_cached_timestamp(conn, symbol, timeframe)
        # If we have cached data, only fetch candles newer than the last one
        fetch_since = max(since_ms, latest + tf_ms) if latest else since_ms

        log.info(
            "Fetching %s %s from %s",
            symbol, timeframe,
            datetime.fromtimestamp(fetch_since / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
        )

        total_new = 0
        current_since = fetch_since

        while True:
            try:
                candles = exchange.fetch_ohlcv(
                    symbol, timeframe, since=current_since, limit=OHLCV_LIMIT_PER_REQUEST
                )
            except ccxt.RateLimitExceeded:
                log.warning("Rate limit hit — sleeping 10s")
                time.sleep(10)
                continue
            except ccxt.NetworkError as e:
                log.error("Network error: %s", e)
                break

            if not candles:
                break

            _insert_candles(conn, symbol, timeframe, candles)
            total_new += len(candles)
            log.info("  Fetched %d candles (total new: %d)", len(candles), total_new)

            if len(candles) < OHLCV_LIMIT_PER_REQUEST:
                break

            current_since = candles[-1][0] + tf_ms
            time.sleep(exchange.rateLimit / 1000)  # respect rate limit

        log.info("Done. %d new candles stored for %s %s.", total_new, symbol, timeframe)

        return _load_from_cache(conn, symbol, timeframe, since_ms)


def _load_from_cache(
    conn: sqlite3.Connection, symbol: str, timeframe: str, since_ms: int
) -> pd.DataFrame:
    """Load cached candles into a DataFrame."""
    df = pd.read_sql_query(
        """
        SELECT timestamp, open, high, low, close, volume
        FROM ohlcv_cache
        WHERE symbol=? AND timeframe=? AND timestamp>=?
        ORDER BY timestamp ASC
        """,
        conn,
        params=(symbol, timeframe, since_ms),
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df


def load_cached(symbol: str, timeframe: str = TIMEFRAME, days: int = OHLCV_HISTORY_DAYS) -> pd.DataFrame:
    """
    Load OHLCV data from cache only — no network calls.
    Raises RuntimeError if no data is cached for the given symbol/timeframe.
    """
    since_ms = int(
        (datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000
    )
    with _get_conn() as conn:
        df = _load_from_cache(conn, symbol, timeframe, since_ms)
    if df.empty:
        raise RuntimeError(
            f"No cached data for {symbol} {timeframe}. Run fetch_ohlcv() first."
        )
    return df


def refresh_all():
    """Refresh OHLCV cache for all configured symbols."""
    for symbol in SYMBOLS:
        log.info("=== Refreshing %s ===", symbol)
        fetch_ohlcv(symbol, TIMEFRAME)


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    refresh_all()
