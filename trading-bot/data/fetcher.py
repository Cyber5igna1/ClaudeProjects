"""
data/fetcher.py — Downloads OHLCV data via ccxt and caches it in SQLite.
Supports incremental refresh to avoid re-downloading existing candles.
"""
