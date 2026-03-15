"""
database/init_db.py — Creates all SQLite tables on first run.
Tables: ohlcv_cache, backtest_results, live_trades, ml_feature_importance.
Safe to re-run (uses CREATE TABLE IF NOT EXISTS).
"""
