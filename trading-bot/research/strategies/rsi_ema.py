"""
research/strategies/rsi_ema.py — Strategy 1: RSI + EMA
Buy when RSI(14) < 35 and price > 200 EMA.
Sell when RSI > 60 or take-profit / stop-loss hit.

Interface contract (shared by all strategies):
    generate_signals(df) -> (entries: pd.Series, exits: pd.Series)

    df must have columns: open, high, low, close, volume
    with a UTC DatetimeIndex.

    entries / exits are boolean Series aligned to df.index.
    True = signal fires on that candle's close.
"""

import pandas as pd
import pandas_ta as ta

# ── Parameters ─────────────────────────────────────────────────────────────────
PARAMS = {
    "rsi_period": 14,
    "rsi_buy_threshold": 35,
    "rsi_sell_threshold": 60,
    "ema_period": 200,
}

NAME = "rsi_ema"


def generate_signals(df: pd.DataFrame, params: dict = PARAMS) -> tuple[pd.Series, pd.Series]:
    """
    Compute entry and exit signals for the RSI + EMA strategy.

    Entry:  RSI(14) < 35  AND  close > EMA(200)
    Exit:   RSI(14) > 60

    Stop-loss and take-profit are enforced by the backtester (vectorbt)
    and the execution layer (risk.py), not here.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with DatetimeIndex.
    params : dict
        Override default strategy parameters.

    Returns
    -------
    entries : pd.Series[bool]
    exits   : pd.Series[bool]
    """
    p = {**PARAMS, **params}

    close = df["close"]

    rsi = ta.rsi(close, length=p["rsi_period"])
    ema = ta.ema(close, length=p["ema_period"])

    entries = (rsi < p["rsi_buy_threshold"]) & (close > ema)
    exits = rsi > p["rsi_sell_threshold"]

    # Ensure no NaN leaks into signals (treat NaN as False)
    entries = entries.fillna(False)
    exits = exits.fillna(False)

    return entries, exits


def describe() -> dict:
    """Return strategy metadata for reporting."""
    return {
        "name": NAME,
        "description": "Buy when RSI(14) < 35 and price above EMA(200). Sell when RSI > 60.",
        "params": PARAMS,
    }
