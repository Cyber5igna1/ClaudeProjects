"""
research/strategies/rsi_divergence.py — Strategy 5: RSI Divergence
Buy on bullish divergence: price makes a lower low while RSI makes a higher low.
Sell on bearish divergence: price makes a higher high while RSI makes a lower high.

Divergence is detected over a rolling lookback window.

Interface: generate_signals(df) -> (entries, exits)
"""

import pandas as pd
import pandas_ta as ta

PARAMS = {
    "rsi_period": 14,
    "lookback": 5,   # candles to look back for swing highs/lows
}

NAME = "rsi_divergence"


def _rolling_min_idx(series: pd.Series, window: int) -> pd.Series:
    """Position (iloc offset) of the minimum in each rolling window."""
    return series.rolling(window).apply(lambda x: x.argmin(), raw=True)


def generate_signals(df: pd.DataFrame, params: dict = PARAMS) -> tuple[pd.Series, pd.Series]:
    """
    Bullish divergence (entry):
        close makes a lower low over the lookback window
        AND RSI makes a higher low over the same window.

    Bearish divergence (exit):
        close makes a higher high over the lookback window
        AND RSI makes a lower high over the same window.
    """
    p = {**PARAMS, **params}
    lb = p["lookback"]

    close = df["close"]
    rsi = ta.rsi(close, length=p["rsi_period"])

    # Rolling min/max for price and RSI
    price_min  = close.rolling(lb).min()
    price_max  = close.rolling(lb).max()
    rsi_min    = rsi.rolling(lb).min()
    rsi_max    = rsi.rolling(lb).max()

    # Bullish divergence: current close is at rolling low, but RSI low is rising
    price_at_low  = close == price_min
    rsi_low_rising = rsi_min > rsi_min.shift(lb)

    # Bearish divergence: current close is at rolling high, but RSI high is falling
    price_at_high  = close == price_max
    rsi_high_falling = rsi_max < rsi_max.shift(lb)

    entries = price_at_low & rsi_low_rising
    exits   = price_at_high & rsi_high_falling

    return entries.fillna(False), exits.fillna(False)


def describe() -> dict:
    return {
        "name": NAME,
        "description": "Buy on bullish RSI divergence (price lower low, RSI higher low). Sell on bearish divergence.",
        "params": PARAMS,
    }
