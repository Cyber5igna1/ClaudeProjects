"""
research/strategies/triple_ema.py — Strategy 4: Triple EMA Crossover
Buy when EMA9 > EMA21 > EMA50 (full bullish stack).
Sell when EMA9 crosses below EMA21.

Interface: generate_signals(df) -> (entries, exits)
"""

import pandas as pd
import pandas_ta as ta

PARAMS = {
    "ema_fast": 9,
    "ema_mid": 21,
    "ema_slow": 50,
}

NAME = "triple_ema"


def generate_signals(df: pd.DataFrame, params: dict = PARAMS) -> tuple[pd.Series, pd.Series]:
    """
    Entry:  EMA9 > EMA21  AND  EMA21 > EMA50  (bullish alignment, enter on transition)
    Exit:   EMA9 crosses below EMA21
    """
    p = {**PARAMS, **params}

    close = df["close"]

    ema_fast = ta.ema(close, length=p["ema_fast"])
    ema_mid  = ta.ema(close, length=p["ema_mid"])
    ema_slow = ta.ema(close, length=p["ema_slow"])

    bullish_stack = (ema_fast > ema_mid) & (ema_mid > ema_slow)
    prev_stack    = (ema_fast.shift(1) <= ema_mid.shift(1))  # wasn't stacked before

    # Enter on the first candle the full bullish stack forms
    entries = bullish_stack & prev_stack

    # Exit when fast EMA crosses below mid EMA
    exits = (ema_fast < ema_mid) & (ema_fast.shift(1) >= ema_mid.shift(1))

    return entries.fillna(False), exits.fillna(False)


def describe() -> dict:
    return {
        "name": NAME,
        "description": "Buy when EMA9 > EMA21 > EMA50 stack forms. Sell when EMA9 crosses below EMA21.",
        "params": PARAMS,
    }
