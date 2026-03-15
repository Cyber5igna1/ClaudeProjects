"""
research/strategies/macd_bb.py — Strategy 2: MACD + Bollinger Bands
Buy when MACD line crosses above signal line AND price is at or below lower BB.
Sell when price touches upper BB OR MACD crosses back down.

Interface: generate_signals(df) -> (entries, exits)
"""

import pandas as pd
import pandas_ta as ta

PARAMS = {
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2.0,
}

NAME = "macd_bb"


def generate_signals(df: pd.DataFrame, params: dict = PARAMS) -> tuple[pd.Series, pd.Series]:
    """
    Entry:  MACD line crosses above signal line  AND  close <= lower Bollinger Band
    Exit:   close >= upper Bollinger Band  OR  MACD line crosses below signal line
    """
    p = {**PARAMS, **params}

    close = df["close"]

    macd_df = ta.macd(close, fast=p["macd_fast"], slow=p["macd_slow"], signal=p["macd_signal"])
    macd_line = macd_df[f"MACD_{p['macd_fast']}_{p['macd_slow']}_{p['macd_signal']}"]
    macd_sig  = macd_df[f"MACDs_{p['macd_fast']}_{p['macd_slow']}_{p['macd_signal']}"]

    bb_df = ta.bbands(close, length=p["bb_period"], std=p["bb_std"])
    bb_lower = bb_df[f"BBL_{p['bb_period']}_{p['bb_std']}"]
    bb_upper = bb_df[f"BBU_{p['bb_period']}_{p['bb_std']}"]

    # Crossover: current bar macd > signal, previous bar macd <= signal
    macd_cross_up   = (macd_line > macd_sig) & (macd_line.shift(1) <= macd_sig.shift(1))
    macd_cross_down = (macd_line < macd_sig) & (macd_line.shift(1) >= macd_sig.shift(1))

    entries = macd_cross_up & (close <= bb_lower)
    exits   = (close >= bb_upper) | macd_cross_down

    return entries.fillna(False), exits.fillna(False)


def describe() -> dict:
    return {
        "name": NAME,
        "description": "Buy on MACD cross-up at lower Bollinger Band. Sell at upper BB or MACD cross-down.",
        "params": PARAMS,
    }
