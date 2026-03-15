"""
research/strategies/momentum.py — Strategy 3: Momentum (ROC + Volume)
Buy when Rate of Change is positive and volume spikes above rolling average.
Sell when ROC turns negative or volume spike fades.

Interface: generate_signals(df) -> (entries, exits)
"""

import pandas as pd
import pandas_ta as ta

PARAMS = {
    "roc_period": 10,
    "volume_ma_period": 20,
    "volume_spike_multiplier": 1.5,  # volume must be > 1.5x rolling average
}

NAME = "momentum"


def generate_signals(df: pd.DataFrame, params: dict = PARAMS) -> tuple[pd.Series, pd.Series]:
    """
    Entry:  ROC > 0  AND  volume > volume_ma * spike_multiplier
    Exit:   ROC < 0
    """
    p = {**PARAMS, **params}

    close  = df["close"]
    volume = df["volume"]

    roc = ta.roc(close, length=p["roc_period"])
    volume_ma = volume.rolling(window=p["volume_ma_period"]).mean()

    volume_spike = volume > (volume_ma * p["volume_spike_multiplier"])

    entries = (roc > 0) & volume_spike
    exits   = roc < 0

    return entries.fillna(False), exits.fillna(False)


def describe() -> dict:
    return {
        "name": NAME,
        "description": "Buy when ROC positive and volume spikes 1.5x above average. Sell when ROC turns negative.",
        "params": PARAMS,
    }
