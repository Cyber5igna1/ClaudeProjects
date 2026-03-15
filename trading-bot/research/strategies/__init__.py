"""
research/strategies/__init__.py — Strategy registry.

Each strategy module exposes:
    generate_signals(df, params={}) -> (entries: pd.Series, exits: pd.Series)
    describe() -> dict  with keys: name, description, params

REGISTRY maps strategy name strings to their modules for use by
backtest_runner and the execution bot.
"""

from research.strategies import rsi_ema, macd_bb, momentum, triple_ema, rsi_divergence

REGISTRY: dict = {
    "rsi_ema":        rsi_ema,
    "macd_bb":        macd_bb,
    "momentum":       momentum,
    "triple_ema":     triple_ema,
    "rsi_divergence": rsi_divergence,
    # ml_rules, rf_classifier, xgb_classifier added after Phase 3
}
