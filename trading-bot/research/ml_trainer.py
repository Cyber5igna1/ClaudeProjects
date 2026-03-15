"""
research/ml_trainer.py — Feature engineering, Random Forest and XGBoost training.
Uses walk-forward validation to avoid lookahead bias.

Usage:
    python research/ml_trainer.py
    python research/ml_trainer.py --symbol ETH/USDT
"""

import os
import sys
import json
import logging
import argparse
import pickle
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    SYMBOLS, TIMEFRAME, MODELS_DIR,
    ML_LABEL_THRESHOLD,
    ML_WALK_FORWARD_TRAIN_MONTHS,
    ML_WALK_FORWARD_TEST_MONTHS,
    DB_PATH,
)
from data.fetcher import load_cached
from database.init_db import init_results_db, get_conn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

LABEL_BUY  =  1
LABEL_SELL = -1
LABEL_HOLD =  0


# ── Feature engineering ────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicator features from OHLCV data.
    All features are computed on closed candles — no lookahead.

    Returns a DataFrame of features aligned to df.index.
    NaN rows (warm-up period) are kept; callers must drop them before training.
    """
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    feat = pd.DataFrame(index=df.index)

    # RSI
    feat["rsi_14"]  = ta.rsi(close, length=14)
    feat["rsi_7"]   = ta.rsi(close, length=7)

    # MACD
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    feat["macd_line"]   = macd["MACD_12_26_9"]
    feat["macd_signal"] = macd["MACDs_12_26_9"]
    feat["macd_hist"]   = macd["MACDh_12_26_9"]

    # EMA slopes (normalised as % of price)
    for p in [9, 21, 50, 200]:
        ema = ta.ema(close, length=p)
        feat[f"ema_{p}_dist"] = (close - ema) / close  # % distance from EMA

    # Bollinger Band position (0 = lower band, 1 = upper band)
    bb = ta.bbands(close, length=20, std=2.0)
    bb_upper = bb["BBU_20_2.0"]
    bb_lower = bb["BBL_20_2.0"]
    bb_width = bb_upper - bb_lower
    feat["bb_position"] = (close - bb_lower) / bb_width.replace(0, np.nan)
    feat["bb_width_pct"] = bb_width / close

    # ATR (normalised)
    atr = ta.atr(high, low, close, length=14)
    feat["atr_pct"] = atr / close

    # Rate of Change
    feat["roc_5"]  = ta.roc(close, length=5)
    feat["roc_10"] = ta.roc(close, length=10)
    feat["roc_20"] = ta.roc(close, length=20)

    # Volume z-score (rolling 20-period)
    vol_mean = volume.rolling(20).mean()
    vol_std  = volume.rolling(20).std()
    feat["volume_zscore"] = (volume - vol_mean) / vol_std.replace(0, np.nan)

    # Returns
    feat["return_1h"]  = close.pct_change(1)
    feat["return_4h"]  = close.pct_change(4)
    feat["return_24h"] = close.pct_change(24)

    # High/Low range
    feat["candle_range"] = (high - low) / close
    feat["candle_body"]  = (close - df["open"]).abs() / close

    return feat


def label_candles(df: pd.DataFrame, threshold: float = ML_LABEL_THRESHOLD) -> pd.Series:
    """
    Label each candle based on the NEXT candle's return.
        Buy  (+1): next close > current close * (1 + threshold)
        Sell (-1): next close < current close * (1 - threshold)
        Hold  (0): otherwise

    Uses shift(-1) so each row's label is based on the next candle — no lookahead
    because the label is only used for training, never for live decisions.
    """
    future_return = df["close"].shift(-1) / df["close"] - 1
    labels = pd.Series(LABEL_HOLD, index=df.index, dtype=int)
    labels[future_return >  threshold] = LABEL_BUY
    labels[future_return < -threshold] = LABEL_SELL
    return labels


# ── Walk-forward validation ────────────────────────────────────────────────────

def walk_forward_splits(
    index: pd.DatetimeIndex,
    train_months: int = ML_WALK_FORWARD_TRAIN_MONTHS,
    test_months: int  = ML_WALK_FORWARD_TEST_MONTHS,
) -> list[tuple]:
    """
    Generate (train_mask, test_mask) pairs for walk-forward validation.
    Each fold: train on [start, train_end), test on [train_end, test_end).
    Window rolls forward by test_months each fold.
    """
    splits = []
    start = index[0]
    end   = index[-1]

    fold_start = start
    while True:
        train_end = fold_start + relativedelta(months=train_months)
        test_end  = train_end  + relativedelta(months=test_months)

        if test_end > end:
            break

        train_mask = (index >= fold_start) & (index < train_end)
        test_mask  = (index >= train_end)  & (index < test_end)

        if train_mask.sum() > 100 and test_mask.sum() > 10:
            splits.append((train_mask, test_mask))

        fold_start += relativedelta(months=test_months)

    return splits


# ── Model training ─────────────────────────────────────────────────────────────

def _build_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def _build_xgb() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )


def train_and_evaluate(
    features: pd.DataFrame,
    labels: pd.Series,
    symbol: str,
) -> dict:
    """
    Train RF and XGBoost using walk-forward validation.
    Returns evaluation summary and the final models trained on all data.
    """
    feat_clean = features.copy()
    lbl_clean  = labels.copy()

    # Drop rows with NaN features or the last row (label is NaN due to shift(-1))
    valid_mask = feat_clean.notna().all(axis=1) & lbl_clean.notna()
    feat_clean = feat_clean[valid_mask]
    lbl_clean  = lbl_clean[valid_mask]

    feature_names = feat_clean.columns.tolist()
    splits = walk_forward_splits(feat_clean.index)
    log.info("Walk-forward splits: %d folds", len(splits))

    rf_scores, xgb_scores = [], []
    rf_importances, xgb_importances = [], []

    for i, (train_mask, test_mask) in enumerate(splits):
        X_train = feat_clean[train_mask].values
        y_train = lbl_clean[train_mask].values
        X_test  = feat_clean[test_mask].values
        y_test  = lbl_clean[test_mask].values

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # XGBoost needs labels 0,1,2 (not -1,0,1)
        y_train_xgb = y_train + 1
        y_test_xgb  = y_test  + 1

        rf  = _build_rf()
        xgb = _build_xgb()

        rf.fit(X_train, y_train)
        xgb.fit(X_train, y_train_xgb)

        rf_acc  = accuracy_score(y_test, rf.predict(X_test))
        xgb_acc = accuracy_score(y_test_xgb, xgb.predict(X_test))

        rf_scores.append(rf_acc)
        xgb_scores.append(xgb_acc)
        rf_importances.append(rf.feature_importances_)
        xgb_importances.append(xgb.feature_importances_)

        log.info("  Fold %2d | RF acc=%.3f  XGB acc=%.3f  (train=%d test=%d)",
                 i + 1, rf_acc, xgb_acc, len(y_train), len(y_test))

    # ── Final models trained on all available data ─────────────────────────────
    X_all = feat_clean.values
    y_all = lbl_clean.values

    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X_all)

    final_rf  = _build_rf()
    final_xgb = _build_xgb()
    final_rf.fit(X_all_scaled, y_all)
    final_xgb.fit(X_all_scaled, y_all + 1)

    # ── Average feature importances across folds ───────────────────────────────
    avg_rf_imp  = np.mean(rf_importances,  axis=0)
    avg_xgb_imp = np.mean(xgb_importances, axis=0)

    summary = {
        "symbol": symbol,
        "feature_names": feature_names,
        "rf_mean_accuracy":  float(np.mean(rf_scores)),
        "xgb_mean_accuracy": float(np.mean(xgb_scores)),
        "rf_importance":     dict(zip(feature_names, avg_rf_imp.tolist())),
        "xgb_importance":    dict(zip(feature_names, avg_xgb_imp.tolist())),
        "final_rf":          final_rf,
        "final_xgb":         final_xgb,
        "final_scaler":      final_scaler,
        "feature_names":     feature_names,
    }

    log.info(
        "Final | RF mean_acc=%.3f  XGB mean_acc=%.3f",
        summary["rf_mean_accuracy"], summary["xgb_mean_accuracy"],
    )

    return summary


# ── Persist models and feature importance ──────────────────────────────────────

def save_models(summary: dict, symbol: str):
    """Pickle final models and scaler to MODELS_DIR."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    slug = symbol.replace("/", "_")

    for key, fname in [
        ("final_rf",     f"rf_{slug}.pkl"),
        ("final_xgb",    f"xgb_{slug}.pkl"),
        ("final_scaler", f"scaler_{slug}.pkl"),
    ]:
        path = os.path.join(MODELS_DIR, fname)
        with open(path, "wb") as f:
            pickle.dump(summary[key], f)
        log.info("Saved %s", path)

    # Save feature names alongside models
    meta_path = os.path.join(MODELS_DIR, f"features_{slug}.json")
    with open(meta_path, "w") as f:
        json.dump(summary["feature_names"], f)


def save_feature_importance(summary: dict, symbol: str):
    """Write feature importance rows to ml_feature_importance table."""
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows = []
    for model_name, importance_dict in [
        ("random_forest", summary["rf_importance"]),
        ("xgboost",       summary["xgb_importance"]),
    ]:
        for feature, importance in importance_dict.items():
            rows.append((model_name, run_date, feature, float(importance), symbol))

    with get_conn() as conn:
        conn.executemany(
            "INSERT INTO ml_feature_importance (model_name, run_date, feature_name, importance) VALUES (?,?,?,?)",
            [(r[0], r[1], r[2], r[3]) for r in rows],
        )
    log.info("Feature importance saved for %s.", symbol)


# ── Public entry point ─────────────────────────────────────────────────────────

def train(symbols: list[str] | None = None) -> dict:
    """Train models for all (or specified) symbols. Returns summaries keyed by symbol."""
    init_results_db()
    target_symbols = symbols or SYMBOLS
    summaries = {}

    for symbol in target_symbols:
        log.info("=== Training models for %s ===", symbol)
        try:
            df = load_cached(symbol, TIMEFRAME)
        except RuntimeError as e:
            log.error("%s — skipping. Run data/fetcher.py first.", e)
            continue

        features = build_features(df)
        labels   = label_candles(df)

        summary = train_and_evaluate(features, labels, symbol)
        save_models(summary, symbol)
        save_feature_importance(summary, symbol)
        summaries[symbol] = summary

        log.info(
            "%s — RF acc: %.3f | XGB acc: %.3f",
            symbol, summary["rf_mean_accuracy"], summary["xgb_mean_accuracy"],
        )

    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML models for trading strategies")
    parser.add_argument("--symbol", type=str, help="Train on a single symbol, e.g. BTC/USDT")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else None
    train(symbols=symbols)
