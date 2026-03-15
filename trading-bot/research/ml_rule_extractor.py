"""
research/ml_rule_extractor.py — Extracts interpretable if/then rules from trained ML models.
Writes rules as Python code into research/strategies/ml_rules.py.

Extraction method:
  - Fits a shallow decision tree (depth 4) on the same features used by RF/XGBoost.
  - Walks every leaf path that predicts Buy or Sell.
  - Converts each path into a Python boolean expression.
  - Writes generate_signals() to ml_rules.py so it can be backtested like any other strategy.

Usage:
    python research/ml_rule_extractor.py
    python research/ml_rule_extractor.py --symbol ETH/USDT
"""

import os
import sys
import json
import pickle
import logging
import argparse
import textwrap
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SYMBOLS, TIMEFRAME, MODELS_DIR
from data.fetcher import load_cached
from research.ml_trainer import build_features, label_candles, LABEL_BUY, LABEL_SELL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

EXTRACTOR_TREE_DEPTH = 4        # shallow tree → readable rules
MIN_LEAF_SAMPLES_PCT = 0.01     # leaf must cover ≥ 1% of training data


# ── Load artefacts ─────────────────────────────────────────────────────────────

def _load_artefacts(symbol: str) -> tuple[object, object, list[str]]:
    slug = symbol.replace("/", "_")
    rf_path      = os.path.join(MODELS_DIR, f"rf_{slug}.pkl")
    scaler_path  = os.path.join(MODELS_DIR, f"scaler_{slug}.pkl")
    feature_path = os.path.join(MODELS_DIR, f"features_{slug}.json")

    for p in [rf_path, scaler_path, feature_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing artefact: {p}. Run ml_trainer.py first.")

    with open(rf_path,      "rb") as f: rf     = pickle.load(f)
    with open(scaler_path,  "rb") as f: scaler = pickle.load(f)
    with open(feature_path, "r")  as f: feature_names = json.load(f)

    return rf, scaler, feature_names


# ── Fit surrogate tree ─────────────────────────────────────────────────────────

def _fit_surrogate_tree(
    X_scaled: np.ndarray,
    y: np.ndarray,
    n_samples: int,
) -> DecisionTreeClassifier:
    """
    Fit a shallow decision tree that mimics the RF predictions.
    Using RF predictions (not true labels) makes the rules consistent
    with what the ensemble actually learned.
    """
    min_samples = max(10, int(n_samples * MIN_LEAF_SAMPLES_PCT))

    tree = DecisionTreeClassifier(
        max_depth=EXTRACTOR_TREE_DEPTH,
        min_samples_leaf=min_samples,
        class_weight="balanced",
        random_state=42,
    )
    tree.fit(X_scaled, y)
    return tree


# ── Rule extraction ────────────────────────────────────────────────────────────

def _extract_rules(
    tree: DecisionTreeClassifier,
    feature_names: list[str],
    scaler: StandardScaler,
    n_samples: int,
) -> list[dict]:
    """
    Walk every leaf in the decision tree and collect paths that predict
    Buy (+1) or Sell (-1).

    Returns a list of rule dicts:
        {
          "label":      1 | -1,
          "conditions": ["rsi_14 < 35.2", "ema_200_dist > -0.01", ...],
          "raw_conditions": [("feature", threshold, leq), ...],  # for code gen
          "confidence": float,   # fraction of leaf samples with this label
          "coverage":   float,   # fraction of training samples in leaf
        }
    """
    t = tree.tree_
    class_labels = tree.classes_   # e.g. [-1, 0, 1]
    rules = []

    def _recurse(node: int, conditions: list):
        if t.feature[node] == _tree.TREE_UNDEFINED:
            # Leaf node
            class_counts = t.value[node][0]
            predicted_idx = int(np.argmax(class_counts))
            label = class_labels[predicted_idx]

            if label not in (LABEL_BUY, LABEL_SELL):
                return  # skip Hold leaves

            total = class_counts.sum()
            confidence = class_counts[predicted_idx] / total if total > 0 else 0.0
            coverage   = total / n_samples

            if confidence < 0.50:   # skip low-confidence leaves
                return

            rules.append({
                "label":          int(label),
                "conditions":     [c["display"] for c in conditions],
                "raw_conditions": [(c["feature"], c["threshold"], c["leq"]) for c in conditions],
                "confidence":     round(float(confidence), 3),
                "coverage":       round(float(coverage), 4),
            })
            return

        feat_idx   = t.feature[node]
        feat_name  = feature_names[feat_idx]
        threshold_scaled = t.threshold[node]

        # Un-scale threshold back to original feature space for readability
        mean  = scaler.mean_[feat_idx]
        scale = scaler.scale_[feat_idx]
        threshold_orig = threshold_scaled * scale + mean

        # Left child: feature <= threshold
        _recurse(t.children_left[node], conditions + [{
            "feature":   feat_name,
            "threshold": threshold_orig,
            "leq":       True,
            "display":   f"{feat_name} <= {threshold_orig:.6g}",
        }])

        # Right child: feature > threshold
        _recurse(t.children_right[node], conditions + [{
            "feature":   feat_name,
            "threshold": threshold_orig,
            "leq":       False,
            "display":   f"{feat_name} > {threshold_orig:.6g}",
        }])

    _recurse(0, [])
    return rules


# ── Code generation ────────────────────────────────────────────────────────────

def _condition_to_code(feature: str, threshold: float, leq: bool) -> str:
    op = "<=" if leq else ">"
    return f'feat["{feature}"] {op} {threshold:.6g}'


def _rules_to_python(
    buy_rules:  list[dict],
    sell_rules: list[dict],
    symbol:     str,
    generated:  str,
) -> str:
    """Render extracted rules as a complete Python strategy module."""

    def _rule_block(rules: list[dict], indent: int = 8) -> str:
        if not rules:
            return " " * indent + "pd.Series(False, index=df.index)"
        pad = " " * indent
        parts = []
        for r in rules:
            conds = " & ".join(f'({_condition_to_code(*c)})' for c in r["raw_conditions"])
            comment = f"# conf={r['confidence']:.0%}  cov={r['coverage']:.2%}"
            parts.append(f"{pad}({conds})  {comment}")
        return " |\n".join(parts)

    buy_block  = _rule_block(buy_rules)
    sell_block = _rule_block(sell_rules)

    buy_rules_repr  = json.dumps([r["conditions"] for r in buy_rules],  indent=8)
    sell_rules_repr = json.dumps([r["conditions"] for r in sell_rules], indent=8)

    code = f'''\
"""
research/strategies/ml_rules.py — Strategy 6: ML-Extracted Rules
Auto-generated by research/ml_rule_extractor.py on {generated}.
Symbol: {symbol}

Do not edit manually — re-run ml_rule_extractor.py to regenerate.

Buy rules  ({len(buy_rules)}):
{textwrap.indent(buy_rules_repr, "    ")}

Sell rules ({len(sell_rules)}):
{textwrap.indent(sell_rules_repr, "    ")}
"""

import pandas as pd
import pandas_ta as ta

from research.ml_trainer import build_features

NAME = "ml_rules"

PARAMS = {{
    "source_symbol": "{symbol}",
    "tree_depth":    {EXTRACTOR_TREE_DEPTH},
    "n_buy_rules":   {len(buy_rules)},
    "n_sell_rules":  {len(sell_rules)},
}}


def generate_signals(df: pd.DataFrame, params: dict = PARAMS) -> tuple[pd.Series, pd.Series]:
    """
    Entry / exit signals derived from decision-tree rules extracted from the
    Random Forest model.  Rules operate on the same feature set as the ML models.
    """
    feat = build_features(df)

    entries = (
{buy_block}
    ).fillna(False)

    exits = (
{sell_block}
    ).fillna(False)

    return entries, exits


def describe() -> dict:
    return {{
        "name": NAME,
        "description": "Buy/sell signals from decision-tree rules extracted from the Random Forest model.",
        "params": PARAMS,
    }}
'''
    return code


# ── Main extraction pipeline ───────────────────────────────────────────────────

def extract(symbols: list[str] | None = None):
    target_symbols = symbols or SYMBOLS

    for symbol in target_symbols:
        log.info("=== Extracting rules for %s ===", symbol)

        try:
            rf, scaler, feature_names = _load_artefacts(symbol)
        except FileNotFoundError as e:
            log.error("%s", e)
            continue

        # Build features + labels on full dataset
        df       = load_cached(symbol, TIMEFRAME)
        features = build_features(df)
        labels   = label_candles(df)

        valid_mask = features.notna().all(axis=1) & labels.notna()
        X = features[valid_mask].values
        y = labels[valid_mask].values
        X_scaled = scaler.transform(X)

        # Use RF predictions as surrogate targets (reflects what the ensemble learned)
        y_rf = rf.predict(X_scaled)

        tree = _fit_surrogate_tree(X_scaled, y_rf, n_samples=len(y_rf))
        rules = _extract_rules(tree, feature_names, scaler, n_samples=len(y_rf))

        buy_rules  = [r for r in rules if r["label"] == LABEL_BUY]
        sell_rules = [r for r in rules if r["label"] == LABEL_SELL]

        log.info("Extracted %d buy rules and %d sell rules.", len(buy_rules), len(sell_rules))
        for r in buy_rules:
            log.info("  BUY  conf=%.0f%%  cov=%.2f%%  %s", r["confidence"]*100, r["coverage"]*100, r["conditions"])
        for r in sell_rules:
            log.info("  SELL conf=%.0f%%  cov=%.2f%%  %s", r["confidence"]*100, r["coverage"]*100, r["conditions"])

        generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        code = _rules_to_python(buy_rules, sell_rules, symbol, generated)

        out_path = os.path.join(os.path.dirname(__file__), "strategies", "ml_rules.py")
        with open(out_path, "w") as f:
            f.write(code)
        log.info("Written: %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ML rules into ml_rules.py")
    parser.add_argument("--symbol", type=str, help="Symbol to extract rules for, e.g. BTC/USDT")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else None
    extract(symbols=symbols)
