"""
execution/exchange.py — ccxt wrapper.
Handles order placement, balance queries, and price fetching.
Supports paper trade mode (simulates orders without hitting the exchange).
"""

import logging
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN

import ccxt

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    EXCHANGE, API_KEY, API_SECRET,
    PAPER_TRADE, USE_TESTNET, BINANCE_TESTNET_URL,
    SYMBOL, TIMEFRAME,
)

log = logging.getLogger(__name__)


# ── Paper trade state (in-memory, reset on restart) ───────────────────────────

class _PaperState:
    """Minimal in-memory paper trading ledger."""

    def __init__(self, starting_balance: float = 100.0):
        self.balance_usdt: float = starting_balance
        self.position: dict | None = None      # {symbol, side, qty, entry_price, entry_time}
        self.order_id_counter: int = 1

    def next_order_id(self) -> str:
        oid = f"PAPER-{self.order_id_counter:06d}"
        self.order_id_counter += 1
        return oid


_paper = _PaperState()


# ── Exchange factory ───────────────────────────────────────────────────────────

_PLACEHOLDER_KEYS = {None, "", "your_binance_api_key", "your_binance_api_secret"}


def _build_exchange() -> ccxt.Exchange:
    exchange_class = getattr(ccxt, EXCHANGE)
    has_keys = API_KEY not in _PLACEHOLDER_KEYS and API_SECRET not in _PLACEHOLDER_KEYS
    params = {
        "enableRateLimit": True,
        "options": {
            "fetchCurrencies": False,
            "defaultType": "spot",
            "fetchMarkets": {"types": ["spot"]},  # skip margin/futures private endpoints
        },
    }
    if has_keys:
        params["apiKey"] = API_KEY
        params["secret"] = API_SECRET
    if USE_TESTNET and EXCHANGE == "binance":
        params["urls"] = {
            "api": {
                "public":  BINANCE_TESTNET_URL,
                "private": BINANCE_TESTNET_URL,
            }
        }
    ex = exchange_class(params)
    ex.load_markets()
    return ex


_exchange: ccxt.Exchange | None = None


def get_exchange() -> ccxt.Exchange:
    """Return a cached exchange instance (lazy init)."""
    global _exchange
    if _exchange is None:
        _exchange = _build_exchange()
    return _exchange


# ── Price ──────────────────────────────────────────────────────────────────────

def get_price(symbol: str = SYMBOL) -> float:
    """Return the latest mid-price for a symbol."""
    ex = get_exchange()
    ticker = ex.fetch_ticker(symbol)
    return float(ticker["last"])


def get_ohlcv(symbol: str = SYMBOL, timeframe: str = TIMEFRAME, limit: int = 300):
    """
    Fetch recent OHLCV candles directly from the exchange (live data).
    Returns a list of [timestamp_ms, open, high, low, close, volume].
    """
    ex = get_exchange()
    return ex.fetch_ohlcv(symbol, timeframe, limit=limit)


# ── Balance ────────────────────────────────────────────────────────────────────

def get_balance(asset: str = "USDT") -> float:
    """Return the free balance for a given asset."""
    if PAPER_TRADE:
        if asset == "USDT":
            return _paper.balance_usdt
        # Return held crypto qty if in a position
        pos = _paper.position
        if pos and pos["symbol"].startswith(asset):
            return pos["qty"]
        return 0.0

    ex = get_exchange()
    balance = ex.fetch_balance()
    return float(balance["free"].get(asset, 0.0))


# ── Quantity precision ─────────────────────────────────────────────────────────

def _round_qty(symbol: str, qty: float) -> float:
    """Round quantity down to the exchange's allowed precision for a symbol."""
    if PAPER_TRADE:
        return round(qty, 6)
    ex = get_exchange()
    market = ex.market(symbol)
    precision = market.get("precision", {}).get("amount", 6)
    factor = Decimal(10) ** -int(precision)
    return float(Decimal(str(qty)).quantize(factor, rounding=ROUND_DOWN))


# ── Orders ─────────────────────────────────────────────────────────────────────

def place_market_buy(symbol: str, usdt_amount: float) -> dict:
    """
    Place a market buy order for `usdt_amount` of USDT worth of `symbol`.
    Returns an order dict with: id, symbol, side, price, qty, timestamp, paper.
    """
    price = get_price(symbol)
    qty   = _round_qty(symbol, usdt_amount / price)

    if qty <= 0:
        raise ValueError(f"Computed qty={qty} is too small to place an order.")

    if PAPER_TRADE:
        if usdt_amount > _paper.balance_usdt:
            raise ValueError(
                f"Paper trade: insufficient balance "
                f"(need {usdt_amount:.2f}, have {_paper.balance_usdt:.2f})"
            )
        _paper.balance_usdt -= usdt_amount
        _paper.position = {
            "symbol":      symbol,
            "side":        "buy",
            "qty":         qty,
            "entry_price": price,
            "entry_time":  _now(),
        }
        order = {
            "id":        _paper.next_order_id(),
            "symbol":    symbol,
            "side":      "buy",
            "price":     price,
            "qty":       qty,
            "timestamp": _now(),
            "paper":     True,
        }
        log.info("[PAPER] BUY  %s qty=%.6f @ %.2f  (cost=%.2f USDT)", symbol, qty, price, usdt_amount)
        return order

    # Live order
    ex = get_exchange()
    raw = ex.create_market_buy_order(symbol, qty)
    filled_price = float(raw.get("average") or raw.get("price") or price)
    order = {
        "id":        str(raw["id"]),
        "symbol":    symbol,
        "side":      "buy",
        "price":     filled_price,
        "qty":       float(raw.get("filled") or qty),
        "timestamp": _now(),
        "paper":     False,
    }
    log.info("[LIVE]  BUY  %s qty=%.6f @ %.2f", symbol, order["qty"], order["price"])
    return order


def place_market_sell(symbol: str, qty: float) -> dict:
    """
    Place a market sell order for `qty` units of `symbol`.
    Returns an order dict with: id, symbol, side, price, qty, timestamp, paper.
    """
    qty = _round_qty(symbol, qty)
    if qty <= 0:
        raise ValueError(f"Computed qty={qty} is too small to place an order.")

    price = get_price(symbol)

    if PAPER_TRADE:
        proceeds = qty * price
        _paper.balance_usdt += proceeds
        _paper.position = None
        order = {
            "id":        _paper.next_order_id(),
            "symbol":    symbol,
            "side":      "sell",
            "price":     price,
            "qty":       qty,
            "timestamp": _now(),
            "paper":     True,
        }
        log.info("[PAPER] SELL %s qty=%.6f @ %.2f  (proceeds=%.2f USDT)", symbol, qty, price, proceeds)
        return order

    # Live order
    ex = get_exchange()
    raw = ex.create_market_sell_order(symbol, qty)
    filled_price = float(raw.get("average") or raw.get("price") or price)
    order = {
        "id":        str(raw["id"]),
        "symbol":    symbol,
        "side":      "sell",
        "price":     filled_price,
        "qty":       float(raw.get("filled") or qty),
        "timestamp": _now(),
        "paper":     False,
    }
    log.info("[LIVE]  SELL %s qty=%.6f @ %.2f", symbol, order["qty"], order["price"])
    return order


# ── Position state ─────────────────────────────────────────────────────────────

def get_open_position(symbol: str = SYMBOL) -> dict | None:
    """
    Return the current open position for a symbol, or None.
    In paper mode: reads from in-memory state.
    In live mode:  queries account positions / open orders.

    Returns dict with keys: symbol, side, qty, entry_price, entry_time — or None.
    """
    if PAPER_TRADE:
        pos = _paper.position
        if pos and pos["symbol"] == symbol:
            return pos
        return None

    # Live: check free balance for the base asset
    ex = get_exchange()
    base = symbol.split("/")[0]   # e.g. "BTC" from "BTC/USDT"
    balance = ex.fetch_balance()
    qty = float(balance["free"].get(base, 0.0))
    if qty > 0:
        # We have a position; entry price not known from balance alone
        return {
            "symbol":      symbol,
            "side":        "buy",
            "qty":         qty,
            "entry_price": None,
            "entry_time":  None,
        }
    return None


# ── Utilities ──────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def reset_paper_state(starting_balance: float = 100.0):
    """Reset paper trading state (useful for testing)."""
    global _paper
    _paper = _PaperState(starting_balance)
    log.info("Paper state reset. Starting balance: %.2f USDT", starting_balance)
