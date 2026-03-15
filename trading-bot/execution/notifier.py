"""
execution/notifier.py — Telegram Bot API alerts.
Sends real-time notifications for trade entries, exits, errors, and kill switch triggers.

All functions are fire-and-forget — failures are logged but never raise,
so a Telegram outage never stops the bot from trading.
"""

import logging
import os
import sys
from datetime import datetime, timezone

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, PAPER_TRADE, ACTIVE_STRATEGY

log = logging.getLogger(__name__)

_BASE_URL = "https://api.telegram.org/bot{token}/sendMessage"

# Emoji prefixes per alert type
_ICON = {
    "buy":          "🟢",
    "sell":         "🔴",
    "stop_loss":    "🛑",
    "take_profit":  "✅",
    "kill_switch":  "☠️",
    "error":        "⚠️",
    "info":         "ℹ️",
    "start":        "🚀",
}


# ── Core send ──────────────────────────────────────────────────────────────────

def _send(text: str) -> bool:
    """
    POST a message to Telegram. Returns True on success, False on failure.
    Never raises — Telegram errors must not interrupt the trading loop.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.debug("Telegram not configured — skipping notification.")
        return False

    url = _BASE_URL.format(token=TELEGRAM_BOT_TOKEN)
    try:
        resp = httpx.post(
            url,
            json={
                "chat_id":    TELEGRAM_CHAT_ID,
                "text":       text,
                "parse_mode": "HTML",
            },
            timeout=10.0,
        )
        if not resp.is_success:
            log.warning("Telegram send failed: %s %s", resp.status_code, resp.text)
            return False
        return True
    except Exception as e:
        log.warning("Telegram error: %s", e)
        return False


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _mode_tag() -> str:
    return "[PAPER]" if PAPER_TRADE else "[LIVE]"


# ── Trade alerts ───────────────────────────────────────────────────────────────

def alert_buy(symbol: str, price: float, qty: float, usdt_spent: float) -> bool:
    """Alert on a new position opened."""
    text = (
        f"{_ICON['buy']} <b>BUY {_mode_tag()}</b>\n"
        f"Strategy : <code>{ACTIVE_STRATEGY}</code>\n"
        f"Symbol   : <code>{symbol}</code>\n"
        f"Price    : <code>${price:,.4f}</code>\n"
        f"Qty      : <code>{qty:.6f}</code>\n"
        f"Spent    : <code>${usdt_spent:,.2f} USDT</code>\n"
        f"Time     : {_now()}"
    )
    log.info("Telegram: BUY alert sent for %s", symbol)
    return _send(text)


def alert_sell(
    symbol:      str,
    price:       float,
    qty:         float,
    pnl_usd:     float,
    reason:      str,
    entry_price: float,
) -> bool:
    """Alert on a position closed."""
    pct_change = (price - entry_price) / entry_price * 100 if entry_price else 0.0
    icon = _ICON.get(reason, _ICON["sell"])
    pnl_sign = "+" if pnl_usd >= 0 else ""

    text = (
        f"{icon} <b>SELL {_mode_tag()} — {reason.upper().replace('_', ' ')}</b>\n"
        f"Strategy : <code>{ACTIVE_STRATEGY}</code>\n"
        f"Symbol   : <code>{symbol}</code>\n"
        f"Entry    : <code>${entry_price:,.4f}</code>\n"
        f"Exit     : <code>${price:,.4f}</code>  ({pct_change:+.2f}%)\n"
        f"Qty      : <code>{qty:.6f}</code>\n"
        f"P&amp;L  : <code>{pnl_sign}{pnl_usd:,.4f} USD</code>\n"
        f"Time     : {_now()}"
    )
    log.info("Telegram: SELL alert sent for %s (reason=%s, pnl=%.4f)", symbol, reason, pnl_usd)
    return _send(text)


# ── System alerts ──────────────────────────────────────────────────────────────

def alert_kill_switch(reason: str, balance_usdt: float) -> bool:
    """Alert when the kill switch triggers."""
    text = (
        f"{_ICON['kill_switch']} <b>KILL SWITCH TRIGGERED {_mode_tag()}</b>\n"
        f"Reason  : {reason}\n"
        f"Balance : <code>${balance_usdt:,.2f} USDT</code>\n"
        f"Action  : All trading halted.\n"
        f"Time    : {_now()}"
    )
    log.critical("Telegram: Kill switch alert sent.")
    return _send(text)


def alert_error(context: str, error: Exception) -> bool:
    """Alert on unexpected errors in the bot loop."""
    text = (
        f"{_ICON['error']} <b>BOT ERROR {_mode_tag()}</b>\n"
        f"Context : {context}\n"
        f"Error   : <code>{type(error).__name__}: {error}</code>\n"
        f"Time    : {_now()}"
    )
    log.error("Telegram: Error alert sent — %s: %s", context, error)
    return _send(text)


def alert_start(symbol: str, starting_balance: float) -> bool:
    """Alert when the bot loop starts."""
    text = (
        f"{_ICON['start']} <b>BOT STARTED {_mode_tag()}</b>\n"
        f"Strategy : <code>{ACTIVE_STRATEGY}</code>\n"
        f"Symbol   : <code>{symbol}</code>\n"
        f"Balance  : <code>${starting_balance:,.2f} USDT</code>\n"
        f"Time     : {_now()}"
    )
    return _send(text)


def alert_info(message: str) -> bool:
    """Send a plain informational message."""
    text = f"{_ICON['info']} {message}\n<i>{_now()}</i>"
    return _send(text)
