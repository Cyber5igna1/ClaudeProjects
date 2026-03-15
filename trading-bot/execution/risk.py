"""
execution/risk.py — Risk management.
Handles position sizing (80% capital), stop-loss (1.5%), take-profit (2%),
and the kill switch (halt if balance < $40).

All rules are non-negotiable and enforced before every trade decision.
"""

import logging
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    CAPITAL_ALLOCATION,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    KILL_SWITCH_BALANCE,
    MAX_OPEN_POSITIONS,
)

log = logging.getLogger(__name__)


# ── Kill switch state ──────────────────────────────────────────────────────────

class KillSwitchError(Exception):
    """Raised when the kill switch is triggered. The bot must stop trading."""


_kill_switch_triggered: bool = False
_kill_switch_reason:    str  = ""


def is_kill_switch_active() -> bool:
    return _kill_switch_triggered


def get_kill_switch_reason() -> str:
    return _kill_switch_reason


def _trigger_kill_switch(reason: str):
    global _kill_switch_triggered, _kill_switch_reason
    _kill_switch_triggered = True
    _kill_switch_reason    = reason
    log.critical("KILL SWITCH TRIGGERED: %s", reason)


def reset_kill_switch():
    """Reset the kill switch (for testing only — never call in production)."""
    global _kill_switch_triggered, _kill_switch_reason
    _kill_switch_triggered = False
    _kill_switch_reason    = ""
    log.warning("Kill switch reset.")


# ── Balance check ──────────────────────────────────────────────────────────────

def check_balance(balance_usdt: float) -> None:
    """
    Raise KillSwitchError if the balance is below the minimum threshold.
    Call this before every trade decision.
    """
    if _kill_switch_triggered:
        raise KillSwitchError(f"Kill switch already active: {_kill_switch_reason}")

    if balance_usdt < KILL_SWITCH_BALANCE:
        reason = (
            f"Balance ${balance_usdt:.2f} dropped below kill-switch floor "
            f"${KILL_SWITCH_BALANCE:.2f}"
        )
        _trigger_kill_switch(reason)
        raise KillSwitchError(reason)


# ── Position sizing ────────────────────────────────────────────────────────────

def compute_position_size(balance_usdt: float) -> float:
    """
    Return the USDT amount to deploy on the next trade.

    Rules:
      - Allocate CAPITAL_ALLOCATION (80%) of free balance.
      - Never allocate more than the free balance.
      - Returns 0.0 if the resulting amount is too small to trade (<$1).
    """
    check_balance(balance_usdt)

    amount = balance_usdt * CAPITAL_ALLOCATION
    amount = min(amount, balance_usdt)

    if amount < 1.0:
        log.warning("Position size %.4f USDT is too small to trade.", amount)
        return 0.0

    log.debug("Position size: %.4f USDT (%.0f%% of %.4f)", amount, CAPITAL_ALLOCATION * 100, balance_usdt)
    return round(amount, 4)


# ── Stop-loss / take-profit ────────────────────────────────────────────────────

def compute_stop_loss(entry_price: float, side: str = "buy") -> float:
    """
    Return the stop-loss price for a position.

    Long (buy):  stop = entry * (1 - STOP_LOSS_PCT)
    Short (sell): not supported — spot only, no shorts.
    """
    if side != "buy":
        raise ValueError("Only long (buy) positions are supported — spot trading only.")
    return round(entry_price * (1 - STOP_LOSS_PCT), 8)


def compute_take_profit(entry_price: float, side: str = "buy") -> float:
    """
    Return the take-profit price for a position.

    Long (buy):  target = entry * (1 + TAKE_PROFIT_PCT)
    """
    if side != "buy":
        raise ValueError("Only long (buy) positions are supported — spot trading only.")
    return round(entry_price * (1 + TAKE_PROFIT_PCT), 8)


# ── Exit signal evaluation ─────────────────────────────────────────────────────

class ExitReason:
    STRATEGY_SIGNAL = "strategy_signal"
    STOP_LOSS       = "stop_loss"
    TAKE_PROFIT     = "take_profit"
    KILL_SWITCH     = "kill_switch"


def should_exit(
    current_price: float,
    entry_price:   float,
    strategy_exit: bool,
    balance_usdt:  float,
) -> tuple[bool, str]:
    """
    Evaluate all exit conditions in priority order.

    Returns (should_exit: bool, reason: str).

    Priority:
        1. Kill switch (balance floor)
        2. Stop-loss
        3. Take-profit
        4. Strategy signal
    """
    # 1 — Kill switch
    try:
        check_balance(balance_usdt)
    except KillSwitchError as e:
        return True, ExitReason.KILL_SWITCH

    # 2 — Stop-loss
    stop_price = compute_stop_loss(entry_price)
    if current_price <= stop_price:
        log.warning(
            "Stop-loss hit: price=%.4f <= stop=%.4f (entry=%.4f, loss=%.2f%%)",
            current_price, stop_price, entry_price,
            (current_price - entry_price) / entry_price * 100,
        )
        return True, ExitReason.STOP_LOSS

    # 3 — Take-profit
    tp_price = compute_take_profit(entry_price)
    if current_price >= tp_price:
        log.info(
            "Take-profit hit: price=%.4f >= target=%.4f (entry=%.4f, gain=%.2f%%)",
            current_price, tp_price, entry_price,
            (current_price - entry_price) / entry_price * 100,
        )
        return True, ExitReason.TAKE_PROFIT

    # 4 — Strategy signal
    if strategy_exit:
        log.info("Strategy exit signal fired at price=%.4f (entry=%.4f)", current_price, entry_price)
        return True, ExitReason.STRATEGY_SIGNAL

    return False, ""


# ── Entry guard ────────────────────────────────────────────────────────────────

def can_enter(
    has_open_position: bool,
    balance_usdt:      float,
    strategy_entry:    bool,
) -> tuple[bool, str]:
    """
    Check all pre-conditions before entering a new position.

    Returns (can_enter: bool, reason: str).
    """
    if not strategy_entry:
        return False, "no_signal"

    # Kill switch
    if _kill_switch_triggered:
        return False, f"kill_switch:{_kill_switch_reason}"

    # Max positions
    if has_open_position:
        return False, "already_in_position"

    # Balance check
    try:
        check_balance(balance_usdt)
    except KillSwitchError as e:
        return False, f"kill_switch:{e}"

    # Minimum viable position
    size = compute_position_size(balance_usdt)
    if size <= 0:
        return False, "insufficient_balance"

    return True, "ok"


# ── P&L calculation ────────────────────────────────────────────────────────────

def compute_pnl(entry_price: float, exit_price: float, qty: float, fees: float = 0.001) -> float:
    """
    Compute net P&L in USD after round-trip fees.

    pnl = (exit_price - entry_price) * qty - fees * (entry_price + exit_price) * qty
    """
    gross = (exit_price - entry_price) * qty
    fee_cost = fees * (entry_price + exit_price) * qty
    return round(gross - fee_cost, 6)


# ── Status summary ─────────────────────────────────────────────────────────────

def status_summary(balance_usdt: float, open_position: dict | None) -> dict:
    """Return a dict snapshot of current risk state for logging/dashboard."""
    summary = {
        "balance_usdt":          round(balance_usdt, 4),
        "kill_switch_active":    _kill_switch_triggered,
        "kill_switch_reason":    _kill_switch_reason,
        "kill_switch_floor":     KILL_SWITCH_BALANCE,
        "capital_allocation_pct": CAPITAL_ALLOCATION * 100,
        "stop_loss_pct":         STOP_LOSS_PCT * 100,
        "take_profit_pct":       TAKE_PROFIT_PCT * 100,
        "open_position":         open_position,
        "timestamp":             datetime.now(timezone.utc).isoformat(),
    }
    if open_position and open_position.get("entry_price"):
        ep = open_position["entry_price"]
        summary["stop_loss_price"]   = compute_stop_loss(ep)
        summary["take_profit_price"] = compute_take_profit(ep)
    return summary
