"""
execution/bot.py — Main trading loop.
Runs every hour on candle close, evaluates the active strategy,
places orders via ccxt, enforces risk rules, logs trades, sends alerts.
"""
