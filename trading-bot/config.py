"""
config.py — Central configuration for the trading bot.
All sensitive values are loaded from environment variables (.env file).
Never hardcode secrets here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Exchange ───────────────────────────────────────────────────────────────────
EXCHANGE = os.getenv("EXCHANGE", "binance")
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

# ── Trading ────────────────────────────────────────────────────────────────────
SYMBOLS = ["BTC/USDT", "ETH/USDT"]
SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
TIMEFRAME = "1h"
CAPITAL_ALLOCATION = 0.80       # 80% of available balance per trade
STOP_LOSS_PCT = 0.015           # 1.5% stop-loss from entry
TAKE_PROFIT_PCT = 0.020         # 2.0% take-profit from entry
KILL_SWITCH_BALANCE = 40.0      # USD — halt all trading if balance drops below this
MAX_OPEN_POSITIONS = 1
FEES = 0.001                    # 0.1% per trade (used in backtests)

# ── Active strategy ────────────────────────────────────────────────────────────
# Change this value to switch strategies without editing any other file.
# Options: rsi_ema | macd_bb | momentum | triple_ema | rsi_divergence
#          ml_rules | rf_classifier | xgb_classifier
ACTIVE_STRATEGY = os.getenv("ACTIVE_STRATEGY", "rsi_ema")

# ── Paper trading ──────────────────────────────────────────────────────────────
PAPER_TRADE = os.getenv("PAPER_TRADE", "true").lower() == "true"

# ── Telegram ───────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ── Paths ──────────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "database", "results.db")
CACHE_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "cache.db")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "research", "models")

# ── Data ───────────────────────────────────────────────────────────────────────
OHLCV_HISTORY_DAYS = 730        # 2 years of historical data
OHLCV_LIMIT_PER_REQUEST = 1000  # Binance max candles per API call

# ── ML ─────────────────────────────────────────────────────────────────────────
ML_LABEL_THRESHOLD = 0.005      # 0.5% move to classify as Buy or Sell
ML_WALK_FORWARD_TRAIN_MONTHS = 12
ML_WALK_FORWARD_TEST_MONTHS = 1

# ── Binance testnet ────────────────────────────────────────────────────────────
BINANCE_TESTNET_URL = "https://testnet.binance.vision/api"
USE_TESTNET = os.getenv("USE_TESTNET", "false").lower() == "true"
