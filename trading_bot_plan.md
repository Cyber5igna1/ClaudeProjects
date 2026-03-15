# Trading Bot — Project Planning Document

## Overview

A profit-focused, fully automated cryptocurrency trading bot that runs on a local virtual machine. The system includes a **research layer** (backtesting multiple rule-based and ML strategies), an **execution layer** (live trading with the best validated strategy), and a **web dashboard** (visual comparison of strategies and live monitoring).

**Budget:** $50 USD (live trading capital)  
**Exchange:** Binance (Bybit as fallback)  
**Target market:** Crypto spot — BTC/USDT and ETH/USDT  
**Infrastructure:** Local virtual machine (Linux)  
**Primary goal:** Generate real profit with a fully automated system checked weekly

---

## Objectives

1. Validate multiple trading strategies against historical data before risking any capital
2. Use ML to discover new rules that can be tested alongside hand-crafted ones
3. Deploy only the best-performing, validated strategy for live trading
4. Monitor everything through a single web dashboard
5. Re-evaluate strategies monthly and update the live bot if a better one emerges

---

## Architecture

The system is split into two clearly separated layers:

### Research Layer
Responsible for backtesting, ML training, rule extraction, and strategy comparison. Runs offline (or on a schedule). Never touches live money.

### Execution Layer
Responsible for live trading using exactly one validated strategy at a time. Receives its strategy configuration from the research layer output.

### Dashboard Layer
FastAPI backend + Chart.js frontend. Reads from SQLite. Presents both research results and live trading status.

---

## Project Structure

```
trading-bot/
├── data/
│   ├── fetcher.py              # Download OHLCV data via ccxt
│   └── cache.db                # SQLite OHLCV cache (avoid re-fetching)
│
├── research/
│   ├── backtest_runner.py      # Run all strategies, collect metrics
│   ├── ml_trainer.py           # Train Random Forest + XGBoost
│   ├── ml_rule_extractor.py    # Extract interpretable rules from trained models
│   ├── strategy_compare.py     # Generate ranked comparison report → SQLite
│   └── strategies/
│       ├── rsi_ema.py          # Strategy 1
│       ├── macd_bb.py          # Strategy 2
│       ├── momentum.py         # Strategy 3
│       ├── triple_ema.py       # Strategy 4
│       ├── rsi_divergence.py   # Strategy 5
│       └── ml_rules.py         # Strategy 6 (ML-extracted rules)
│
├── execution/
│   ├── bot.py                  # Main trading loop
│   ├── exchange.py             # ccxt wrapper (orders, balance, prices)
│   ├── risk.py                 # Position sizing, kill switch, stop-loss
│   └── notifier.py             # Telegram Bot API alerts
│
├── dashboard/
│   ├── server.py               # FastAPI app
│   ├── templates/              # Jinja2 HTML templates
│   └── static/                 # Chart.js and CSS
│
├── database/
│   └── results.db              # SQLite: backtest results + live trade log
│
├── config.py                   # All parameters and env var loading
├── requirements.txt
└── README.md
```

---

## Strategies to Test (8 Total)

### Rule-Based (6)

| # | Name | Logic Summary |
|---|------|---------------|
| 1 | RSI + EMA | Buy when RSI(14) < 35 and price > 200 EMA. Sell when RSI > 60 or TP/SL hit |
| 2 | MACD + Bollinger Bands | Buy when MACD crosses up and price touches lower BB. Sell on upper BB or MACD cross down |
| 3 | Momentum (ROC + Volume) | Buy when Rate of Change is positive and volume spikes above average |
| 4 | Triple EMA Crossover | Buy when EMA9 > EMA21 > EMA50. Sell when EMA9 crosses below EMA21 |
| 5 | RSI Divergence | Buy on bullish divergence (price lower low, RSI higher low). Sell on bearish divergence |
| 6 | ML-Extracted Rules | Decision tree paths from trained ML model converted into explicit if/then rules |

### ML Models (2)

| # | Name | Output |
|---|------|--------|
| 7 | Random Forest Classifier | Predicts Buy / Sell / Hold from technical features |
| 8 | XGBoost Classifier | Same features as RF, compared for accuracy and profitability |

Both ML models serve a dual purpose:
- Their signals are backtested directly (as Strategy 7 and 8)
- Their decision paths are extracted and converted to Strategy 6 (ML-extracted rules)

---

## Tech Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| Language | Python 3.11+ | Everything |
| Exchange connectivity | `ccxt` | Unified API for Binance/Bybit, avoids rewrite on exchange switch |
| Backtesting | `vectorbt` | Fast vectorized backtesting, handles multiple strategies in parallel |
| ML | `scikit-learn`, `XGBoost` | Train classifiers on technical features |
| Rule extraction | `sklearn` tree rules or `rulefit` | Convert ML model logic to human-readable if/then rules |
| Technical indicators | `pandas-ta` or `ta-lib` | RSI, EMA, MACD, Bollinger Bands, etc. |
| Data storage | SQLite (via `sqlite3`) | OHLCV cache + backtest results + live trade log |
| Web backend | `FastAPI` | REST API serving dashboard data |
| Web frontend | Chart.js + Jinja2 HTML | Visual dashboard, no build toolchain needed |
| Alerts | Telegram Bot API | Free, real-time notifications for trades and errors |
| Config management | `python-dotenv` | Keep API keys out of code |

---

## Dashboard Pages

### 1. Strategy Comparison
- Table of all 8 strategies ranked by Sharpe Ratio
- Columns: Strategy name, ROI %, Sharpe Ratio, Max Drawdown, Win Rate, Total Trades
- Bar chart comparing key metrics side by side

### 2. Strategy Detail
- Select any strategy from a dropdown
- Equity curve over backtest period
- Trade markers overlaid on price chart
- Trade log table (entry, exit, P&L per trade)

### 3. ML Insights
- Feature importance bar chart (which indicators matter most)
- Extracted rules from ML model displayed as readable conditions
- Model accuracy, precision, recall table

### 4. Live Monitor
- Current active strategy name and parameters
- Current position (open/closed, entry price, current P&L)
- Recent trade history
- Account balance over time
- Kill switch status (active/triggered)
- Telegram alert log

---

## Risk Management (Non-Negotiable)

| Rule | Value |
|------|-------|
| Capital per trade | 80% of available balance (one position at a time) |
| Stop-loss | −1.5% from entry |
| Take-profit | +2.0% from entry |
| Kill switch | Halt all trading if balance drops below $40 (−20%) |
| API key permissions | Trade only — withdrawal permissions must be disabled |
| Max open positions | 1 at a time |
| Leverage | None — spot trading only |

---

## Build Phases

### Phase 1 — Data Pipeline
**File:** `data/fetcher.py`  
**Goal:** Fetch and cache 2 years of BTC/USDT and ETH/USDT 1h OHLCV data from Binance via `ccxt`. Store in SQLite to avoid re-downloading. Include a refresh mechanism for new candles.  
**Expected output:** `data/cache.db` populated with OHLCV tables, accessible by all other modules.

---

### Phase 2 — Rule-Based Strategy Backtesting
**Files:** `research/strategies/*.py`, `research/backtest_runner.py`  
**Goal:** Implement all 5 rule-based strategies (Strategy 1–5) as `vectorbt` portfolio objects. `backtest_runner.py` runs all of them on the cached data and writes performance metrics to `results.db`.  
**Metrics to capture:** Total return %, Sharpe Ratio, Max Drawdown %, Win Rate %, Total Trades, Avg trade duration.  
**Expected output:** `results.db` table `backtest_results` with one row per strategy.

---

### Phase 3 — ML Pipeline
**Files:** `research/ml_trainer.py`, `research/ml_rule_extractor.py`, `research/strategies/ml_rules.py`  
**Goal:**  
1. Engineer features from OHLCV data: RSI, MACD, EMA slopes, volume z-score, Bollinger Band position, etc.  
2. Label each candle: Buy (next candle up >0.5%), Sell (down >0.5%), Hold.  
3. Train Random Forest and XGBoost classifiers. Evaluate with walk-forward validation (not simple train/test split — avoids lookahead bias).  
4. Extract decision tree rules from the best model using `sklearn` `export_text` or `rulefit`.  
5. Implement extracted rules as Strategy 6 and backtest with `vectorbt`.  
**Expected output:** Trained model files (`.pkl`), feature importance data in `results.db`, Strategy 6 backtest results.

---

### Phase 4 — Dashboard
**Files:** `dashboard/server.py`, `dashboard/templates/`, `dashboard/static/`  
**Goal:** FastAPI app with 4 pages described above. Reads exclusively from `results.db` and the exchange API (for live data). No business logic in the dashboard — it only displays.  
**Expected output:** Running dashboard at `http://localhost:8000`.

---

### Phase 5 — Execution Bot
**Files:** `execution/bot.py`, `execution/exchange.py`, `execution/risk.py`, `execution/notifier.py`  
**Goal:** Main loop that runs every hour (on candle close), fetches latest OHLCV, evaluates the active strategy's entry/exit conditions, places orders via `ccxt`, enforces risk rules, logs all trades to `results.db`, and sends Telegram alerts.  
**Config-driven:** The active strategy is set in `config.py` — changing it requires no code change.  
**Paper trading first:** The bot must support a `PAPER_TRADE=true` env var that simulates orders without hitting the exchange.  
**Expected output:** Bot running on Binance testnet in paper mode for 2–4 weeks before going live.

---

### Phase 6 — Feedback Loop (Ongoing)
**Cadence:** Monthly  
**Process:**  
1. Run `data/fetcher.py` to refresh OHLCV cache  
2. Run `research/backtest_runner.py` on all strategies including any new ones  
3. Re-train ML models with fresh data  
4. Review dashboard Strategy Comparison page  
5. If a different strategy outperforms the current live one significantly (Sharpe > 20% better), update `config.py` and restart the bot  
**Goal:** Ensure the bot stays adapted to current market conditions.

---

## Configuration (`config.py`)

All sensitive values must come from environment variables (`.env` file, never committed to git).

```python
# Exchange
EXCHANGE = "binance"                  # or "bybit"
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

# Trading
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
CAPITAL_ALLOCATION = 0.80             # 80% of balance per trade
STOP_LOSS_PCT = 0.015                 # 1.5%
TAKE_PROFIT_PCT = 0.020              # 2.0%
KILL_SWITCH_BALANCE = 40.0           # USD — halt if balance drops below this

# Active strategy (change without code edit)
ACTIVE_STRATEGY = "rsi_ema"          # Options: rsi_ema, macd_bb, momentum, triple_ema, rsi_divergence, ml_rules, rf_classifier, xgb_classifier

# Paper trading
PAPER_TRADE = os.getenv("PAPER_TRADE", "true").lower() == "true"

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Paths
DB_PATH = "database/results.db"
CACHE_DB_PATH = "data/cache.db"
```

---

## Database Schema

### Table: `ohlcv_cache`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | |
| symbol | TEXT | e.g. BTC/USDT |
| timeframe | TEXT | e.g. 1h |
| timestamp | INTEGER | Unix ms |
| open, high, low, close, volume | REAL | OHLCV values |

### Table: `backtest_results`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | |
| strategy_name | TEXT | |
| symbol | TEXT | |
| run_date | TEXT | ISO date |
| total_return_pct | REAL | |
| sharpe_ratio | REAL | |
| max_drawdown_pct | REAL | |
| win_rate_pct | REAL | |
| total_trades | INTEGER | |
| params_json | TEXT | JSON of strategy parameters |

### Table: `live_trades`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | |
| strategy_name | TEXT | |
| symbol | TEXT | |
| side | TEXT | buy / sell |
| entry_price | REAL | |
| exit_price | REAL | |
| quantity | REAL | |
| pnl_usd | REAL | |
| entry_time | TEXT | ISO datetime |
| exit_time | TEXT | ISO datetime |
| paper_trade | INTEGER | 1 = paper, 0 = live |

### Table: `ml_feature_importance`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | |
| model_name | TEXT | |
| run_date | TEXT | |
| feature_name | TEXT | |
| importance | REAL | |

---

## Important Notes for Implementation

### Avoid Lookahead Bias
When backtesting, never use future data to make decisions. `vectorbt` handles this correctly when using `close.shift(-1)` for entry execution — always verify signals are based on closed candles only.

### Walk-Forward Validation for ML
Do not use a simple train/test split for ML. Use walk-forward (rolling window) cross-validation to simulate real deployment. Example: train on months 1–12, test on 13, then train on 1–13, test on 14, etc.

### Fee Simulation
Always include 0.1% round-trip fees in backtests. Without this, results are misleading for small accounts. In `vectorbt`: `fees=0.001`.

### Binance Testnet
Before running live, use Binance Spot Testnet (https://testnet.binance.vision). Requires separate API keys. Set `PAPER_TRADE=true` and point `ccxt` to the testnet URL.

### Rate Limits
Binance enforces strict rate limits. The bot loop runs once per closed 1h candle (not continuously). Add `time.sleep()` between API calls in fetcher to avoid bans.

### API Key Security
- Create a dedicated Binance API key for this bot
- Enable: Spot trading only
- Disable: Withdrawals, futures, margin
- Whitelist your local machine's IP address

---

## Dependencies (`requirements.txt`)

```
ccxt>=4.0.0
vectorbt>=0.26.0
scikit-learn>=1.3.0
xgboost>=2.0.0
rulefit>=0.3.0
pandas>=2.0.0
pandas-ta>=0.3.14b
numpy>=1.24.0
fastapi>=0.104.0
uvicorn>=0.24.0
jinja2>=3.1.0
python-dotenv>=1.0.0
httpx>=0.25.0
sqlite3  # stdlib
```

---

## Step-by-Step Build Order for Claude Code

1. **Create project skeleton** — all folders and empty files with docstrings
2. **`config.py`** — env var loading, all constants in one place
3. **`data/fetcher.py`** — ccxt OHLCV fetch + SQLite cache + refresh logic
4. **`research/strategies/rsi_ema.py`** — first strategy as a template for the others
5. **`research/strategies/*.py`** — remaining 4 rule-based strategies following the same interface
6. **`research/backtest_runner.py`** — load all strategies, run vectorbt, write to results.db
7. **`research/ml_trainer.py`** — feature engineering, RF + XGBoost training, walk-forward validation
8. **`research/ml_rule_extractor.py`** — extract rules, write to research/strategies/ml_rules.py
9. **`research/strategy_compare.py`** — generate ranked report, print and write to DB
10. **`database/` schema init script** — create all tables on first run
11. **`dashboard/server.py`** — FastAPI routes for all 4 pages
12. **`dashboard/templates/` + `static/`** — HTML + Chart.js visualizations
13. **`execution/exchange.py`** — ccxt wrapper with paper trade mode
14. **`execution/risk.py`** — position sizing, stop-loss, kill switch logic
15. **`execution/notifier.py`** — Telegram alerts for trade events and errors
16. **`execution/bot.py`** — main loop tying everything together
17. **End-to-end test** — run full pipeline in paper mode on testnet, verify dashboard

---

*Document generated from planning session. Ready to begin Phase 1: Data Pipeline.*
