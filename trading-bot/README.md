# Trading Bot

A fully automated cryptocurrency trading bot with backtesting, ML strategy discovery, and live execution.

## Structure

- `data/` — OHLCV fetcher and SQLite cache
- `research/` — Backtesting, ML training, strategy comparison
- `execution/` — Live trading loop, risk management, Telegram alerts
- `dashboard/` — FastAPI web dashboard
- `database/` — SQLite results database
- `config.py` — All configuration and env var loading

## Setup

```bash
cp .env.example .env
# Fill in your API keys in .env
pip install -r requirements.txt
```

## Usage

See `trading_bot_plan.md` for full build phases and architecture.
