#!/usr/bin/env bash
# setup.sh — Install dependencies and run end-to-end validation.
# Run once after cloning: bash setup.sh

set -e
cd "$(dirname "$0")"

# Install pip only if missing (requires sudo — run manually if this fails)
if ! python3 -m pip --version &>/dev/null; then
  echo "=== Installing system pip (requires sudo) ==="
  sudo apt-get update -q
  sudo apt-get install -y python3-pip python3-venv
fi

echo "=== Creating virtual environment ==="
python3 -m venv .venv
source .venv/bin/activate

echo "=== Installing Python dependencies ==="
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "=== Running validation ==="
python validate.py

echo ""
echo "Setup complete. Activate the venv with:"
echo "  source .venv/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Copy .env.example to .env and fill in your API keys"
echo "  2. python data/fetcher.py            # download 2 years of OHLCV data"
echo "  3. python research/backtest_runner.py # backtest all strategies"
echo "  4. python research/ml_trainer.py      # train ML models"
echo "  5. python research/ml_rule_extractor.py"
echo "  6. python research/strategy_compare.py"
echo "  7. uvicorn dashboard.server:app --port 8000"
echo "  8. python execution/bot.py            # paper trade on testnet"
