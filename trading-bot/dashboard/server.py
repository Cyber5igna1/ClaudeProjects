"""
dashboard/server.py — FastAPI application.
Serves 4 pages: Strategy Comparison, Strategy Detail, ML Insights, Live Monitor.
Reads exclusively from results.db and the exchange API for live data.
No business logic here — display only.

Run:
    uvicorn dashboard.server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import json
import sqlite3
import logging

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DB_PATH, ACTIVE_STRATEGY, SYMBOL, PAPER_TRADE
from database.init_db import get_conn

log = logging.getLogger(__name__)

app = FastAPI(title="Trading Bot Dashboard")

_dir = os.path.dirname(__file__)
app.mount("/static", StaticFiles(directory=os.path.join(_dir, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(_dir, "templates"))
templates.env.globals["enumerate"] = enumerate


# ── Helpers ────────────────────────────────────────────────────────────────────

def _rows(query: str, params: tuple = ()) -> list[dict]:
    """Execute a SELECT and return rows as a list of dicts."""
    with get_conn() as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(query, params)
        return [dict(r) for r in cur.fetchall()]


# ── Page 1 — Strategy Comparison ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
@app.get("/comparison", response_class=HTMLResponse)
async def page_comparison(request: Request):
    rows = _rows("""
        SELECT r.strategy_name, r.symbol, r.run_date,
               r.total_return_pct, r.sharpe_ratio, r.max_drawdown_pct,
               r.win_rate_pct, r.total_trades
        FROM backtest_results r
        INNER JOIN (
            SELECT strategy_name, symbol, MAX(run_date) AS latest
            FROM backtest_results GROUP BY strategy_name, symbol
        ) l ON r.strategy_name=l.strategy_name
           AND r.symbol=l.symbol AND r.run_date=l.latest
        ORDER BY r.sharpe_ratio DESC
    """)
    return templates.TemplateResponse("comparison.html", {
        "request": request,
        "strategies": rows,
        "active_strategy": ACTIVE_STRATEGY,
    })


# ── Page 2 — Strategy Detail ───────────────────────────────────────────────────

@app.get("/detail/{strategy_name}", response_class=HTMLResponse)
async def page_detail(request: Request, strategy_name: str, symbol: str = SYMBOL):
    meta = _rows("""
        SELECT * FROM backtest_results
        WHERE strategy_name=? AND symbol=?
        ORDER BY run_date DESC LIMIT 1
    """, (strategy_name, symbol))

    if not meta:
        raise HTTPException(status_code=404, detail="Strategy not found in results")

    trades = _rows("""
        SELECT side, entry_price, exit_price, quantity, pnl_usd,
               entry_time, exit_time, paper_trade
        FROM live_trades
        WHERE strategy_name=? AND symbol=?
        ORDER BY entry_time DESC LIMIT 200
    """, (strategy_name, symbol))

    return templates.TemplateResponse("detail.html", {
        "request":       request,
        "strategy":      meta[0],
        "trades":        trades,
        "strategy_name": strategy_name,
        "symbol":        symbol,
    })


# ── Page 3 — ML Insights ───────────────────────────────────────────────────────

@app.get("/ml", response_class=HTMLResponse)
async def page_ml(request: Request):
    rf_importance = _rows("""
        SELECT feature_name, importance FROM ml_feature_importance
        WHERE model_name='random_forest'
        AND run_date=(SELECT MAX(run_date) FROM ml_feature_importance WHERE model_name='random_forest')
        ORDER BY importance DESC LIMIT 20
    """)

    xgb_importance = _rows("""
        SELECT feature_name, importance FROM ml_feature_importance
        WHERE model_name='xgboost'
        AND run_date=(SELECT MAX(run_date) FROM ml_feature_importance WHERE model_name='xgboost')
        ORDER BY importance DESC LIMIT 20
    """)

    ml_results = _rows("""
        SELECT strategy_name, symbol, total_return_pct, sharpe_ratio,
               max_drawdown_pct, win_rate_pct, total_trades, run_date
        FROM backtest_results
        WHERE strategy_name IN ('rf_classifier','xgb_classifier','ml_rules')
        ORDER BY run_date DESC, sharpe_ratio DESC
    """)

    return templates.TemplateResponse("ml.html", {
        "request":        request,
        "rf_importance":  rf_importance,
        "xgb_importance": xgb_importance,
        "ml_results":     ml_results,
    })


# ── Page 4 — Live Monitor ──────────────────────────────────────────────────────

@app.get("/live", response_class=HTMLResponse)
async def page_live(request: Request):
    recent_trades = _rows("""
        SELECT strategy_name, symbol, side, entry_price, exit_price,
               pnl_usd, entry_time, exit_time, paper_trade
        FROM live_trades
        ORDER BY entry_time DESC LIMIT 50
    """)

    balance_history = _rows("""
        SELECT entry_time AS ts,
               SUM(pnl_usd) OVER (ORDER BY entry_time) AS cumulative_pnl
        FROM live_trades
        ORDER BY entry_time ASC
    """)

    open_position = _rows("""
        SELECT * FROM live_trades
        WHERE exit_time IS NULL OR exit_time=''
        ORDER BY entry_time DESC LIMIT 1
    """)

    return templates.TemplateResponse("live.html", {
        "request":         request,
        "recent_trades":   recent_trades,
        "balance_history": balance_history,
        "open_position":   open_position[0] if open_position else None,
        "active_strategy": ACTIVE_STRATEGY,
        "paper_trade":     PAPER_TRADE,
        "symbol":          SYMBOL,
    })


# ── JSON API endpoints (consumed by Chart.js) ─────────────────────────────────

@app.get("/api/comparison")
async def api_comparison():
    return _rows("""
        SELECT r.strategy_name, r.symbol, r.total_return_pct, r.sharpe_ratio,
               r.max_drawdown_pct, r.win_rate_pct, r.total_trades
        FROM backtest_results r
        INNER JOIN (
            SELECT strategy_name, symbol, MAX(run_date) AS latest
            FROM backtest_results GROUP BY strategy_name, symbol
        ) l ON r.strategy_name=l.strategy_name
           AND r.symbol=l.symbol AND r.run_date=l.latest
        ORDER BY r.sharpe_ratio DESC
    """)


@app.get("/api/trades/{strategy_name}")
async def api_trades(strategy_name: str, symbol: str = SYMBOL):
    return _rows("""
        SELECT side, entry_price, exit_price, pnl_usd, entry_time, exit_time
        FROM live_trades WHERE strategy_name=? AND symbol=?
        ORDER BY entry_time ASC
    """, (strategy_name, symbol))


@app.get("/api/feature_importance/{model_name}")
async def api_feature_importance(model_name: str):
    return _rows("""
        SELECT feature_name, importance FROM ml_feature_importance
        WHERE model_name=?
        AND run_date=(SELECT MAX(run_date) FROM ml_feature_importance WHERE model_name=?)
        ORDER BY importance DESC
    """, (model_name, model_name))


@app.get("/api/balance_history")
async def api_balance_history():
    return _rows("""
        SELECT entry_time AS ts,
               SUM(pnl_usd) OVER (ORDER BY entry_time) AS cumulative_pnl
        FROM live_trades ORDER BY entry_time ASC
    """)


@app.get("/health")
async def health():
    return {"status": "ok", "paper_trade": PAPER_TRADE, "active_strategy": ACTIVE_STRATEGY}
