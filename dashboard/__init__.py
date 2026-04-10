"""Lightweight web dashboard (Flask).

Provides a real-time browser view of the trading bot:
  GET /               - Main HTML page with equity curve and tables
  GET /api/positions  - JSON list of open positions
  GET /api/trades     - JSON list of recent trades
  GET /api/performance - JSON performance metrics
  GET /api/signals    - JSON recent signals feed

The dashboard is started in a background daemon thread so it does not
block the trading loop.  Enable it with DASHBOARD_ENABLED=true in .env
and browse to http://127.0.0.1:8080 (or the configured host/port).
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from alerts import get_logger
from config import DashboardConfig

logger = get_logger("dashboard")

# ---------------------------------------------------------------------------
# Signal feed buffer (ring buffer of the last 100 signals)
# ---------------------------------------------------------------------------
_MAX_SIGNALS = 100
_signal_feed: List[Dict[str, Any]] = []
_signal_lock = threading.Lock()


def push_signal(symbol: str, action: str, price: float, strategy: str, reason: str) -> None:
    """Add a signal to the in-memory feed (called from the trading engine)."""
    entry = {
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "symbol": symbol,
        "action": action,
        "price": price,
        "strategy": strategy,
        "reason": reason,
    }
    with _signal_lock:
        _signal_feed.append(entry)
        if len(_signal_feed) > _MAX_SIGNALS:
            _signal_feed.pop(0)


# ---------------------------------------------------------------------------
# Dashboard server
# ---------------------------------------------------------------------------

_INLINE_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>AlpacaTrader Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <style>
    body{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:1rem;}
    h1{color:#38bdf8;} h2{color:#7dd3fc;border-bottom:1px solid #1e3a5f;padding-bottom:.3rem;}
    table{border-collapse:collapse;width:100%;margin-bottom:1.5rem;}
    th{background:#1e3a5f;padding:.5rem;text-align:left;}
    td{padding:.4rem .5rem;border-bottom:1px solid #1e293b;}
    .pos{color:#4ade80;} .neg{color:#f87171;} canvas{max-height:300px;}
    .card{background:#1e293b;border-radius:.5rem;padding:1rem;margin-bottom:1rem;}
    .badge{display:inline-block;padding:.2rem .6rem;border-radius:.3rem;font-size:.75rem;}
    .buy{background:#14532d;color:#4ade80;} .sell{background:#450a0a;color:#f87171;}
    #refresh{float:right;background:#0284c7;border:none;color:#fff;padding:.4rem .8rem;
             border-radius:.3rem;cursor:pointer;}
  </style>
</head>
<body>
  <h1>&#x1F4C8; AlpacaTrader Dashboard <button id="refresh" onclick="loadAll()">Refresh</button></h1>
  <div class="card"><h2>Equity Curve (last 100 trades)</h2><canvas id="equity"></canvas></div>
  <div class="card"><h2>Performance</h2><div id="perf"></div></div>
  <div class="card"><h2>Open Positions</h2><table id="pos-tbl">
    <thead><tr><th>Symbol</th><th>Qty</th><th>Entry $</th><th>Current $</th><th>P&amp;L $</th><th>P&amp;L %</th></tr></thead>
    <tbody id="pos-body"></tbody></table></div>
  <div class="card"><h2>Recent Trades</h2><table id="trade-tbl">
    <thead><tr><th>Time</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Price $</th><th>P&amp;L $</th><th>Strategy</th></tr></thead>
    <tbody id="trade-body"></tbody></table></div>
  <div class="card"><h2>Signal Feed</h2><table id="sig-tbl">
    <thead><tr><th>Time</th><th>Symbol</th><th>Action</th><th>Price $</th><th>Strategy</th><th>Reason</th></tr></thead>
    <tbody id="sig-body"></tbody></table></div>
<script>
const fmt2 = v => (+v).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
const pnlCls = v => +v >= 0 ? 'pos' : 'neg';
let chart = null;

async function loadAll(){
  try{
    const [pos,trades,perf,sigs] = await Promise.all([
      fetch('/api/positions').then(r=>r.json()),
      fetch('/api/trades').then(r=>r.json()),
      fetch('/api/performance').then(r=>r.json()),
      fetch('/api/signals').then(r=>r.json()),
    ]);
    renderPositions(pos);
    renderTrades(trades);
    renderPerformance(perf);
    renderSignals(sigs);
    renderEquity(trades);
  }catch(e){console.error(e);}
}

function renderPositions(data){
  const b=document.getElementById('pos-body'); b.innerHTML='';
  (data.positions||[]).forEach(p=>{
    const r=document.createElement('tr');
    r.innerHTML=`<td>${p.symbol}</td><td>${fmt2(p.qty)}</td><td>$${fmt2(p.avg_entry_price)}</td>
      <td>$${fmt2(p.current_price)}</td>
      <td class="${pnlCls(p.unrealized_pl)}">$${fmt2(p.unrealized_pl)}</td>
      <td class="${pnlCls(p.unrealized_plpc)}">${fmt2(+p.unrealized_plpc*100)}%</td>`;
    b.appendChild(r);
  });
}

function renderTrades(data){
  const b=document.getElementById('trade-body'); b.innerHTML='';
  (data.trades||[]).slice(0,50).forEach(t=>{
    const r=document.createElement('tr');
    r.innerHTML=`<td>${t.timestamp.slice(0,19)}</td><td>${t.symbol}</td>
      <td><span class="badge ${t.side}">${t.side.toUpperCase()}</span></td>
      <td>${fmt2(t.qty)}</td><td>$${fmt2(t.price)}</td>
      <td class="${pnlCls(t.pnl)}">$${fmt2(t.pnl)}</td>
      <td>${t.strategy}</td>`;
    b.appendChild(r);
  });
}

function renderPerformance(data){
  const div=document.getElementById('perf');
  const pv = data.portfolio_value||0, dp=data.daily_pnl||0;
  div.innerHTML=`
    <b>Portfolio:</b> $${fmt2(pv)} &nbsp;|&nbsp;
    <b>Daily P&amp;L:</b> <span class="${pnlCls(dp)}">$${fmt2(dp)} (${fmt2(data.daily_pnl_pct||0)}%)</span> &nbsp;|&nbsp;
    <b>Drawdown:</b> <span class="neg">${fmt2(data.drawdown_pct||0)}%</span>`;
}

function renderSignals(data){
  const b=document.getElementById('sig-body'); b.innerHTML='';
  [...(data.signals||[])].reverse().slice(0,30).forEach(s=>{
    const r=document.createElement('tr');
    r.innerHTML=`<td>${s.ts.slice(0,19)}</td><td>${s.symbol}</td>
      <td><span class="badge ${s.action.toLowerCase()}">${s.action}</span></td>
      <td>$${fmt2(s.price)}</td><td>${s.strategy}</td><td>${s.reason}</td>`;
    b.appendChild(r);
  });
}

function renderEquity(data){
  const trades=(data.trades||[]).slice().reverse();
  const labels=trades.map(t=>t.timestamp.slice(0,10));
  let running=0; const vals=trades.map(t=>{running+=+t.pnl;return running;});
  const ctx=document.getElementById('equity').getContext('2d');
  if(chart) chart.destroy();
  chart=new Chart(ctx,{type:'line',data:{labels,datasets:[{
    label:'Cumulative P&L ($)',data:vals,borderColor:'#38bdf8',
    backgroundColor:'rgba(56,189,248,.15)',fill:true,tension:.3,pointRadius:0
  }]},options:{scales:{x:{ticks:{maxTicksLimit:10}},y:{ticks:{callback:v=>'$'+v.toLocaleString()}}}}});
}

loadAll();
setInterval(loadAll, 30000);
</script>
</body>
</html>
"""


class Dashboard:
    """Flask-based dashboard that runs in a background daemon thread."""

    def __init__(
        self,
        config: DashboardConfig,
        db_path: str,
        get_positions_fn: Optional[Callable] = None,
        get_performance_fn: Optional[Callable] = None,
    ) -> None:
        self.config = config
        self.db_path = db_path
        self._get_positions = get_positions_fn or (lambda: [])
        self._get_performance = get_performance_fn or (lambda: {})
        self._app = None
        self._thread: Optional[threading.Thread] = None
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Flask app setup
    # ------------------------------------------------------------------

    def _build_app(self):
        from flask import Flask, jsonify

        app = Flask("alpacatrader_dashboard")
        app.config["JSON_SORT_KEYS"] = False
        # Suppress Flask startup banner
        import logging as _logging
        _logging.getLogger("werkzeug").setLevel(_logging.ERROR)

        @app.route("/")
        def index():
            from flask import Response
            return Response(_INLINE_HTML, mimetype="text/html")

        @app.route("/api/positions")
        def api_positions():
            try:
                raw = self._get_positions()
                positions = [
                    {
                        "symbol": p.symbol,
                        "qty": str(p.qty),
                        "avg_entry_price": str(p.avg_entry_price),
                        "current_price": str(p.current_price),
                        "unrealized_pl": str(p.unrealized_pl),
                        "unrealized_plpc": str(p.unrealized_plpc),
                    }
                    for p in raw
                ]
            except Exception as exc:
                logger.warning("Dashboard positions error: %s", exc)
                positions = []
            return jsonify({"positions": positions})

        @app.route("/api/trades")
        def api_trades():
            trades = self._get_trades()
            return jsonify({"trades": trades})

        @app.route("/api/performance")
        def api_performance():
            try:
                perf = self._get_performance()
            except Exception as exc:
                logger.warning("Dashboard performance error: %s", exc)
                perf = {}
            return jsonify(perf)

        @app.route("/api/signals")
        def api_signals():
            with _signal_lock:
                signals = list(_signal_feed)
            return jsonify({"signals": signals})

        @app.route("/health")
        def health():
            import time as _time
            uptime = int(_time.time() - self._start_time) if self._start_time else 0
            return jsonify({"status": "ok", "uptime_seconds": uptime})

        return app

    # ------------------------------------------------------------------
    # Trade query helper
    # ------------------------------------------------------------------

    def _get_trades(self, limit: int = 200) -> List[Dict[str, Any]]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,)
                ).fetchall()
            return [dict(row) for row in rows]
        except Exception as exc:
            logger.warning("Dashboard trade query error: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the Flask server in a background daemon thread."""
        if not self.config.enabled:
            return
        if self._thread and self._thread.is_alive():
            return

        import time as _time
        self._start_time = _time.time()
        self._app = self._build_app()
        host = self.config.host
        port = self.config.port

        def _run():
            self._app.run(host=host, port=port, debug=False, use_reloader=False)

        self._thread = threading.Thread(target=_run, name="dashboard", daemon=True)
        self._thread.start()
        logger.info("Dashboard running at http://%s:%d", host, port)
