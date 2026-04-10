"""
Real-time Trading Dashboard with FastAPI and Plotly
Provides live monitoring, analytics, and control interface
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

app = FastAPI(title="AlpacaTrader Dashboard", version="2.0")

# In-memory storage (in production, use database)
trading_state = {
    "is_running": False,
    "current_regime": "UNKNOWN",
    "active_positions": [],
    "today_pnl": 0.0,
    "today_trades": 0,
    "last_update": datetime.now()
}

class TradingStatus(BaseModel):
    is_running: bool
    current_regime: str
    active_positions: int
    today_pnl: float
    today_trades: int
    last_update: datetime

class SignalRequest(BaseModel):
    symbol: str
    strategy: str
    signal_type: str
    strength: float
    confidence: float

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the dashboard HTML"""
    return get_dashboard_html()

@app.get("/api/status")
async def get_status():
    """Get current trading status"""
    return trading_state

@app.post("/api/start")
async def start_trading():
    """Start trading engine"""
    trading_state["is_running"] = True
    trading_state["last_update"] = datetime.now()
    return {"status": "started", "timestamp": datetime.now()}

@app.post("/api/stop")
async def stop_trading():
    """Stop trading engine"""
    trading_state["is_running"] = False
    trading_state["last_update"] = datetime.now()
    return {"status": "stopped", "timestamp": datetime.now()}

@app.get("/api/positions")
async def get_positions():
    """Get active positions"""
    return {"positions": trading_state["active_positions"]}

@app.get("/api/performance")
async def get_performance():
    """Get performance metrics"""
    # Generate sample performance data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    cumulative_pnl = np.random.randn(30).cumsum() * 100
    
    return {
        "daily_pnl": [
            {"date": d.strftime("%Y-%m-%d"), "pnl": float(p)} 
            for d, p in zip(dates, cumulative_pnl)
        ],
        "total_pnl": float(cumulative_pnl[-1]),
        "sharpe_ratio": float(np.mean(cumulative_pnl) / (np.std(cumulative_pnl) + 1e-8)),
        "max_drawdown": float(min(0, cumulative_pnl.min())),
        "win_rate": 0.55 + np.random.randn() * 0.1
    }

@app.get("/api/signals")
async def get_signals():
    """Get recent signals"""
    return {
        "signals": [
            {
                "symbol": "AAPL",
                "strategy": "momentum",
                "type": "BUY",
                "strength": 0.75,
                "confidence": 0.82,
                "timestamp": datetime.now() - timedelta(minutes=5)
            },
            {
                "symbol": "TSLA",
                "strategy": "mean_reversion",
                "type": "SELL",
                "strength": 0.65,
                "confidence": 0.71,
                "timestamp": datetime.now() - timedelta(minutes=15)
            }
        ]
    }

@app.get("/api/regime")
async def get_regime():
    """Get current market regime"""
    return {
        "regime": trading_state["current_regime"],
        "trend_strength": 0.65,
        "volatility": "normal",
        "adx": 28.5,
        "rsi": 52.3,
        "recommended_strategies": ["momentum", "volatility_breakout"],
        "risk_adjustment": 1.0
    }

def get_dashboard_html():
    """Generate interactive dashboard HTML"""
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlpacaTrader Pro Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .card { @apply bg-white rounded-lg shadow-md p-6; }
        .metric { @apply text-3xl font-bold; }
        .label { @apply text-gray-500 text-sm; }
        .btn { @apply px-4 py-2 rounded-lg font-semibold transition; }
        .btn-primary { @apply bg-blue-500 hover:bg-blue-600 text-white; }
        .btn-danger { @apply bg-red-500 hover:bg-red-600 text-white; }
        .btn-success { @apply bg-green-500 hover:bg-green-600 text-white; }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <nav class="bg-gray-800 text-white p-4">
        <div class="container mx-auto flex justify-between items-center">
            <h1 class="text-2xl font-bold">🚀 AlpacaTrader Pro</h1>
            <div class="space-x-4">
                <span id="status-indicator" class="px-3 py-1 rounded-full bg-gray-500">STOPPED</span>
                <span id="clock" class="text-lg"></span>
            </div>
        </div>
    </nav>

    <div class="container mx-auto p-6">
        <!-- Control Panel -->
        <div class="card mb-6">
            <h2 class="text-xl font-semibold mb-4">Control Panel</h2>
            <div class="flex space-x-4">
                <button onclick="startTrading()" class="btn btn-success">▶ Start Trading</button>
                <button onclick="stopTrading()" class="btn btn-danger">⏹ Stop Trading</button>
                <button onclick="refreshData()" class="btn btn-primary">🔄 Refresh</button>
            </div>
        </div>

        <!-- Key Metrics -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
            <div class="card">
                <div class="label">Today's P&L</div>
                <div id="pnl" class="metric text-green-500">$0.00</div>
            </div>
            <div class="card">
                <div class="label">Active Positions</div>
                <div id="positions" class="metric">0</div>
            </div>
            <div class="card">
                <div class="label">Trades Today</div>
                <div id="trades" class="metric">0</div>
            </div>
            <div class="card">
                <div class="label">Market Regime</div>
                <div id="regime" class="metric text-sm">UNKNOWN</div>
            </div>
        </div>

        <!-- Charts -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            <div class="card">
                <h3 class="font-semibold mb-4">Performance (30 Days)</h3>
                <div id="performance-chart" class="h-64"></div>
            </div>
            <div class="card">
                <h3 class="font-semibold mb-4">Signal Strength</h3>
                <div id="signal-chart" class="h-64"></div>
            </div>
        </div>

        <!-- Recent Signals -->
        <div class="card">
            <h3 class="font-semibold mb-4">Recent Signals</h3>
            <div id="signals-table" class="overflow-x-auto">
                <table class="w-full">
                    <thead>
                        <tr class="border-b">
                            <th class="text-left py-2">Symbol</th>
                            <th class="text-left py-2">Strategy</th>
                            <th class="text-left py-2">Type</th>
                            <th class="text-left py-2">Strength</th>
                            <th class="text-left py-2">Confidence</th>
                            <th class="text-left py-2">Time</th>
                        </tr>
                    </thead>
                    <tbody id="signals-body"></tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        // Update clock
        function updateClock() {
            document.getElementById('clock').textContent = new Date().toLocaleTimeString();
        }
        setInterval(updateClock, 1000);
        updateClock();

        // API calls
        async function fetchStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                
                document.getElementById('pnl').textContent = '$' + data.today_pnl.toFixed(2);
                document.getElementById('pnl').className = 'metric ' + (data.today_pnl >= 0 ? 'text-green-500' : 'text-red-500');
                document.getElementById('positions').textContent = data.active_positions.length;
                document.getElementById('trades').textContent = data.today_trades;
                document.getElementById('regime').textContent = data.current_regime;
                
                const indicator = document.getElementById('status-indicator');
                if (data.is_running) {
                    indicator.textContent = 'RUNNING';
                    indicator.className = 'px-3 py-1 rounded-full bg-green-500';
                } else {
                    indicator.textContent = 'STOPPED';
                    indicator.className = 'px-3 py-1 rounded-full bg-gray-500';
                }
            } catch (e) {
                console.error('Error fetching status:', e);
            }
        }

        async function fetchPerformance() {
            try {
                const res = await fetch('/api/performance');
                const data = await res.json();
                
                const trace = {
                    x: data.daily_pnl.map(d => d.date),
                    y: data.daily_pnl.map(d => d.pnl),
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: {color: data.total_pnl >= 0 ? '#10B981' : '#EF4444'}
                };
                
                Plotly.newPlot('performance-chart', [trace], {
                    margin: {t: 20, b: 40, l: 40, r: 20},
                    showlegend: false
                });
            } catch (e) {
                console.error('Error fetching performance:', e);
            }
        }

        async function fetchSignals() {
            try {
                const res = await fetch('/api/signals');
                const data = await res.json();
                
                const tbody = document.getElementById('signals-body');
                tbody.innerHTML = data.signals.map(s => `
                    <tr class="border-b hover:bg-gray-50">
                        <td class="py-2 font-semibold">${s.symbol}</td>
                        <td class="py-2">${s.strategy}</td>
                        <td class="py-2">
                            <span class="px-2 py-1 rounded ${s.type === 'BUY' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}">
                                ${s.type}
                            </span>
                        </td>
                        <td class="py-2">${(s.strength * 100).toFixed(0)}%</td>
                        <td class="py-2">${(s.confidence * 100).toFixed(0)}%</td>
                        <td class="py-2 text-gray-500">${new Date(s.timestamp).toLocaleTimeString()}</td>
                    </tr>
                `).join('');
                
                // Signal strength chart
                const trace = {
                    x: data.signals.map(s => s.symbol),
                    y: data.signals.map(s => s.strength),
                    type: 'bar',
                    marker: {
                        color: data.signals.map(s => s.type === 'BUY' ? '#10B981' : '#EF4444')
                    }
                };
                
                Plotly.newPlot('signal-chart', [trace], {
                    margin: {t: 20, b: 40, l: 40, r: 20},
                    showlegend: false,
                    yaxis: {range: [0, 1]}
                });
            } catch (e) {
                console.error('Error fetching signals:', e);
            }
        }

        async function startTrading() {
            await fetch('/api/start', {method: 'POST'});
            refreshData();
        }

        async function stopTrading() {
            await fetch('/api/stop', {method: 'POST'});
            refreshData();
        }

        function refreshData() {
            fetchStatus();
            fetchPerformance();
            fetchSignals();
        }

        // Initial load and auto-refresh
        refreshData();
        setInterval(refreshData, 5000);
    </script>
</body>
</html>
'''

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
