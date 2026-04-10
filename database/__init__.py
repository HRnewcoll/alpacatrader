"""
Database module for trade logging and analytics
"""
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

Base = declarative_base()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./alpaca_trader.db")

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Trade(Base):
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    side = Column(String)  # BUY or SELL
    quantity = Column(Float)
    entry_price = Column(Float)
    exit_price = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)
    pnl_percent = Column(Float, nullable=True)
    strategy = Column(String)
    timeframe = Column(String)
    entry_time = Column(DateTime, default=datetime.utcnow)
    exit_time = Column(DateTime, nullable=True)
    status = Column(String, default="OPEN")  # OPEN, CLOSED, CANCELLED
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    trade_metadata = Column(JSON, nullable=True)
    regime = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)


class Signal(Base):
    __tablename__ = "signals"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    strategy = Column(String)
    signal_type = Column(String)  # BUY, SELL, HOLD
    strength = Column(Float)
    price = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    regime = Column(String)
    signal_metadata = Column(JSON, nullable=True)


class PerformanceMetric(Base):
    __tablename__ = "performance_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, default=datetime.utcnow)
    total_pnl = Column(Float)
    total_pnl_percent = Column(Float)
    sharpe_ratio = Column(Float, nullable=True)
    sortino_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)
    num_trades = Column(Integer, default=0)
    num_winning_trades = Column(Integer, default=0)
    avg_win = Column(Float, nullable=True)
    avg_loss = Column(Float, nullable=True)
    exposure = Column(Float, nullable=True)
    regime = Column(String, nullable=True)


class MarketRegimeLog(Base):
    __tablename__ = "market_regime_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    regime = Column(String)
    trend_strength = Column(Float, nullable=True)
    volatility_level = Column(String, nullable=True)
    adx = Column(Float, nullable=True)
    rsi = Column(Float, nullable=True)
    hurst = Column(Float, nullable=True)
    recommended_strategies = Column(JSON, nullable=True)
    risk_adjustment = Column(Float, nullable=True)


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        return db
    finally:
        pass


def log_trade(db, symbol, side, quantity, entry_price, strategy, timeframe, 
              stop_loss=None, take_profit=None, metadata=None, regime=None, confidence=None):
    """Log a new trade"""
    trade = Trade(
        symbol=symbol,
        side=side,
        quantity=quantity,
        entry_price=entry_price,
        strategy=strategy,
        timeframe=timeframe,
        stop_loss=stop_loss,
        take_profit=take_profit,
        metadata=metadata,
        regime=regime,
        confidence=confidence
    )
    db.add(trade)
    db.commit()
    db.refresh(trade)
    return trade


def close_trade(db, trade_id, exit_price):
    """Close a trade"""
    trade = db.query(Trade).filter(Trade.id == trade_id).first()
    if trade:
        trade.exit_price = exit_price
        trade.exit_time = datetime.utcnow()
        trade.status = "CLOSED"
        
        # Calculate P&L
        if trade.side == "BUY":
            trade.pnl = (exit_price - trade.entry_price) * trade.quantity
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.quantity
        
        trade.pnl_percent = (trade.pnl / (trade.entry_price * trade.quantity)) * 100
        
        db.commit()
        db.refresh(trade)
    return trade


def log_signal(db, symbol, strategy, signal_type, strength, price, regime=None, metadata=None):
    """Log a trading signal"""
    signal = Signal(
        symbol=symbol,
        strategy=strategy,
        signal_type=signal_type,
        strength=strength,
        price=price,
        regime=regime,
        metadata=metadata
    )
    db.add(signal)
    db.commit()
    db.refresh(signal)
    return signal


def log_regime(db, regime, trend_strength=None, volatility_level=None, adx=None,
               rsi=None, hurst=None, recommended_strategies=None, risk_adjustment=None):
    """Log market regime detection"""
    regime_log = MarketRegimeLog(
        regime=regime,
        trend_strength=trend_strength,
        volatility_level=volatility_level,
        adx=adx,
        rsi=rsi,
        hurst=hurst,
        recommended_strategies=recommended_strategies,
        risk_adjustment=risk_adjustment
    )
    db.add(regime_log)
    db.commit()
    db.refresh(regime_log)
    return regime_log


def log_performance(db, total_pnl, total_pnl_percent, sharpe_ratio=None, sortino_ratio=None,
                   max_drawdown=None, win_rate=None, profit_factor=None, num_trades=0,
                   num_winning_trades=0, avg_win=None, avg_loss=None, exposure=None, regime=None):
    """Log performance metrics"""
    metric = PerformanceMetric(
        total_pnl=total_pnl,
        total_pnl_percent=total_pnl_percent,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        profit_factor=profit_factor,
        num_trades=num_trades,
        num_winning_trades=num_winning_trades,
        avg_win=avg_win,
        avg_loss=avg_loss,
        exposure=exposure,
        regime=regime
    )
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return metric
