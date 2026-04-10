"""
Neural Network Strategies using Deep Learning
LSTM, Transformer, and Reinforcement Learning models for price prediction
Inspired by DeepMind, Two Sigma, and Renaissance Technologies
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import deque
import asyncio


# ==================== DATASETS ====================

class PriceDataset(Dataset):
    """PyTorch dataset for price sequences"""
    
    def __init__(
        self,
        prices: np.ndarray,
        features: Optional[np.ndarray] = None,
        sequence_length: int = 60,
        prediction_horizon: int = 5,
    ):
        self.prices = prices
        self.features = features if features is not None else prices.reshape(-1, 1)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Normalize data
        self.price_scaler = StandardScaler()
        self.feature_scaler = StandardScaler()
        
        prices_normalized = self.price_scaler.fit_transform(prices.reshape(-1, 1)).flatten()
        features_normalized = self.feature_scaler.fit_transform(self.features)
        
        self.prices_norm = prices_normalized
        self.features_norm = features_normalized
        
    def __len__(self):
        return len(self.prices) - self.sequence_length - self.prediction_horizon
    
    def __getitem__(self, idx):
        x = self.features_norm[idx:idx + self.sequence_length]
        y = self.prices_norm[idx + self.sequence_length:idx + self.sequence_length + self.prediction_horizon]
        
        # Convert to tensors
        x_tensor = torch.FloatTensor(x)
        y_tensor = torch.FloatTensor(y)
        
        return x_tensor, y_tensor


# ==================== MODELS ====================

class LSTMPricePredictor(nn.Module):
    """
    LSTM-based price prediction model
    Multi-layer LSTM with attention mechanism
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 3,
        output_size: int = 5,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        super(LSTMPricePredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1),
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )
        
    def forward(self, x):
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Attention weights
        attn_weights = self.attention(lstm_out)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        # Final prediction
        output = self.fc(context)
        return output


class TransformerPredictor(nn.Module):
    """
    Transformer-based price prediction model
    Self-attention architecture for sequence modeling
    """
    
    def __init__(
        self,
        input_size: int = 1,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        output_size: int = 5,
        dropout: float = 0.1,
        max_seq_length: int = 5000,
    ):
        super(TransformerPredictor, self).__init__()
        
        self.d_model = d_model
        
        # Input embedding
        self.input_embedding = nn.Linear(input_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_length, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size),
        )
        
    def forward(self, x):
        # Embedding + positional encoding
        x = self.input_embedding(x)
        x = x + self.positional_encoding[:, :x.size(1), :]
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Global pooling
        x = x.transpose(1, 2)  # (batch, features, seq)
        x = self.global_pool(x).squeeze(-1)
        
        # Output
        output = self.output_layers(x)
        return output


class CNNLSTMPredictor(nn.Module):
    """
    Hybrid CNN-LSTM model
    CNN for feature extraction, LSTM for temporal modeling
    """
    
    def __init__(
        self,
        input_size: int = 1,
        cnn_channels: List[int] = [32, 64, 128],
        kernel_sizes: List[int] = [3, 3, 3],
        lstm_hidden_size: int = 128,
        output_size: int = 5,
        dropout: float = 0.2,
    ):
        super(CNNLSTMPredictor, self).__init__()
        
        # CNN layers
        cnn_layers = []
        in_channels = input_size
        
        for i, (channels, kernel) in enumerate(zip(cnn_channels, kernel_sizes)):
            cnn_layers.extend([
                nn.Conv1d(in_channels, channels, kernel_size=kernel, padding=kernel//2),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout),
            ])
            in_channels = channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size),
        )
        
    def forward(self, x):
        # x shape: (batch, seq, features)
        x = x.transpose(1, 2)  # (batch, features, seq)
        
        # CNN feature extraction
        x = self.cnn(x)
        
        # Back to (batch, seq, features) for LSTM
        x = x.transpose(1, 2)
        
        # LSTM temporal modeling
        lstm_out, _ = self.lstm(x)
        
        # Use last output
        x = lstm_out[:, -1, :]
        
        # Output
        output = self.fc(x)
        return output


# ==================== TRADING AGENT (RL) ====================

class TradingEnvironment:
    """
    Reinforcement Learning environment for trading
    Supports PPO, DQN, A2C algorithms
    """
    
    def __init__(
        self,
        prices: np.ndarray,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,
        window_size: int = 60,
    ):
        self.prices = prices
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.window_size = window_size
        
        self.reset()
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = self.window_size
        self.capital = self.initial_capital
        self.position = 0  # Number of shares held
        self.trades = []
        self.pnl_history = [0.0]
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        # Price features
        prices = self.prices[self.current_step - self.window_size:self.current_step]
        returns = np.diff(prices) / prices[:-1]
        
        # Technical indicators
        ma_short = np.mean(prices[-10:])
        ma_long = np.mean(prices[-30:])
        rsi = self._calculate_rsi(prices)
        
        # Portfolio state
        position_pct = self.position * prices[-1] / self.capital if self.capital > 0 else 0
        
        state = np.concatenate([
            returns[-20:],  # Last 20 returns
            [ma_short / prices[-1] - 1],
            [ma_long / prices[-1] - 1],
            [rsi],
            [position_pct],
        ])
        
        return state
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return (state, reward, done, info)
        Actions: 0=Hold, 1=Buy, 2=Sell
        """
        current_price = self.prices[self.current_step]
        
        # Execute action
        reward = 0.0
        info = {}
        
        if action == 1:  # Buy
            shares_to_buy = int(self.capital * 0.1 / current_price)  # Use 10% of capital
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                self.capital -= cost
                self.position += shares_to_buy
                
        elif action == 2:  # Sell
            shares_to_sell = min(self.position, int(self.position * 0.5))  # Sell 50%
            if shares_to_sell > 0:
                proceeds = shares_to_sell * current_price * (1 - self.transaction_cost)
                self.capital += proceeds
                self.position -= shares_to_sell
        
        # Calculate reward (change in portfolio value)
        portfolio_value = self.capital + self.position * current_price
        current_pnl = (portfolio_value - self.initial_capital) / self.initial_capital
        previous_pnl = self.pnl_history[-1]
        reward = current_pnl - previous_pnl
        
        self.pnl_history.append(current_pnl)
        self.trades.append({
            'step': self.current_step,
            'price': current_price,
            'action': action,
            'pnl': current_pnl,
        })
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1
        
        # Get new state
        state = self._get_state()
        
        info['portfolio_value'] = portfolio_value
        info['pnl'] = current_pnl
        info['position'] = self.position
        
        return state, reward, done, info


class TradingAgent(nn.Module):
    """
    Deep RL agent for trading
    Actor-Critic architecture for PPO
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        hidden_dim: int = 128,
    ):
        super(TradingAgent, self).__init__()
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )
        
        # Critic network (value)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(self, state):
        """Forward pass returning action probabilities and value"""
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value
    
    def get_action(self, state, deterministic=False):
        """Sample action from policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs, _ = self.forward(state_tensor)
            
            if deterministic:
                action = torch.argmax(action_probs, dim=-1)
            else:
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
            
            return action.item()


# ==================== TRAINING & PREDICTION ====================

class NeuralStrategy:
    """
    Main neural network trading strategy class
    Combines multiple models with ensemble predictions
    """
    
    def __init__(
        self,
        symbols: List[str],
        model_type: str = 'lstm',  # 'lstm', 'transformer', 'cnn_lstm'
        sequence_length: int = 60,
        prediction_horizon: int = 5,
        use_ensemble: bool = True,
    ):
        self.symbols = symbols
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.use_ensemble = use_ensemble
        
        self.models: Dict[str, nn.Module] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.is_trained = False
        
    def train(
        self,
        prices_dict: Dict[str, np.ndarray],
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
    ):
        """Train models on historical data"""
        
        for symbol, prices in prices_dict.items():
            print(f"🧠 Training {self.model_type.upper()} model for {symbol}...")
            
            # Create dataset
            dataset = PriceDataset(
                prices=prices,
                sequence_length=self.sequence_length,
                prediction_horizon=self.prediction_horizon,
            )
            
            # Split train/val
            val_size = int(len(dataset) * validation_split)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Initialize model
            if self.model_type == 'lstm':
                model = LSTMPricePredictor(
                    input_size=dataset.features_norm.shape[1],
                    output_size=self.prediction_horizon,
                )
            elif self.model_type == 'transformer':
                model = TransformerPredictor(
                    input_size=dataset.features_norm.shape[1],
                    output_size=self.prediction_horizon,
                )
            elif self.model_type == 'cnn_lstm':
                model = CNNLSTMPredictor(
                    input_size=dataset.features_norm.shape[1],
                    output_size=self.prediction_horizon,
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
            
            # Training loop
            best_val_loss = float('inf')
            
            for epoch in range(epochs):
                # Train
                model.train()
                train_loss = 0.0
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    predictions = model(batch_x)
                    loss = criterion(predictions, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                
                # Validate
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        predictions = model(batch_x)
                        loss = criterion(predictions, batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                
                if (epoch + 1) % 10 == 0:
                    print(f"   Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.6f}, "
                          f"Val Loss: {val_loss:.6f}")
            
            self.models[symbol] = model
            self.scalers[symbol] = dataset.price_scaler
            print(f"✅ Model trained for {symbol} (Best Val Loss: {best_val_loss:.6f})")
        
        self.is_trained = True
    
    def predict(self, symbol: str, recent_prices: np.ndarray) -> np.ndarray:
        """Generate price prediction"""
        if not self.is_trained or symbol not in self.models:
            raise ValueError("Model not trained")
        
        model = self.models[symbol]
        scaler = self.scalers[symbol]
        
        # Prepare input
        if len(recent_prices) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} prices")
        
        recent_prices = recent_prices[-self.sequence_length:]
        normalized = scaler.transform(recent_prices.reshape(-1, 1)).flatten()
        
        # Predict
        model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(-1)
            prediction = model(x_tensor)
            pred_normalized = prediction.numpy().flatten()
        
        # Denormalize
        pred_prices = scaler.inverse_transform(pred_normalized.reshape(-1, 1)).flatten()
        
        return pred_prices
    
    def generate_signal(self, symbol: str, recent_prices: np.ndarray) -> Dict:
        """Generate trading signal from prediction"""
        predictions = self.predict(symbol, recent_prices)
        
        current_price = recent_prices[-1]
        predicted_return = (predictions[0] - current_price) / current_price
        
        # Signal strength based on predicted return
        if predicted_return > 0.02:
            signal = "STRONG_BUY"
            confidence = min(predicted_return / 0.05, 1.0)
        elif predicted_return > 0.005:
            signal = "BUY"
            confidence = predicted_return / 0.02
        elif predicted_return < -0.02:
            signal = "STRONG_SELL"
            confidence = min(abs(predicted_return) / 0.05, 1.0)
        elif predicted_return < -0.005:
            signal = "SELL"
            confidence = abs(predicted_return) / 0.02
        else:
            signal = "HOLD"
            confidence = 0.0
        
        return {
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence,
            'predicted_return': predicted_return,
            'current_price': current_price,
            'target_price': predictions[0],
            'prediction_horizon': self.prediction_horizon,
        }


def demo_neural_strategies():
    """Demonstrate neural network strategies"""
    print("=" * 70)
    print("🧠 NEURAL NETWORK STRATEGIES DEMO")
    print("=" * 70)
    
    # Generate synthetic price data
    np.random.seed(42)
    n_days = 500
    prices = 100 + np.cumsum(np.random.randn(n_days) * 2)
    prices = np.maximum(prices, 10)  # Ensure positive
    
    print(f"\n📊 Generated {n_days} days of synthetic price data")
    print(f"   Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    
    # Test LSTM model
    print("\n" + "-" * 70)
    print("Testing LSTM Price Predictor")
    print("-" * 70)
    
    strategy = NeuralStrategy(
        symbols=['SYNTH'],
        model_type='lstm',
        sequence_length=30,
        prediction_horizon=5,
    )
    
    # Train
    strategy.train(
        prices_dict={'SYNTH': prices},
        epochs=30,
        batch_size=32,
    )
    
    # Predict
    recent_prices = prices[-30:]
    prediction = strategy.predict('SYNTH', recent_prices)
    
    print(f"\n📈 Prediction Results:")
    print(f"   Current Price: ${recent_prices[-1]:.2f}")
    print(f"   Predicted Price (5 days): ${prediction[0]:.2f}")
    print(f"   Expected Return: {(prediction[0] - recent_prices[-1]) / recent_prices[-1] * 100:.2f}%")
    
    # Generate signal
    signal = strategy.generate_signal('SYNTH', recent_prices)
    print(f"\n🎯 Trading Signal:")
    print(f"   Signal: {signal['signal']}")
    print(f"   Confidence: {signal['confidence']*100:.1f}%")
    print(f"   Target: ${signal['target_price']:.2f}")
    
    # Test RL Agent
    print("\n" + "-" * 70)
    print("Testing RL Trading Agent")
    print("-" * 70)
    
    env = TradingEnvironment(prices=prices, initial_capital=100000)
    agent = TradingAgent(state_dim=25)  # Match state dimension
    
    # Simple training loop (few iterations for demo)
    print("\n🏋️ Training RL agent (100 episodes)...")
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    
    for episode in range(100):
        state = env.reset()
        total_reward = 0
        
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state
        
        if (episode + 1) % 20 == 0:
            print(f"   Episode {episode+1}: Total Reward = {total_reward:.4f}, "
                  f"Final PnL = {info['pnl']*100:.2f}%")
    
    # Test trained agent
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = agent.get_action(state, deterministic=True)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state
    
    print(f"\n📊 RL Agent Performance:")
    print(f"   Total Reward: {total_reward:.4f}")
    print(f"   Final PnL: {info['pnl']*100:.2f}%")
    print(f"   Final Portfolio: ${info['portfolio_value']:.2f}")
    
    print("\n" + "=" * 70)
    print("✅ Neural Strategies Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo_neural_strategies()
