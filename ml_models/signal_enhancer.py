"""
ML-based Signal Enhancer using Ensemble Methods
Enhances trading signals with machine learning models
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
from datetime import datetime


class SignalEnhancer:
    """
    ML model to enhance trading signals from multiple strategies
    Uses ensemble methods to predict signal success probability
    """
    
    def __init__(self, model_path=None):
        self.model_path = model_path or "ml_models/signal_model.pkl"
        self.scaler = StandardScaler()
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=100, 
                max_depth=5, 
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced'
            )
        }
        self.ensemble_weights = {
            'random_forest': 0.4,
            'gradient_boost': 0.4,
            'logistic_regression': 0.2
        }
        self.feature_columns = None
        self.is_trained = False
    
    def create_features(self, df):
        """
        Create ML features from market data
        """
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['returns_1'] = df['close'].pct_change(1)
        features['returns_5'] = df['close'].pct_change(5)
        features['returns_10'] = df['close'].pct_change(10)
        features['returns_20'] = df['close'].pct_change(20)
        
        # Volatility features
        features['volatility_5'] = df['close'].rolling(5).std()
        features['volatility_20'] = df['close'].rolling(20).std()
        features['volatility_ratio'] = features['volatility_5'] / (features['volatility_20'] + 1e-8)
        
        # Volume features
        if 'volume' in df.columns:
            features['volume_ma_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-8)
            features['volume_change'] = df['volume'].pct_change()
        
        # Momentum features
        features['rsi'] = self._calculate_rsi(df['close'], 14)
        features['momentum_10'] = df['close'] - df['close'].shift(10)
        features['price_vs_ma20'] = df['close'] / (df['close'].rolling(20).mean() + 1e-8) - 1
        features['price_vs_ma50'] = df['close'] / (df['close'].rolling(50).mean() + 1e-8) - 1
        
        # Trend features
        features['adx'] = df.get('adx', pd.Series(0, index=df.index))
        features['macd'] = df.get('macd', pd.Series(0, index=df.index))
        
        # Pattern features
        features['higher_highs'] = (df['high'] > df['high'].shift(1)).astype(int)
        features['lower_lows'] = (df['low'] < df['low'].shift(1)).astype(int)
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            features[f'returns_lag_{lag}'] = features['returns_1'].shift(lag)
        
        # Target: Next period return direction
        features['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Drop NaN values
        features = features.dropna()
        
        self.feature_columns = [col for col in features.columns if col != 'target']
        
        return features
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    def train(self, df, symbol="GENERAL"):
        """
        Train ensemble model on historical data
        """
        print(f"Training ML signal enhancer for {symbol}...")
        
        features = self.create_features(df)
        
        if len(features) < 100:
            print("Insufficient data for training")
            return False
        
        X = features[self.feature_columns]
        y = features['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train individual models
        predictions = {}
        for name, model in self.models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            predictions[name] = y_pred
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            print(f"  {name}: Accuracy={acc:.3f}, Precision={prec:.3f}")
        
        self.is_trained = True
        
        # Save model
        if self.model_path:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump({
                'models': self.models,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'weights': self.ensemble_weights
            }, self.model_path)
            print(f"Model saved to {self.model_path}")
        
        return True
    
    def load_model(self):
        """Load trained model from disk"""
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.models = data['models']
            self.scaler = data['scaler']
            self.feature_columns = data['feature_columns']
            self.ensemble_weights = data['weights']
            self.is_trained = True
            print(f"Model loaded from {self.model_path}")
            return True
        return False
    
    def predict(self, df):
        """
        Predict signal success probability
        Returns: probability of success, confidence score
        """
        if not self.is_trained:
            if not self.load_model():
                return 0.5, 0.0  # Default neutral prediction
        
        features = self.create_features(df)
        if len(features) == 0:
            return 0.5, 0.0
        
        X = features[self.feature_columns].iloc[-1:].values
        X_scaled = self.scaler.transform(X)
        
        # Ensemble prediction
        probabilities = []
        for name, model in self.models.items():
            prob = model.predict_proba(X_scaled)[0][1]  # Probability of positive class
            probabilities.append(prob * self.ensemble_weights[name])
        
        final_prob = sum(probabilities)
        confidence = abs(final_prob - 0.5) * 2  # 0 to 1 scale
        
        return final_prob, confidence
    
    def enhance_signal(self, original_signal, original_confidence, df):
        """
        Enhance a trading signal with ML prediction
        Returns: enhanced_signal, enhanced_confidence, ml_probability
        """
        ml_prob, ml_conf = self.predict(df)
        
        # Combine original signal with ML prediction
        # If ML agrees with signal, boost confidence; if disagrees, reduce
        signal_direction = 1 if original_signal > 0 else -1 if original_signal < 0 else 0
        
        if signal_direction > 0 and ml_prob > 0.5:
            agreement = 1
        elif signal_direction < 0 and ml_prob < 0.5:
            agreement = 1
        elif signal_direction == 0:
            agreement = 0
        else:
            agreement = -1
        
        # Weighted combination
        enhanced_confidence = (original_confidence * 0.6 + ml_conf * 0.4)
        
        if agreement == 1:
            enhanced_confidence = min(enhanced_confidence * 1.2, 1.0)
        elif agreement == -1:
            enhanced_confidence = max(enhanced_confidence * 0.7, 0.1)
        
        enhanced_signal = original_signal * (0.7 + 0.3 * (ml_prob - 0.5) * 2)
        
        return enhanced_signal, enhanced_confidence, ml_prob


class ReinforcementPositionSizer:
    """
    Simple reinforcement learning-inspired position sizer
    Adjusts position sizes based on recent performance
    """
    
    def __init__(self, base_position_size=0.02, max_position_size=0.1):
        self.base_position_size = base_position_size
        self.max_position_size = max_position_size
        self.current_size = base_position_size
        self.performance_history = []
        self.win_streak = 0
        self.loss_streak = 0
    
    def update(self, pnl_percent):
        """Update based on trade result"""
        self.performance_history.append(pnl_percent)
        
        if len(self.performance_history) > 20:
            self.performance_history.pop(0)
        
        if pnl_percent > 0:
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.loss_streak += 1
            self.win_streak = 0
        
        # Adjust position size based on recent performance
        if len(self.performance_history) >= 5:
            recent_win_rate = sum(1 for p in self.performance_history[-5:] if p > 0) / 5
            
            if recent_win_rate > 0.6 and self.win_streak >= 2:
                # Increase size on winning streak
                self.current_size = min(self.current_size * 1.1, self.max_position_size)
            elif recent_win_rate < 0.4 or self.loss_streak >= 3:
                # Decrease size on losing streak
                self.current_size = max(self.current_size * 0.9, self.base_position_size * 0.5)
    
    def get_position_size(self, signal_confidence, volatility_adjustment=1.0):
        """
        Calculate position size based on signal confidence and volatility
        """
        base = self.current_size
        confidence_factor = signal_confidence
        vol_factor = 1.0 / volatility_adjustment if volatility_adjustment > 0 else 0.5
        
        position_size = base * confidence_factor * vol_factor
        position_size = max(position_size, self.base_position_size * 0.25)
        position_size = min(position_size, self.max_position_size)
        
        return position_size
