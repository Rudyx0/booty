import math
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging

class MathUtils:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def calculate_profit_percentage(self, buy_price: float, sell_price: float) -> float:
        """Calculate profit percentage"""
        if buy_price <= 0:
            return 0.0
        return ((sell_price - buy_price) / buy_price) * 100
        
    def calculate_slippage_impact(self, trade_size: float, liquidity: float, 
                                base_slippage: float = 0.001) -> float:
        """Calculate slippage impact based on trade size and liquidity"""
        if liquidity <= 0:
            return 1.0  # Maximum slippage if no liquidity
            
        # Slippage increases with trade size relative to liquidity
        size_ratio = trade_size / liquidity
        
        # Exponential slippage model
        if size_ratio <= 0.001:  # < 0.1% of pool
            return base_slippage
        elif size_ratio <= 0.01:  # < 1% of pool
            return base_slippage * (1 + size_ratio * 5)
        else:  # >= 1% of pool
            return base_slippage * (1 + math.exp(size_ratio * 10))
            
    def calculate_price_impact(self, trade_amount: float, liquidity: float) -> float:
        """Calculate price impact using constant product formula"""
        if liquidity <= 0:
            return 1.0
            
        # Simplified constant product model: x * y = k
        # Price impact = trade_amount / (liquidity + trade_amount)
        return trade_amount / (liquidity + trade_amount)
        
    def optimize_trade_size(self, max_size: float, liquidity: float, 
                          target_slippage: float = 0.01) -> float:
        """Optimize trade size to stay within slippage limits"""
        if liquidity <= 0:
            return 0.0
            
        # Binary search for optimal size
        low, high = 0.0, max_size
        optimal_size = 0.0
        
        for _ in range(20):  # Max 20 iterations
            mid = (low + high) / 2
            slippage = self.calculate_slippage_impact(mid, liquidity)
            
            if slippage <= target_slippage:
                optimal_size = mid
                low = mid
            else:
                high = mid
                
            if high - low < 0.001:  # Convergence threshold
                break
                
        return optimal_size
        
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, 
                                avg_loss: float) -> float:
        """Calculate Kelly Criterion for position sizing"""
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
            
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Apply safety factor (never risk more than 25% of Kelly)
        return max(0.0, min(kelly_fraction * 0.25, 0.1))
        
    def calculate_sharpe_ratio(self, returns: List[float], 
                             risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio for performance evaluation"""
        if not returns or len(returns) < 2:
            return 0.0
            
        returns_array = np.array(returns)
        
        # Annualized return
        mean_return = np.mean(returns_array) * 365  # Daily to annual
        
        # Annualized volatility
        volatility = np.std(returns_array) * math.sqrt(365)
        
        if volatility == 0:
            return 0.0
            
        return (mean_return - risk_free_rate) / volatility
        
    def calculate_maximum_drawdown(self, balance_history: List[float]) -> float:
        """Calculate maximum drawdown from balance history"""
        if not balance_history:
            return 0.0
            
        balances = np.array(balance_history)
        peak = np.maximum.accumulate(balances)
        drawdown = (peak - balances) / peak
        
        return np.max(drawdown)
        
    def calculate_compound_annual_growth_rate(self, start_value: float, 
                                           end_value: float, 
                                           periods: int) -> float:
        """Calculate CAGR"""
        if start_value <= 0 or periods <= 0:
            return 0.0
            
        return (end_value / start_value) ** (1 / periods) - 1
        
    def calculate_value_at_risk(self, returns: List[float], 
                              confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk (VaR)"""
        if not returns:
            return 0.0
            
        returns_array = np.array(returns)
        return np.percentile(returns_array, (1 - confidence_level) * 100)
        
    def moving_average(self, data: List[float], window: int) -> List[float]:
        """Calculate moving average"""
        if len(data) < window:
            return data
            
        result = []
        for i in range(len(data)):
            if i < window - 1:
                result.append(data[i])
            else:
                avg = sum(data[i - window + 1:i + 1]) / window
                result.append(avg)
                
        return result
        
    def exponential_moving_average(self, data: List[float], 
                                 alpha: float = 0.1) -> List[float]:
        """Calculate exponential moving average"""
        if not data:
            return []
            
        ema = [data[0]]
        for i in range(1, len(data)):
            ema_value = alpha * data[i] + (1 - alpha) * ema[i - 1]
            ema.append(ema_value)
            
        return ema
        
    def calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
            
        return np.corrcoef(x, y)[0, 1]
        
    def calculate_volatility(self, returns: List[float], 
                           annualize: bool = True) -> float:
        """Calculate volatility (standard deviation of returns)"""
        if not returns or len(returns) < 2:
            return 0.0
            
        volatility = np.std(returns)
        
        if annualize:
            volatility *= math.sqrt(365)  # Assuming daily returns
            
        return volatility
        
    def normalize_data(self, data: List[float]) -> List[float]:
        """Normalize data to 0-1 range"""
        if not data:
            return []
            
        data_array = np.array(data)
        min_val = np.min(data_array)
        max_val = np.max(data_array)
        
        if max_val == min_val:
            return [0.5] * len(data)
            
        normalized = (data_array - min_val) / (max_val - min_val)
        return normalized.tolist()
        
    def calculate_z_score(self, value: float, mean: float, std: float) -> float:
        """Calculate z-score"""
        if std == 0:
            return 0.0
        return (value - mean) / std
        
    def detect_outliers(self, data: List[float], 
                       threshold: float = 2.0) -> List[int]:
        """Detect outliers using z-score"""
        if not data or len(data) < 3:
            return []
            
        data_array = np.array(data)
        mean = np.mean(data_array)
        std = np.std(data_array)
        
        outlier_indices = []
        for i, value in enumerate(data):
            z_score = self.calculate_z_score(value, mean, std)
            if abs(z_score) > threshold:
                outlier_indices.append(i)
                
        return outlier_indices
        
    def smooth_data(self, data: List[float], window: int = 3) -> List[float]:
        """Smooth data using moving average"""
        if len(data) <= window:
            return data
            
        smoothed = []
        half_window = window // 2
        
        for i in range(len(data)):
            start = max(0, i - half_window)
            end = min(len(data), i + half_window + 1)
            smoothed.append(sum(data[start:end]) / (end - start))
            
        return smoothed
        
    def calculate_performance_metrics(self, returns: List[float], 
                                    balance_history: List[float]) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if not returns or not balance_history:
            return {}
            
        try:
            metrics = {
                'total_return': (balance_history[-1] / balance_history[0] - 1) * 100,
                'annualized_return': self.calculate_compound_annual_growth_rate(
                    balance_history[0], balance_history[-1], len(returns) / 365
                ) * 100,
                'volatility': self.calculate_volatility(returns) * 100,
                'sharpe_ratio': self.calculate_sharpe_ratio(returns),
                'max_drawdown': self.calculate_maximum_drawdown(balance_history) * 100,
                'var_95': self.calculate_value_at_risk(returns, 0.95) * 100,
                'win_rate': len([r for r in returns if r > 0]) / len(returns) * 100,
                'avg_win': np.mean([r for r in returns if r > 0]) * 100 if any(r > 0 for r in returns) else 0,
                'avg_loss': np.mean([r for r in returns if r < 0]) * 100 if any(r < 0 for r in returns) else 0,
                'profit_factor': abs(sum(r for r in returns if r > 0) / sum(r for r in returns if r < 0)) if any(r < 0 for r in returns) else float('inf')
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Performance metrics calculation error: {str(e)}")
            return {}
            
    def interpolate_missing_values(self, data: List[Optional[float]]) -> List[float]:
        """Interpolate missing values in time series"""
        if not data:
            return []
            
        # Convert None to np.nan
        data_array = np.array([x if x is not None else np.nan for x in data])
        
        # Find indices of non-NaN values
        valid_indices = ~np.isnan(data_array)
        
        if not np.any(valid_indices):
            return [0.0] * len(data)
            
        # Interpolate
        indices = np.arange(len(data_array))
        data_array[~valid_indices] = np.interp(
            indices[~valid_indices], 
            indices[valid_indices], 
            data_array[valid_indices]
        )
        
        return data_array.tolist()
        
    def calculate_efficiency_ratio(self, prices: List[float], 
                                 period: int = 10) -> List[float]:
        """Calculate Efficiency Ratio for trend detection"""
        if len(prices) < period + 1:
            return [0.0] * len(prices)
            
        er_values = []
        
        for i in range(len(prices)):
            if i < period:
                er_values.append(0.0)
                continue
                
            # Direction (net change)
            direction = abs(prices[i] - prices[i - period])
            
            # Volatility (sum of absolute changes)
            volatility = sum(
                abs(prices[j] - prices[j - 1]) 
                for j in range(i - period + 1, i + 1)
            )
            
            if volatility == 0:
                er = 0.0
            else:
                er = direction / volatility
                
            er_values.append(er)
            
        return er_values
        
    def fibonacci_retracement_levels(self, high: float, low: float) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        diff = high - low
        
        return {
            '0%': high,
            '23.6%': high - 0.236 * diff,
            '38.2%': high - 0.382 * diff,
            '50%': high - 0.5 * diff,
            '61.8%': high - 0.618 * diff,
            '78.6%': high - 0.786 * diff,
            '100%': low
        }
import math
import logging

class MathUtils:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_percentage_change(self, old_value: float, new_value: float) -> float:
        """Calculate percentage change between two values"""
        if old_value == 0:
            return 0.0
        return ((new_value - old_value) / old_value) * 100
    
    def calculate_compound_interest(self, principal: float, rate: float, time: float) -> float:
        """Calculate compound interest"""
        return principal * math.pow(1 + rate, time)
    
    def kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly Criterion for position sizing"""
        if avg_loss <= 0 or win_rate <= 0:
            return 0.0
        
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        return max(0.0, min(kelly_fraction, 0.25))  # Cap at 25%
