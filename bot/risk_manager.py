import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import math

@dataclass
class TradeHistory:
    timestamp: datetime
    token: str
    profit_loss: float
    success: bool
    
@dataclass
class RiskMetrics:
    current_drawdown: float
    max_drawdown: float
    daily_pnl: float
    consecutive_losses: int
    total_trades_today: int
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL

class RiskManager:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Trade history for risk calculations
        self.trade_history: List[TradeHistory] = []
        self.daily_trades: Dict[str, int] = {}  # date -> trade_count
        
        # Risk tracking
        self.starting_balance = config.trading.initial_capital
        self.peak_balance = config.trading.initial_capital
        self.current_balance = config.trading.initial_capital
        self.daily_start_balance = config.trading.initial_capital
        
        # Circuit breaker states
        self.circuit_breaker_active = False
        self.circuit_breaker_until = None
        self.consecutive_failures = 0
        
        # Performance tracking
        self.last_reset_date = datetime.now().date()
        
    def can_trade(self, current_wallet_balance: float, total_profit: float) -> bool:
        """Check if trading is allowed based on risk parameters"""
        try:
            # Update current balance
            self.current_balance = current_wallet_balance
            
            # Check circuit breaker
            if self.is_circuit_breaker_active():
                return False
                
            # Check minimum wallet balance
            if current_wallet_balance < self.config.risk.min_wallet_balance_sol:
                self.logger.warning(f"Wallet balance too low: {current_wallet_balance} SOL")
                return False
                
            # Check maximum drawdown
            if self.check_max_drawdown(total_profit):
                return False
                
            # Check daily loss limit
            if self.check_daily_loss_limit():
                return False
                
            # Check daily trade limit
            if self.check_daily_trade_limit():
                return False
                
            # Check consecutive failures
            if self.consecutive_failures >= self.config.risk.max_consecutive_failures:
                self.logger.warning(f"Too many consecutive failures: {self.consecutive_failures}")
                self.activate_circuit_breaker(minutes=self.config.risk.cooldown_after_loss_minutes)
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Risk check error: {str(e)}")
            return False  # Err on the side of caution
            
    def validate_trade(self, opportunity) -> bool:
        """Validate specific trade against risk parameters"""
        try:
            # Check trade size limits
            if opportunity.trade_size_sol > self.config.risk.max_trade_value_sol:
                self.logger.debug(f"Trade size too large: {opportunity.trade_size_sol} SOL")
                return False
                
            # Check position size as percentage of wallet
            position_pct = opportunity.trade_size_sol / self.current_balance
            if position_pct > self.config.risk.position_size_limit:
                self.logger.debug(f"Position size too large: {position_pct:.2%}")
                return False
                
            # Check minimum profit requirement
            if opportunity.profit_usd < self.config.trading.min_profit_usd:
                self.logger.debug(f"Profit too low: ${opportunity.profit_usd}")
                return False
                
            # Check confidence score
            if opportunity.confidence_score < 0.3:  # Minimum confidence threshold
                self.logger.debug(f"Confidence too low: {opportunity.confidence_score}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Trade validation error: {str(e)}")
            return False
            
    def check_max_drawdown(self, total_profit: float) -> bool:
        """Check if maximum drawdown limit is exceeded"""
        try:
            # Update peak balance
            current_total = self.starting_balance + total_profit
            if current_total > self.peak_balance:
                self.peak_balance = current_total
                
            # Calculate current drawdown
            current_drawdown = (self.peak_balance - current_total) / self.peak_balance
            
            if current_drawdown > self.config.risk.max_drawdown:
                self.logger.warning(f"Max drawdown exceeded: {current_drawdown:.2%}")
                self.activate_circuit_breaker(minutes=self.config.risk.cooldown_after_loss_minutes)
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Drawdown check error: {str(e)}")
            return True  # Stop trading on error
            
    def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit is exceeded"""
        try:
            today = datetime.now().date()
            
            # Reset daily tracking if new day
            if today != self.last_reset_date:
                self.daily_start_balance = self.current_balance
                self.last_reset_date = today
                
            # Calculate daily P&L
            daily_pnl = self.current_balance - self.daily_start_balance
            daily_loss_pct = abs(daily_pnl) / self.daily_start_balance if daily_pnl < 0 else 0
            
            if daily_loss_pct > self.config.risk.max_daily_loss:
                self.logger.warning(f"Daily loss limit exceeded: {daily_loss_pct:.2%}")
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Daily loss check error: {str(e)}")
            return True
            
    def check_daily_trade_limit(self) -> bool:
        """Check if daily trade limit is exceeded"""
        try:
            today = datetime.now().date().isoformat()
            current_trades = self.daily_trades.get(today, 0)
            
            if current_trades >= self.config.risk.max_trades_per_hour * 24:  # Assuming hourly limit applies to daily
                self.logger.warning(f"Daily trade limit exceeded: {current_trades}")
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Daily trade limit check error: {str(e)}")
            return False
            
    def record_trade(self, opportunity, profit_loss: float, success: bool):
        """Record trade for risk tracking"""
        try:
            # Add to trade history
            trade = TradeHistory(
                timestamp=datetime.now(),
                token=opportunity.token_address,
                profit_loss=profit_loss,
                success=success
            )
            self.trade_history.append(trade)
            
            # Update daily trade count
            today = datetime.now().date().isoformat()
            self.daily_trades[today] = self.daily_trades.get(today, 0) + 1
            
            # Update consecutive failures
            if success:
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
                
            # Clean old history (keep last 1000 trades)
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-1000:]
                
            self.logger.debug(f"Trade recorded: {opportunity.token_symbol} "
                            f"P&L: ${profit_loss:.2f} Success: {success}")
            
        except Exception as e:
            self.logger.error(f"Trade recording error: {str(e)}")
            
    def activate_circuit_breaker(self, minutes: int = 30):
        """Activate circuit breaker to stop trading"""
        self.circuit_breaker_active = True
        self.circuit_breaker_until = datetime.now() + timedelta(minutes=minutes)
        
        self.logger.warning(f"Circuit breaker activated for {minutes} minutes")
        
    def is_circuit_breaker_active(self) -> bool:
        """Check if circuit breaker is currently active"""
        if not self.circuit_breaker_active:
            return False
            
        if datetime.now() > self.circuit_breaker_until:
            self.circuit_breaker_active = False
            self.circuit_breaker_until = None
            self.logger.info("Circuit breaker deactivated")
            return False
            
        return True
        
    def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics"""
        try:
            # Calculate current drawdown
            current_drawdown = 0.0
            if self.peak_balance > 0:
                current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
                
            # Calculate max drawdown from history
            max_drawdown = self.calculate_max_historical_drawdown()
            
            # Calculate daily P&L
            daily_pnl = self.current_balance - self.daily_start_balance
            
            # Get today's trade count
            today = datetime.now().date().isoformat()
            trades_today = self.daily_trades.get(today, 0)
            
            # Determine risk level
            risk_level = self.calculate_risk_level(current_drawdown, daily_pnl)
            
            return RiskMetrics(
                current_drawdown=current_drawdown,
                max_drawdown=max_drawdown,
                daily_pnl=daily_pnl,
                consecutive_losses=self.consecutive_failures,
                total_trades_today=trades_today,
                risk_level=risk_level
            )
            
        except Exception as e:
            self.logger.error(f"Risk metrics calculation error: {str(e)}")
            return RiskMetrics(0, 0, 0, 0, 0, "UNKNOWN")
            
    def calculate_max_historical_drawdown(self) -> float:
        """Calculate maximum historical drawdown"""
        if not self.trade_history:
            return 0.0
            
        try:
            balance = self.starting_balance
            peak = balance
            max_dd = 0.0
            
            for trade in self.trade_history:
                balance += trade.profit_loss
                
                if balance > peak:
                    peak = balance
                    
                current_dd = (peak - balance) / peak if peak > 0 else 0
                max_dd = max(max_dd, current_dd)
                
            return max_dd
            
        except Exception as e:
            self.logger.error(f"Max drawdown calculation error: {str(e)}")
            return 0.0
            
    def calculate_risk_level(self, current_drawdown: float, daily_pnl: float) -> str:
        """Calculate current risk level"""
        try:
            risk_score = 0
            
            # Drawdown component
            if current_drawdown > 0.04:  # 4%
                risk_score += 3
            elif current_drawdown > 0.02:  # 2%
                risk_score += 2
            elif current_drawdown > 0.01:  # 1%
                risk_score += 1
                
            # Daily loss component
            daily_loss_pct = abs(daily_pnl) / self.daily_start_balance if daily_pnl < 0 else 0
            if daily_loss_pct > 0.02:  # 2%
                risk_score += 3
            elif daily_loss_pct > 0.01:  # 1%
                risk_score += 2
            elif daily_loss_pct > 0.005:  # 0.5%
                risk_score += 1
                
            # Consecutive failures component
            if self.consecutive_failures >= 3:
                risk_score += 2
            elif self.consecutive_failures >= 2:
                risk_score += 1
                
            # Circuit breaker component
            if self.circuit_breaker_active:
                risk_score += 4
                
            # Determine level
            if risk_score >= 6:
                return "CRITICAL"
            elif risk_score >= 4:
                return "HIGH"
            elif risk_score >= 2:
                return "MEDIUM"
            else:
                return "LOW"
                
        except Exception as e:
            self.logger.error(f"Risk level calculation error: {str(e)}")
            return "UNKNOWN"
            
    def calculate_optimal_position_size(self, opportunity, current_balance: float) -> float:
        """Calculate optimal position size using Kelly Criterion and risk limits"""
        try:
            # Get historical win rate and average returns
            win_rate, avg_win, avg_loss = self.calculate_historical_performance()
            
            if win_rate <= 0 or avg_loss <= 0:
                # Use conservative sizing if no history
                return min(
                    current_balance * self.config.trading.trade_size_pct_of_wallet,
                    self.config.trading.max_trade_size_sol
                )
                
            # Kelly Criterion: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Apply safety factor (use 25% of Kelly)
            safe_kelly = max(0, min(kelly_fraction * 0.25, self.config.risk.position_size_limit))
            
            # Calculate position size
            optimal_size = current_balance * safe_kelly
            
            # Apply absolute limits
            optimal_size = min(optimal_size, self.config.trading.max_trade_size_sol)
            optimal_size = max(optimal_size, self.config.trading.min_trade_size_sol)
            
            return optimal_size
            
        except Exception as e:
            self.logger.error(f"Position sizing error: {str(e)}")
            # Fallback to conservative sizing
            return min(
                current_balance * 0.05,  # 5% of balance
                self.config.trading.max_trade_size_sol
            )
            
    def calculate_historical_performance(self) -> tuple:
        """Calculate historical win rate and average returns"""
        if not self.trade_history:
            return 0.5, 0.0, 0.0  # Default values
            
        try:
            wins = [t for t in self.trade_history if t.success and t.profit_loss > 0]
            losses = [t for t in self.trade_history if not t.success or t.profit_loss <= 0]
            
            total_trades = len(self.trade_history)
            win_rate = len(wins) / total_trades if total_trades > 0 else 0
            
            avg_win = sum(t.profit_loss for t in wins) / len(wins) if wins else 0
            avg_loss = abs(sum(t.profit_loss for t in losses) / len(losses)) if losses else 0
            
            return win_rate, avg_win, avg_loss
            
        except Exception as e:
            self.logger.error(f"Performance calculation error: {str(e)}")
            return 0.5, 0.0, 0.0
            
    def should_reduce_size(self) -> bool:
        """Check if position sizes should be reduced"""
        risk_metrics = self.get_risk_metrics()
        
        return (
            risk_metrics.risk_level in ["HIGH", "CRITICAL"] or
            risk_metrics.current_drawdown > 0.03 or  # 3% drawdown
            self.consecutive_failures >= 2
        )
        
    def get_trade_statistics(self) -> Dict:
        """Get detailed trade statistics"""
        if not self.trade_history:
            return {}
            
        try:
            total_trades = len(self.trade_history)
            successful_trades = len([t for t in self.trade_history if t.success])
            failed_trades = total_trades - successful_trades
            
            total_profit = sum(t.profit_loss for t in self.trade_history)
            avg_profit_per_trade = total_profit / total_trades if total_trades > 0 else 0
            
            # Calculate additional metrics
            win_rate = successful_trades / total_trades * 100 if total_trades > 0 else 0
            
            profits = [t.profit_loss for t in self.trade_history if t.profit_loss > 0]
            losses = [t.profit_loss for t in self.trade_history if t.profit_loss <= 0]
            
            avg_win = sum(profits) / len(profits) if profits else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
            
            profit_factor = abs(sum(profits) / sum(losses)) if losses and sum(losses) != 0 else 0
            
            return {
                'total_trades': total_trades,
                'successful_trades': successful_trades,
                'failed_trades': failed_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'avg_profit_per_trade': avg_profit_per_trade,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'consecutive_failures': self.consecutive_failures,
                'max_consecutive_failures': self.config.risk.max_consecutive_failures
            }
            
        except Exception as e:
            self.logger.error(f"Statistics calculation error: {str(e)}")
            return {}
