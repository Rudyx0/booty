import asyncio
import aiohttp
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from solana.rpc.async_api import AsyncClient
from solders.keypair import Keypair
from solders.transaction import Transaction
from solders.system_program import TransferParams, transfer
from solders.pubkey import Pubkey

from .dex_connectors import DEXManager
from .token_scanner import TokenScanner
from .scam_detector import ScamDetector
from .risk_manager import RiskManager
from .telegram_notifier import TelegramNotifier
from .flash_loan import FlashLoanManager
from utils.solana_utils import SolanaUtils
from utils.math_utils import MathUtils

@dataclass
class ArbitrageOpportunity:
    token_address: str
    token_symbol: str
    buy_dex: str
    sell_dex: str
    buy_price: float
    sell_price: float
    profit_usd: float
    profit_percentage: float
    trade_size_sol: float
    liquidity_buy: float
    liquidity_sell: float
    gas_estimate: float
    confidence_score: float
    timestamp: datetime

@dataclass
class TradeResult:
    success: bool
    opportunity: ArbitrageOpportunity
    actual_profit: float
    gas_used: float
    execution_time: float
    error_message: Optional[str] = None
    transaction_signature: Optional[str] = None

class ArbitrageBot:
    def __init__(self, config):
        self.config = config
        self.running = False
        self.emergency_stopped = False
        
        # Initialize components
        self.setup_logging()
        self.load_wallet()
        self.initialize_components()
        
        # Performance tracking
        self.total_profit = 0.0
        self.total_trades = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.opportunities_found = 0
        
        # Compounding and withdrawal tracking
        self.current_balance = self.config.trading.initial_capital  # 0.53 SOL
        self.withdrawal_completed = False
        self.compounding_phase = True
        self.last_withdrawal_check = time.time()
        
        # Rate limiting
        self.last_request_time = 0
        self.request_count = 0
        self.rate_limit_window_start = time.time()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.monitoring.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_wallet(self):
        """Load wallet keypair from file"""
        try:
            with open('data/wallet_keypair.json', 'r') as f:
                keypair_data = json.load(f)
            self.wallet = Keypair.from_seed(bytes(keypair_data[:32]))
            self.logger.info(f"Wallet loaded: {self.wallet.pubkey()}")
        except Exception as e:
            self.logger.error(f"Failed to load wallet: {str(e)}")
            raise
            
    def initialize_components(self):
        """Initialize all bot components"""
        try:
            # Solana connection
            self.solana_client = AsyncClient(self.config.network.rpc_url)
            self.solana_utils = SolanaUtils(self.solana_client, self.config)
            
            # Core components
            self.dex_manager = DEXManager(self.config, self.solana_client)
            self.token_scanner = TokenScanner(self.config, self.solana_client)
            self.scam_detector = ScamDetector(self.config, self.solana_client)
            self.risk_manager = RiskManager(self.config)
            self.telegram_notifier = TelegramNotifier(self.config)
            self.flash_loan_manager = FlashLoanManager(self.config, self.solana_client)
            
            # Math utilities
            self.math_utils = MathUtils()
            
            self.logger.info("All components initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {str(e)}")
            raise
            
    async def run(self):
        """Main bot execution loop"""
        self.logger.info("Starting Arbitrage Bot...")
        self.running = True
        
        try:
            # Send startup notification
            await self.telegram_notifier.send_message(
                "🚀 Arbitrage Bot Started!\n"
                f"💰 Initial Capital: {self.config.trading.initial_capital} SOL\n"
                f"🎯 Min Profit: ${self.config.trading.min_profit_usd}\n"
                f"⚡ Scan Interval: {self.config.trading.scan_interval_ms}ms"
            )
            
            # Start background tasks
            tasks = [
                asyncio.create_task(self.main_arbitrage_loop()),
                asyncio.create_task(self.token_discovery_loop()),
                asyncio.create_task(self.performance_reporting_loop()),
                asyncio.create_task(self.risk_monitoring_loop()),
            ]
            
            # Run all tasks concurrently
            await asyncio.gather(*tasks)
            
        except Exception as e:
            self.logger.error(f"Bot execution error: {str(e)}")
            await self.telegram_notifier.send_message(f"❌ Bot Error: {str(e)}")
        finally:
            self.running = False
            await self.cleanup()
            
    async def main_arbitrage_loop(self):
        """Main arbitrage scanning and execution loop"""
        self.logger.info("🚀 Starting main arbitrage scanning loop...")
        scan_count = 0
        while self.running and not self.emergency_stopped:
            try:
                scan_count += 1
                
                # Add immediate scan confirmation
                if scan_count == 1 or scan_count % 3 == 0:
                    self.logger.info(f"⚡ Scan #{scan_count} starting...")
                
                # Rate limiting check
                if not await self.check_rate_limit():
                    self.logger.info(f"⏳ Rate limited, waiting...")
                    await asyncio.sleep(0.1)
                    continue
                
                # Get wallet balance
                wallet_balance = await self.solana_utils.get_wallet_balance(self.wallet.pubkey())
                
                # Log wallet balance every few scans
                if scan_count % 5 == 0:
                    self.logger.info(f"💰 Wallet balance: {wallet_balance:.3f} SOL")
                
                # Risk checks
                can_trade = self.risk_manager.can_trade(wallet_balance, self.total_profit)
                if not can_trade:
                    if scan_count % 10 == 0:  # Only log occasionally to avoid spam
                        self.logger.info(f"🚫 Risk manager blocking trades - wallet: {wallet_balance:.3f} SOL")
                    await asyncio.sleep(self.config.trading.scan_interval_ms / 1000)
                    continue
                
                # Log scanning activity every 5 scans to show it's working
                if scan_count % 5 == 0:
                    active_tokens = await self.token_scanner.get_active_tokens()
                    self.logger.info(f"🔍 Scan #{scan_count}: Checking {len(active_tokens)} tokens for arbitrage opportunities...")
                
                # Scan for opportunities
                opportunities = await self.scan_arbitrage_opportunities()
                
                if opportunities:
                    self.opportunities_found += len(opportunities)
                    self.logger.info(f"Found {len(opportunities)} opportunities!")
                    
                    # Execute best opportunity
                    best_opportunity = max(opportunities, key=lambda x: x.profit_usd)
                    self.logger.info(f"Best opportunity: {best_opportunity.token_symbol} - ${best_opportunity.profit_usd:.4f}")
                    
                    if await self.validate_opportunity(best_opportunity):
                        await self.execute_arbitrage(best_opportunity)
                    else:
                        self.logger.info("Opportunity validation failed")
                else:
                    # Log no opportunities found every 10 scans to show it's actively scanning
                    if scan_count % 10 == 0:
                        self.logger.info(f"💤 Scan #{scan_count}: No profitable opportunities found (min profit: ${self.config.trading.min_profit_usd})")
                
                # Reduced sleep time for faster scanning
                sleep_time = max(500, self.config.trading.scan_interval_ms // 2)  # At least 500ms
                await asyncio.sleep(sleep_time / 1000)
                
            except Exception as e:
                self.logger.error(f"Main loop error: {str(e)}")
                await asyncio.sleep(1)
                
    async def scan_arbitrage_opportunities(self) -> List[ArbitrageOpportunity]:
        """Scan all DEXs for arbitrage opportunities"""
        opportunities = []
        
        try:
            # Get all active tokens from token scanner
            active_tokens = await self.token_scanner.get_active_tokens()
            
            if not active_tokens:
                self.logger.debug("No active tokens found to scan")
                return opportunities
            
            self.logger.debug(f"Scanning {len(active_tokens)} active tokens")
            
            # Scan opportunities in parallel for speed
            tasks = []
            scan_limit = min(len(active_tokens), self.config.optimization.max_parallel_scans)
            for token in active_tokens[:scan_limit]:
                task = asyncio.create_task(self.scan_token_opportunities(token))
                tasks.append(task)
            
            # Gather results
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, list):
                    opportunities.extend(result)
                    if result:  # If opportunities found for this token
                        self.logger.debug(f"Token {active_tokens[i][:8]}... found {len(result)} opportunities")
                elif isinstance(result, Exception):
                    self.logger.warning(f"Scan error for token {active_tokens[i][:8]}...: {str(result)}")
                    
        except Exception as e:
            self.logger.error(f"Opportunity scanning error: {str(e)}")
            
        return opportunities
        
    async def scan_token_opportunities(self, token_address: str) -> List[ArbitrageOpportunity]:
        """Scan arbitrage opportunities for a specific token"""
        opportunities = []
        
        try:
            # Check if token is safe
            if not await self.scam_detector.is_token_safe(token_address):
                return opportunities
            
            # Get prices from all DEXs
            dex_prices = await self.dex_manager.get_token_prices(token_address)
            
            # Debug logging for price data
            if dex_prices:
                self.logger.debug(f"Token {token_address[:8]}: Got {len(dex_prices)} prices from DEXs: {list(dex_prices.keys())}")
                for dex, price_data in dex_prices.items():
                    self.logger.debug(f"  {dex}: ${price_data.get('price', 0):.6f}, liquidity: ${price_data.get('liquidity', 0):,.0f}")
            else:
                self.logger.debug(f"Token {token_address[:8]}: No price data from any DEX")
            
            if len(dex_prices) < 2:
                return opportunities
            
            # Find arbitrage opportunities
            for buy_dex, buy_data in dex_prices.items():
                for sell_dex, sell_data in dex_prices.items():
                    if buy_dex == sell_dex:
                        continue
                        
                    buy_price = buy_data['price']
                    sell_price = sell_data['price']
                    
                    # Calculate potential profit
                    if sell_price > buy_price:
                        profit_percentage = ((sell_price - buy_price) / buy_price) * 100
                        
                        # Calculate optimal trade size
                        trade_size = self.calculate_optimal_trade_size(
                            buy_data['liquidity'], 
                            sell_data['liquidity']
                        )
                        
                        # Calculate profit in USD
                        profit_usd = (sell_price - buy_price) * trade_size
                        
                        # Estimate gas costs
                        gas_estimate = await self.estimate_gas_cost(buy_dex, sell_dex)
                        
                        # Net profit after gas
                        net_profit_usd = profit_usd - gas_estimate
                        
                        # Check if profitable
                        if net_profit_usd >= self.config.trading.min_profit_usd:
                            opportunity = ArbitrageOpportunity(
                                token_address=token_address,
                                token_symbol=buy_data.get('symbol', 'UNKNOWN'),
                                buy_dex=buy_dex,
                                sell_dex=sell_dex,
                                buy_price=buy_price,
                                sell_price=sell_price,
                                profit_usd=net_profit_usd,
                                profit_percentage=profit_percentage,
                                trade_size_sol=trade_size,
                                liquidity_buy=buy_data['liquidity'],
                                liquidity_sell=sell_data['liquidity'],
                                gas_estimate=gas_estimate,
                                confidence_score=self.calculate_confidence_score(buy_data, sell_data),
                                timestamp=datetime.now()
                            )
                            opportunities.append(opportunity)
                            
        except Exception as e:
            self.logger.error(f"Token opportunity scan error for {token_address}: {str(e)}")
            
        return opportunities
        
    async def validate_opportunity(self, opportunity: ArbitrageOpportunity) -> bool:
        """Validate arbitrage opportunity before execution"""
        try:
            # Real-time price recheck
            current_prices = await self.dex_manager.get_current_prices(
                opportunity.token_address,
                [opportunity.buy_dex, opportunity.sell_dex]
            )
            
            if not current_prices:
                return False
            
            buy_price = current_prices.get(opportunity.buy_dex, {}).get('price', 0)
            sell_price = current_prices.get(opportunity.sell_dex, {}).get('price', 0)
            
            # Check if opportunity still exists
            if sell_price <= buy_price:
                return False
                
            # Recalculate profit
            new_profit = (sell_price - buy_price) * opportunity.trade_size_sol
            gas_cost = await self.estimate_gas_cost(opportunity.buy_dex, opportunity.sell_dex)
            net_profit = new_profit - gas_cost
            
            # Check if still profitable
            if net_profit < self.config.trading.min_profit_usd:
                return False
                
            # Risk validation
            if not self.risk_manager.validate_trade(opportunity):
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Opportunity validation error: {str(e)}")
            return False
            
    async def execute_arbitrage(self, opportunity: ArbitrageOpportunity) -> TradeResult:
        """Execute arbitrage trade"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Executing arbitrage: {opportunity.token_symbol} "
                           f"{opportunity.buy_dex} -> {opportunity.sell_dex} "
                           f"Profit: ${opportunity.profit_usd:.2f}")
            
            # Check if flash loan is beneficial
            if (opportunity.trade_size_sol > self.config.trading.initial_capital and 
                self.total_profit >= self.config.trading.flash_loan_threshold_usd):
                return await self.execute_flash_loan_arbitrage(opportunity)
            
            # Execute arbitrage (use flash loan if conditions are met)
            if self.should_use_flash_loan(opportunity):
                result = await self.execute_flash_loan_arbitrage(opportunity)
                await self.telegram_notifier.send_message(
                    f"⚡ Flash Loan Trade Executed!\n"
                    f"💰 Larger position size enabled"
                )
                return result
            else:
                return await self.execute_regular_arbitrage(opportunity)
            
        except Exception as e:
            self.logger.error(f"Arbitrage execution error: {str(e)}")
            return TradeResult(
                success=False,
                opportunity=opportunity,
                actual_profit=0.0,
                gas_used=0.0,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
            
    async def execute_regular_arbitrage(self, opportunity: ArbitrageOpportunity) -> TradeResult:
        """Execute regular arbitrage without flash loans"""
        start_time = time.time()
        total_gas_used = 0.0
        
        try:
            # Step 1: Buy from cheaper DEX
            buy_result = await self.dex_manager.execute_buy(
                dex_name=opportunity.buy_dex,
                token_address=opportunity.token_address,
                amount_sol=opportunity.trade_size_sol,
                slippage=self.config.trading.slippage_tolerance
            )
            
            if not buy_result['success']:
                raise Exception(f"Buy failed: {buy_result.get('error', 'Unknown error')}")
                
            total_gas_used += buy_result.get('gas_used', 0)
            token_amount = buy_result['token_amount']
            
            # Step 2: Sell on more expensive DEX
            sell_result = await self.dex_manager.execute_sell(
                dex_name=opportunity.sell_dex,
                token_address=opportunity.token_address,
                token_amount=token_amount,
                slippage=self.config.trading.slippage_tolerance
            )
            
            if not sell_result['success']:
                # Try to recover by selling on original DEX
                await self.dex_manager.execute_sell(
                    dex_name=opportunity.buy_dex,
                    token_address=opportunity.token_address,
                    token_amount=token_amount,
                    slippage=0.05  # Higher slippage for emergency sell
                )
                raise Exception(f"Sell failed: {sell_result.get('error', 'Unknown error')}")
                
            total_gas_used += sell_result.get('gas_used', 0)
            
            # Calculate actual profit
            sol_spent = opportunity.trade_size_sol
            sol_received = sell_result['sol_amount']
            actual_profit_sol = sol_received - sol_spent - total_gas_used
            actual_profit_usd = actual_profit_sol * await self.get_sol_price_usd()
            
            # Update statistics
            self.total_trades += 1
            if actual_profit_usd > 0:
                self.successful_trades += 1
                self.total_profit += actual_profit_usd
                # Update balance for compounding logic
                self.current_balance += actual_profit_sol
                
                # Check if we need to handle withdrawal/compounding
                await self.handle_profit_compounding()
            else:
                self.failed_trades += 1
            
            # Send notification
            await self.telegram_notifier.send_trade_notification(
                opportunity, actual_profit_usd, time.time() - start_time
            )
            
            return TradeResult(
                success=True,
                opportunity=opportunity,
                actual_profit=actual_profit_usd,
                gas_used=total_gas_used,
                execution_time=time.time() - start_time,
                transaction_signature=sell_result.get('signature')
            )
            
        except Exception as e:
            self.failed_trades += 1
            return TradeResult(
                success=False,
                opportunity=opportunity,
                actual_profit=0.0,
                gas_used=total_gas_used,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
            
    async def execute_flash_loan_arbitrage(self, opportunity: ArbitrageOpportunity) -> TradeResult:
        """Execute arbitrage using flash loans for larger positions"""
        try:
            return await self.flash_loan_manager.execute_flash_arbitrage(opportunity)
        except Exception as e:
            self.logger.error(f"Flash loan arbitrage error: {str(e)}")
            # Fallback to regular arbitrage with available capital
            return await self.execute_regular_arbitrage(opportunity)
            
    async def token_discovery_loop(self):
        """Background loop for discovering new tokens"""
        while self.running and not self.emergency_stopped:
            try:
                new_tokens = await self.token_scanner.discover_new_tokens()
                
                for token in new_tokens:
                    # Check if token is safe
                    if await self.scam_detector.is_token_safe(token['address']):
                        await self.telegram_notifier.send_message(
                            f"🆕 New Token Discovered!\n"
                            f"Symbol: {token['symbol']}\n"
                            f"Address: {token['address']}\n"
                            f"Liquidity: ${token['liquidity']:,.2f}\n"
                            f"Volume: ${token['volume']:,.2f}"
                        )
                    else:
                        await self.telegram_notifier.send_message(
                            f"🚫 Scam Token Blocked!\n"
                            f"Symbol: {token['symbol']}\n"
                            f"Reason: Failed safety checks"
                        )
                        
                # Sleep for discovery interval
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                self.logger.error(f"Token discovery error: {str(e)}")
                await asyncio.sleep(60)
                
    async def performance_reporting_loop(self):
        """Background loop for performance reporting"""
        while self.running and not self.emergency_stopped:
            try:
                # Send hourly performance update
                await asyncio.sleep(3600)  # 1 hour
                
                wallet_balance = await self.solana_utils.get_wallet_balance(self.wallet.pubkey())
                success_rate = (self.successful_trades / max(self.total_trades, 1)) * 100
                
                performance_message = (
                    f"📊 Performance Update\n"
                    f"💰 Total Profit: ${self.total_profit:.2f}\n"
                    f"💎 Wallet Balance: {wallet_balance:.3f} SOL\n"
                    f"📈 Success Rate: {success_rate:.1f}%\n"
                    f"🔍 Opportunities Found: {self.opportunities_found}\n"
                    f"✅ Successful Trades: {self.successful_trades}\n"
                    f"❌ Failed Trades: {self.failed_trades}"
                )
                
                await self.telegram_notifier.send_message(performance_message)
                
            except Exception as e:
                self.logger.error(f"Performance reporting error: {str(e)}")
                
    async def risk_monitoring_loop(self):
        """Background loop for risk monitoring"""
        while self.running and not self.emergency_stopped:
            try:
                # Check risk metrics every minute
                await asyncio.sleep(60)
                
                wallet_balance = await self.solana_utils.get_wallet_balance(self.wallet.pubkey())
                
                # Check for emergency conditions
                if wallet_balance < self.config.risk.min_wallet_balance_sol:
                    await self.telegram_notifier.send_message(
                        f"🚨 EMERGENCY: Wallet balance critical!\n"
                        f"Current: {wallet_balance:.3f} SOL\n"
                        f"Minimum: {self.config.risk.min_wallet_balance_sol} SOL\n"
                        f"Bot will stop trading."
                    )
                    self.emergency_stopped = True
                    
                # Check drawdown
                if self.risk_manager.check_max_drawdown(self.total_profit):
                    await self.telegram_notifier.send_message(
                        "⚠️ Max drawdown reached! Trading paused."
                    )
                    await asyncio.sleep(1800)  # 30-minute cooldown
                    
            except Exception as e:
                self.logger.error(f"Risk monitoring error: {str(e)}")
                
    def calculate_optimal_trade_size(self, buy_liquidity: float, sell_liquidity: float) -> float:
        """Calculate optimal trade size based on liquidity"""
        # Use smaller of the two liquidities and apply percentage limit
        min_liquidity = min(buy_liquidity, sell_liquidity)
        max_trade_by_liquidity = min_liquidity * self.config.trading.max_pct_of_pool
        
        # Apply wallet percentage limit
        wallet_balance = self.config.trading.initial_capital  # Simplified
        max_trade_by_wallet = wallet_balance * self.config.trading.trade_size_pct_of_wallet
        
        # Apply absolute limits
        max_trade = min(
            max_trade_by_liquidity,
            max_trade_by_wallet,
            self.config.trading.max_trade_size_sol
        )
        
        return max(max_trade, self.config.trading.min_trade_size_sol)
        
    def calculate_confidence_score(self, buy_data: dict, sell_data: dict) -> float:
        """Calculate confidence score for opportunity"""
        score = 0.0
        
        # Liquidity score (higher is better)
        min_liquidity = min(buy_data['liquidity'], sell_data['liquidity'])
        if min_liquidity > 100000:  # $100k+
            score += 0.3
        elif min_liquidity > 50000:  # $50k+
            score += 0.2
        elif min_liquidity > 10000:  # $10k+
            score += 0.1
            
        # Volume score
        avg_volume = (buy_data.get('volume_24h', 0) + sell_data.get('volume_24h', 0)) / 2
        if avg_volume > 1000000:  # $1M+ daily volume
            score += 0.3
        elif avg_volume > 100000:  # $100k+ daily volume
            score += 0.2
        elif avg_volume > 10000:  # $10k+ daily volume
            score += 0.1
            
        # DEX reputation score (including Meteora)
        dex_scores = {
            'raydium': 0.2,
            'orca': 0.2,
            'meteora': 0.15,
            'jupiter': 0.15,
            'serum': 0.1,
            'saber': 0.05
        }
        
        score += dex_scores.get(buy_data.get('dex', '').lower(), 0)
        score += dex_scores.get(sell_data.get('dex', '').lower(), 0)
        
        return min(score, 1.0)
        
    def calculate_dynamic_sleep(self) -> int:
        """Calculate dynamic sleep time based on performance"""
        base_sleep = self.config.trading.scan_interval_ms
        
        # Adjust based on recent success
        if self.successful_trades > self.failed_trades:
            return max(base_sleep // 2, self.config.trading.min_scan_interval_ms)
        else:
            return min(base_sleep * 2, self.config.trading.max_scan_interval_ms)
            
    async def check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        current_time = time.time()
        
        # Reset counter every second
        if current_time - self.rate_limit_window_start >= 1.0:
            self.request_count = 0
            self.rate_limit_window_start = current_time
            
        # Check if we can make a request
        if self.request_count >= self.config.network.rate_limit_per_second:
            return False
            
        self.request_count += 1
        return True
        
    async def estimate_gas_cost(self, buy_dex: str, sell_dex: str) -> float:
        """Estimate gas cost for arbitrage transaction"""
        # Base gas costs per DEX (in SOL)
        base_costs = {
            'raydium': 0.002,
            'orca': 0.0015,
            'meteora': 0.0018,
            'serum': 0.003,
            'jupiter': 0.001,
            'saber': 0.0025
        }
        
        buy_cost = base_costs.get(buy_dex, 0.002)
        sell_cost = base_costs.get(sell_dex, 0.002)
        
        # Apply gas multiplier
        total_cost = (buy_cost + sell_cost) * self.config.trading.gas_price_multiplier
        
        # Convert to USD
        sol_price = await self.get_sol_price_usd()
        return total_cost * sol_price
        
    async def handle_profit_compounding(self):
        """Handle the sophisticated compounding and withdrawal logic"""
        try:
            current_time = time.time()
            
            # Check if we should process withdrawal (every hour)
            if current_time - self.last_withdrawal_check < 3600:  # 1 hour
                return
                
            self.last_withdrawal_check = current_time
            
            # Get real-time wallet balance
            actual_balance = await self.solana_utils.get_wallet_balance(self.wallet.pubkey())
            self.current_balance = actual_balance
            
            # Phase 1: Compound from 0.53 SOL to 1.9 SOL
            if self.compounding_phase and not self.withdrawal_completed:
                if self.current_balance >= 1.9:
                    await self.execute_profit_withdrawal()
                else:
                    # Still compounding - send progress update
                    progress = ((self.current_balance - 0.53) / (1.9 - 0.53)) * 100
                    await self.telegram_notifier.send_message(
                        f"💎 Compounding Progress: {progress:.1f}%\n"
                        f"💰 Current Balance: {self.current_balance:.3f} SOL\n"
                        f"🎯 Target: 1.9 SOL (Need: {1.9 - self.current_balance:.3f} SOL)"
                    )
            
            # Phase 2: After withdrawal, check for flash loan threshold
            elif not self.compounding_phase and self.current_balance >= 2.3:
                # Enable flash loans for larger trades
                await self.telegram_notifier.send_message(
                    f"🚀 Flash Loan Threshold Reached!\n"
                    f"💰 Balance: {self.current_balance:.3f} SOL\n"
                    f"⚡ Large position trading now available"
                )
                
        except Exception as e:
            self.logger.error(f"Compounding logic error: {str(e)}")
            
    async def execute_profit_withdrawal(self):
        """Execute the 0.4 SOL withdrawal to target wallet"""
        try:
            withdrawal_amount = 0.4  # SOL
            target_wallet = self.config.profit_extraction.target_wallet
            
            # Create withdrawal transaction
            transaction = Transaction()
            transfer_instruction = transfer(
                TransferParams(
                    from_pubkey=self.wallet.pubkey(),
                    to_pubkey=target_wallet,
                    lamports=int(withdrawal_amount * 1e9)  # Convert SOL to lamports
                )
            )
            transaction.add(transfer_instruction)
            
            # Send transaction
            response = await self.solana_client.send_transaction(
                transaction, self.wallet
            )
            
            if response and response.value:
                # Update state
                self.current_balance -= withdrawal_amount
                self.withdrawal_completed = True
                self.compounding_phase = False
                
                await self.telegram_notifier.send_message(
                    f"💸 Profit Withdrawal Executed!\n"
                    f"💰 Amount: {withdrawal_amount} SOL ($${withdrawal_amount * await self.get_sol_price_usd():.2f})\n"
                    f"🎯 To Wallet: {target_wallet}\n"
                    f"📊 Remaining Balance: {self.current_balance:.3f} SOL\n"
                    f"✅ Transaction: {response.value}\n\n"
                    f"🎲 Bot will now maintain 1.9 SOL minimum and compound additional profits!"
                )
                
                self.logger.info(f"Withdrawal completed: {withdrawal_amount} SOL to {target_wallet}")
                
            else:
                raise Exception("Transaction failed")
                
        except Exception as e:
            self.logger.error(f"Withdrawal execution error: {str(e)}")
            await self.telegram_notifier.send_message(
                f"🚨 Withdrawal Failed!\n"
                f"Error: {str(e)}\n"
                f"Will retry on next cycle."
            )
            
    def should_use_flash_loan(self, opportunity: ArbitrageOpportunity) -> bool:
        """Determine if flash loan should be used for this opportunity"""
        # Only use flash loans after reaching 2.3 SOL threshold
        if self.current_balance < 2.3:
            return False
            
        # Use flash loan for high-profit opportunities
        if opportunity.profit_usd > 50 and opportunity.confidence_score > 0.7:
            return True
            
        # Use flash loan for large liquidity pools
        if min(opportunity.liquidity_buy, opportunity.liquidity_sell) > 100000:
            return True
            
        return False

    async def get_sol_price_usd(self) -> float:
        """Get current SOL price in USD"""
        try:
            # Use Jupiter API for SOL price
            async with aiohttp.ClientSession() as session:
                async with session.get('https://price.jup.ag/v4/price?ids=So11111111111111111111111111111111111111112') as response:
                    data = await response.json()
                    return data['data']['So11111111111111111111111111111111111111112']['price']
        except:
            return 200.0  # Fallback price
            
    def stop(self):
        """Stop the bot gracefully"""
        self.running = False
        
    def emergency_stop(self):
        """Emergency stop with immediate halt"""
        self.emergency_stopped = True
        self.running = False
        
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'solana_client'):
                await self.solana_client.close()
            self.logger.info("Bot cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")
