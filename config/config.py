import os
import json
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TradingConfig:
    initial_capital: float = 0.53
    min_profit_usd: float = 0.50
    max_trade_size_sol: float = 0.1
    min_trade_size_sol: float = 0.001
    max_position_size: float = 0.20  # 20% of wallet
    trade_size_pct_of_wallet: float = 0.05  # 5% of wallet per trade
    slippage_tolerance: float = 0.01  # 1%
    scan_interval_ms: int = 1000
    min_scan_interval_ms: int = 500
    max_scan_interval_ms: int = 5000
    max_pct_of_pool: float = 0.05  # 5% of pool liquidity
    flash_loan_threshold_usd: float = 100.0
    gas_price_multiplier: float = 1.2

@dataclass
class RiskConfig:
    max_drawdown: float = 0.10  # 10%
    max_daily_loss: float = 0.05  # 5%
    min_wallet_balance_sol: float = 0.01
    max_consecutive_failures: int = 3
    cooldown_after_loss_minutes: int = 30
    position_size_limit: float = 0.20  # 20%
    max_trade_value_sol: float = 0.1
    max_trades_per_hour: int = 10

@dataclass
class DEXConfig:
    enabled: bool = True
    program_id: str = ""
    min_liquidity: float = 1000.0

@dataclass
class DEXesConfig:
    raydium: DEXConfig
    orca: DEXConfig
    serum: DEXConfig
    jupiter: DEXConfig
    saber: DEXConfig

@dataclass
class NetworkConfig:
    rpc_url: str = "https://api.mainnet-beta.solana.com"
    cache_duration_seconds: int = 30
    rate_limit_per_second: int = 10
    request_timeout_seconds: int = 30

@dataclass
class TelegramConfig:
    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""
    send_trade_notifications: bool = True
    send_performance_updates: bool = True
    send_withdrawal_notifications: bool = True
    send_alerts: bool = True
    send_daily_summaries: bool = True
    send_milestone_notifications: bool = True

@dataclass
class MonitoringConfig:
    log_level: str = "INFO"

@dataclass
class OptimizationConfig:
    max_parallel_scans: int = 10
    min_trade_size: float = 0.001

@dataclass
class ProfitExtractionConfig:
    target_wallet: str = ""
    withdrawal_threshold_sol: float = 1.9
    withdrawal_amount_sol: float = 0.4

@dataclass
class Config:
    trading: TradingConfig
    risk: RiskConfig
    dexes: DEXesConfig
    network: NetworkConfig
    telegram: TelegramConfig
    monitoring: MonitoringConfig
    optimization: OptimizationConfig
    profit_extraction: ProfitExtractionConfig

def load_config() -> Config:
    """Load configuration from environment variables and defaults"""

    # DEX configurations
    raydium = DEXConfig(
        enabled=True,
        program_id="675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
        min_liquidity=1000.0
    )

    orca = DEXConfig(
        enabled=True,
        program_id="9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM",
        min_liquidity=1000.0
    )

    serum = DEXConfig(
        enabled=True,
        program_id="9xQeWvG816bUx9EPjHmaT23yvVM2ZWbrrpZb9PusVFin",
        min_liquidity=1000.0
    )

    jupiter = DEXConfig(
        enabled=True,
        program_id="JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",
        min_liquidity=1000.0
    )

    saber = DEXConfig(
        enabled=True,
        program_id="SSwpkEEcbUqx4vtoEByFjSkhKdCT862DNVb52nZg1UZ",
        min_liquidity=1000.0
    )

    dexes = DEXesConfig(
        raydium=raydium,
        orca=orca,
        serum=serum,
        jupiter=jupiter,
        saber=saber
    )

    # Main configuration
    trading = TradingConfig(
        initial_capital=float(os.getenv("INITIAL_CAPITAL", "0.53")),
        min_profit_usd=float(os.getenv("MIN_PROFIT_USD", "0.05")),
        max_trade_size_sol=float(os.getenv("MAX_TRADE_SIZE_SOL", "0.1")),
        min_trade_size_sol=float(os.getenv("MIN_TRADE_SIZE_SOL", "0.001")),
        max_position_size=float(os.getenv("MAX_POSITION_SIZE", "0.20")),
        trade_size_pct_of_wallet=float(os.getenv("TRADE_SIZE_PCT", "0.05")),
        slippage_tolerance=float(os.getenv("SLIPPAGE_TOLERANCE", "0.01")),
        scan_interval_ms=int(os.getenv("SCAN_INTERVAL_MS", "1000")),
        min_scan_interval_ms=int(os.getenv("MIN_SCAN_INTERVAL_MS", "500")),
        max_scan_interval_ms=int(os.getenv("MAX_SCAN_INTERVAL_MS", "5000")),
        max_pct_of_pool=float(os.getenv("MAX_PCT_OF_POOL", "0.05")),
        flash_loan_threshold_usd=float(os.getenv("FLASH_LOAN_THRESHOLD", "100.0")),
        gas_price_multiplier=float(os.getenv("GAS_PRICE_MULTIPLIER", "1.2"))
    )

    risk = RiskConfig(
        max_drawdown=float(os.getenv("MAX_DRAWDOWN", "0.10")),
        max_daily_loss=float(os.getenv("MAX_DAILY_LOSS", "0.05")),
        min_wallet_balance_sol=float(os.getenv("MIN_WALLET_BALANCE", "0.01")),
        max_consecutive_failures=int(os.getenv("MAX_CONSECUTIVE_FAILURES", "3")),
        cooldown_after_loss_minutes=int(os.getenv("COOLDOWN_MINUTES", "30")),
        position_size_limit=float(os.getenv("POSITION_SIZE_LIMIT", "0.20")),
        max_trade_value_sol=float(os.getenv("MAX_TRADE_VALUE", "0.1")),
        max_trades_per_hour=int(os.getenv("MAX_TRADES_PER_HOUR", "10"))
    )

    network = NetworkConfig(
        rpc_url=os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"),
        cache_duration_seconds=int(os.getenv("CACHE_DURATION", "30")),
        rate_limit_per_second=int(os.getenv("RATE_LIMIT_PER_SECOND", "5")),
        request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT", "30"))
    )

    telegram = TelegramConfig(
        enabled=os.getenv("TELEGRAM_ENABLED", "false").lower() == "true",
        bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        send_trade_notifications=os.getenv("TELEGRAM_TRADE_NOTIFICATIONS", "true").lower() == "true",
        send_performance_updates=os.getenv("TELEGRAM_PERFORMANCE_UPDATES", "true").lower() == "true",
        send_withdrawal_notifications=os.getenv("TELEGRAM_WITHDRAWAL_NOTIFICATIONS", "true").lower() == "true",
        send_alerts=os.getenv("TELEGRAM_ALERTS", "true").lower() == "true",
        send_daily_summaries=os.getenv("TELEGRAM_DAILY_SUMMARIES", "true").lower() == "true",
        send_milestone_notifications=os.getenv("TELEGRAM_MILESTONE_NOTIFICATIONS", "true").lower() == "true"
    )

    monitoring = MonitoringConfig(
        log_level=os.getenv("LOG_LEVEL", "INFO")
    )

    optimization = OptimizationConfig(
        max_parallel_scans=int(os.getenv("MAX_PARALLEL_SCANS", "10")),
        min_trade_size=float(os.getenv("MIN_TRADE_SIZE", "0.001"))
    )

    profit_extraction = ProfitExtractionConfig(
        target_wallet=os.getenv("TARGET_WALLET", ""),
        withdrawal_threshold_sol=float(os.getenv("WITHDRAWAL_THRESHOLD", "1.9")),
        withdrawal_amount_sol=float(os.getenv("WITHDRAWAL_AMOUNT", "0.4"))
    )

    return Config(
        trading=trading,
        risk=risk,
        dexes=dexes,
        network=network,
        telegram=telegram,
        monitoring=monitoring,
        optimization=optimization,
        profit_extraction=profit_extraction
    )