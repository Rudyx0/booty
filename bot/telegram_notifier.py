import asyncio
import aiohttp
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import traceback

class TelegramNotifier:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Telegram configuration
        self.enabled = config.telegram.enabled
        self.bot_token = config.telegram.bot_token
        self.chat_id = config.telegram.chat_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        
        # Message settings
        self.max_message_length = 4096
        self.rate_limit_delay = 1.0  # Minimum delay between messages
        self.last_message_time = 0
        
        if self.enabled:
            self.logger.info(f"Telegram notifications enabled for chat {self.chat_id}")
        else:
            self.logger.info("Telegram notifications disabled")
            
    async def send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """Send a message to Telegram"""
        if not self.enabled:
            return False
            
        try:
            # Rate limiting
            current_time = asyncio.get_event_loop().time()
            if current_time - self.last_message_time < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay)
                
            # Truncate message if too long
            if len(text) > self.max_message_length:
                text = text[:self.max_message_length - 100] + "\n\n... (message truncated)"
                
            payload = {
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': parse_mode
            }
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/sendMessage"
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        self.last_message_time = current_time
                        return True
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Telegram API error {response.status}: {error_text}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"Failed to send Telegram message: {str(e)}")
            return False
            
    async def send_trade_notification(self, opportunity, actual_profit: float, execution_time: float) -> bool:
        """Send trade execution notification"""
        if not self.config.telegram.send_trade_notifications:
            return False
            
        try:
            # Determine emoji based on success
            success_emoji = "✅" if actual_profit > 0 else "❌"
            profit_emoji = "💰" if actual_profit > 0 else "💸"
            
            message = (
                f"{success_emoji} **Trade Executed**\n\n"
                f"🎯 **Token**: {opportunity.token_symbol}\n"
                f"📊 **Strategy**: {opportunity.buy_dex} → {opportunity.sell_dex}\n"
                f"💵 **Trade Size**: {opportunity.trade_size_sol:.3f} SOL\n"
                f"{profit_emoji} **Actual Profit**: ${actual_profit:.4f}\n"
                f"📈 **Expected Profit**: ${opportunity.profit_usd:.4f}\n"
                f"⚡ **Execution Time**: {execution_time:.2f}s\n"
                f"🔢 **Buy Price**: ${opportunity.buy_price:.6f}\n"
                f"🔢 **Sell Price**: ${opportunity.sell_price:.6f}\n"
                f"💧 **Liquidity**: ${min(opportunity.liquidity_buy, opportunity.liquidity_sell):,.0f}\n"
                f"⏰ **Time**: {datetime.now().strftime('%H:%M:%S')}"
            )
            
            return await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Failed to send trade notification: {str(e)}")
            return False
            
    async def send_performance_update(self, metrics: Dict[str, Any]) -> bool:
        """Send performance update"""
        if not self.config.telegram.send_performance_updates:
            return False
            
        try:
            # Calculate success rate
            total_trades = metrics.get('total_trades', 0)
            successful_trades = metrics.get('successful_trades', 0)
            success_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Determine performance emoji
            if success_rate >= 80:
                perf_emoji = "🔥"
            elif success_rate >= 60:
                perf_emoji = "📈"
            elif success_rate >= 40:
                perf_emoji = "📊"
            else:
                perf_emoji = "⚠️"
                
            message = (
                f"{perf_emoji} **Performance Update**\n\n"
                f"💰 **Total Profit**: ${metrics.get('total_profit', 0):.2f}\n"
                f"💎 **Wallet Balance**: {metrics.get('wallet_balance', 0):.3f} SOL\n"
                f"📊 **Success Rate**: {success_rate:.1f}%\n"
                f"📈 **Total Trades**: {total_trades}\n"
                f"✅ **Successful**: {successful_trades}\n"
                f"❌ **Failed**: {metrics.get('failed_trades', 0)}\n"
                f"🔍 **Opportunities Found**: {metrics.get('opportunities_found', 0)}\n"
                f"🛡️ **Scams Blocked**: {metrics.get('scam_tokens_blocked', 0)}\n"
                f"⏰ **Updated**: {datetime.now().strftime('%H:%M:%S')}"
            )
            
            return await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Failed to send performance update: {str(e)}")
            return False
            
    async def send_discovery_notification(self, token: Dict[str, Any]) -> bool:
        """Send new token discovery notification"""
        try:
            message = (
                f"🆕 **New Token Discovered**\n\n"
                f"🎯 **Symbol**: {token.get('symbol', 'UNKNOWN')}\n"
                f"📝 **Name**: {token.get('name', 'Unknown')}\n"
                f"🔗 **Address**: `{token.get('address', '')}`\n"
                f"💧 **Liquidity**: ${token.get('liquidity', 0):,.2f}\n"
                f"📊 **Volume 24h**: ${token.get('volume', 0):,.2f}\n"
                f"🏪 **Source**: {token.get('source', 'Unknown').title()}\n"
                f"⏰ **Discovered**: {datetime.now().strftime('%H:%M:%S')}"
            )
            
            return await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Failed to send discovery notification: {str(e)}")
            return False
            
    async def send_scam_alert(self, token: Dict[str, Any], flags: list) -> bool:
        """Send scam token alert"""
        try:
            flags_text = "\n".join([f"• {flag.replace('_', ' ').title()}" for flag in flags])
            
            message = (
                f"🚫 **Scam Token Blocked**\n\n"
                f"⚠️ **Symbol**: {token.get('symbol', 'UNKNOWN')}\n"
                f"🔗 **Address**: `{token.get('address', '')}`\n"
                f"📊 **Risk Flags**:\n{flags_text}\n"
                f"🛡️ **Action**: Token blacklisted\n"
                f"⏰ **Time**: {datetime.now().strftime('%H:%M:%S')}"
            )
            
            return await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Failed to send scam alert: {str(e)}")
            return False
            
    async def send_withdrawal_notification(self, amount: float, target_wallet: str) -> bool:
        """Send withdrawal notification"""
        if not self.config.telegram.send_withdrawal_notifications:
            return False
            
        try:
            message = (
                f"💸 **Profit Withdrawal**\n\n"
                f"💰 **Amount**: {amount:.3f} SOL\n"
                f"💵 **USD Value**: ~${amount * 200:.2f}\n"  # Approximate USD value
                f"🎯 **Target Wallet**: `{target_wallet}`\n"
                f"📊 **Strategy**: Core bankroll protected\n"
                f"⏰ **Time**: {datetime.now().strftime('%H:%M:%S')}"
            )
            
            return await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Failed to send withdrawal notification: {str(e)}")
            return False
            
    async def send_flash_loan_notification(self, opportunity, loan_amount: float, profit: float) -> bool:
        """Send flash loan execution notification"""
        if not self.config.telegram.send_flash_loan_notifications:
            return False
            
        try:
            message = (
                f"⚡ **Flash Loan Executed**\n\n"
                f"🎯 **Token**: {opportunity.token_symbol}\n"
                f"💰 **Loan Amount**: {loan_amount:.3f} SOL\n"
                f"📊 **Strategy**: {opportunity.buy_dex} → {opportunity.sell_dex}\n"
                f"💵 **Profit**: ${profit:.4f}\n"
                f"📈 **ROI**: {(profit / (loan_amount * 200) * 100):.2f}%\n"  # Approximate ROI
                f"⚡ **Leverage**: Enhanced position size\n"
                f"⏰ **Time**: {datetime.now().strftime('%H:%M:%S')}"
            )
            
            return await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Failed to send flash loan notification: {str(e)}")
            return False
            
    async def send_risk_alert(self, alert_type: str, message_details: str) -> bool:
        """Send risk management alert"""
        if not self.config.telegram.send_alerts:
            return False
            
        try:
            risk_emojis = {
                'circuit_breaker': '🚨',
                'max_drawdown': '📉',
                'daily_loss': '💸',
                'low_balance': '⚠️',
                'consecutive_failures': '❌'
            }
            
            emoji = risk_emojis.get(alert_type, '⚠️')
            
            message = (
                f"{emoji} **Risk Alert**\n\n"
                f"🔴 **Type**: {alert_type.replace('_', ' ').title()}\n"
                f"📝 **Details**: {message_details}\n"
                f"🛡️ **Action**: Trading restrictions applied\n"
                f"⏰ **Time**: {datetime.now().strftime('%H:%M:%S')}"
            )
            
            return await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Failed to send risk alert: {str(e)}")
            return False
            
    async def send_daily_summary(self, summary: Dict[str, Any]) -> bool:
        """Send daily trading summary"""
        if not self.config.telegram.send_daily_summaries:
            return False
            
        try:
            total_trades = summary.get('total_trades', 0)
            success_rate = summary.get('success_rate', 0)
            
            # Performance emoji based on results
            if summary.get('daily_profit', 0) > 0 and success_rate > 60:
                summary_emoji = "🎉"
            elif summary.get('daily_profit', 0) > 0:
                summary_emoji = "📈"
            elif summary.get('daily_profit', 0) == 0:
                summary_emoji = "📊"
            else:
                summary_emoji = "📉"
                
            message = (
                f"{summary_emoji} **Daily Summary**\n\n"
                f"📅 **Date**: {datetime.now().strftime('%Y-%m-%d')}\n"
                f"💰 **Daily P&L**: ${summary.get('daily_profit', 0):.2f}\n"
                f"📊 **Total Trades**: {total_trades}\n"
                f"✅ **Success Rate**: {success_rate:.1f}%\n"
                f"💎 **Final Balance**: {summary.get('wallet_balance', 0):.3f} SOL\n"
                f"🔍 **Opportunities**: {summary.get('opportunities_found', 0)}\n"
                f"🛡️ **Scams Blocked**: {summary.get('scam_tokens_blocked', 0)}\n"
                f"⚡ **Best Trade**: ${summary.get('best_trade', 0):.4f}\n"
                f"🏆 **Status**: {summary.get('status', 'Active')}"
            )
            
            return await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Failed to send daily summary: {str(e)}")
            return False
            
    async def send_milestone_notification(self, milestone_type: str, value: float) -> bool:
        """Send milestone achievement notification"""
        if not self.config.telegram.send_milestone_notifications:
            return False
            
        try:
            milestone_messages = {
                'profit_target': f"🎯 **Profit Target Reached!**\n\nTotal Profit: ${value:.2f}",
                'wallet_growth': f"💎 **Wallet Milestone!**\n\nWallet Balance: {value:.3f} SOL",
                'trade_count': f"📊 **Trade Milestone!**\n\nTotal Trades: {int(value)}",
                'success_rate': f"🔥 **Performance Milestone!**\n\nSuccess Rate: {value:.1f}%"
            }
            
            base_message = milestone_messages.get(
                milestone_type, 
                f"🏆 **Milestone Achieved!**\n\nValue: {value}"
            )
            
            message = (
                f"{base_message}\n"
                f"🎉 **Achievement Unlocked!**\n"
                f"⏰ **Time**: {datetime.now().strftime('%H:%M:%S')}"
            )
            
            return await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Failed to send milestone notification: {str(e)}")
            return False
            
    async def send_startup_notification(self) -> bool:
        """Send bot startup notification"""
        try:
            message = (
                f"🚀 **Arbitrage Bot Started**\n\n"
                f"💰 **Initial Capital**: {self.config.trading.initial_capital} SOL\n"
                f"🎯 **Min Profit**: ${self.config.trading.min_profit_usd}\n"
                f"⚡ **Scan Interval**: {self.config.trading.scan_interval_ms}ms\n"
                f"🏪 **Active DEXs**: {self.count_active_dexs()}\n"
                f"🛡️ **Risk Management**: Enabled\n"
                f"📱 **Notifications**: Enabled\n"
                f"⏰ **Started**: {datetime.now().strftime('%H:%M:%S')}"
            )
            
            return await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Failed to send startup notification: {str(e)}")
            return False
            
    async def send_shutdown_notification(self, reason: str = "Manual stop") -> bool:
        """Send bot shutdown notification"""
        try:
            message = (
                f"🛑 **Arbitrage Bot Stopped**\n\n"
                f"📝 **Reason**: {reason}\n"
                f"⏰ **Stopped**: {datetime.now().strftime('%H:%M:%S')}\n"
                f"🔒 **Status**: Inactive"
            )
            
            return await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Failed to send shutdown notification: {str(e)}")
            return False
            
    async def send_error_notification(self, error: Exception, context: str = "") -> bool:
        """Send error notification"""
        try:
            error_msg = str(error)
            if len(error_msg) > 500:
                error_msg = error_msg[:500] + "..."
                
            message = (
                f"❌ **Bot Error**\n\n"
                f"🐛 **Error**: {error_msg}\n"
                f"📍 **Context**: {context}\n"
                f"🔧 **Action**: Check logs for details\n"
                f"⏰ **Time**: {datetime.now().strftime('%H:%M:%S')}"
            )
            
            return await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Failed to send error notification: {str(e)}")
            return False
            
    def count_active_dexs(self) -> int:
        """Count number of active DEXs"""
        count = 0
        if hasattr(self.config, 'dexes'):
            if hasattr(self.config.dexes, 'raydium') and self.config.dexes.raydium.enabled:
                count += 1
            if hasattr(self.config.dexes, 'orca') and self.config.dexes.orca.enabled:
                count += 1
            if hasattr(self.config.dexes, 'jupiter') and self.config.dexes.jupiter.enabled:
                count += 1
            if hasattr(self.config.dexes, 'serum') and self.config.dexes.serum.enabled:
                count += 1
            if hasattr(self.config.dexes, 'saber') and self.config.dexes.saber.enabled:
                count += 1
        return count
        
    async def test_connection(self) -> bool:
        """Test Telegram connection"""
        if not self.enabled:
            return False
            
        try:
            test_message = "🧪 Telegram connection test - Bot is operational!"
            return await self.send_message(test_message)
        except Exception as e:
            self.logger.error(f"Telegram connection test failed: {str(e)}")
            return False
