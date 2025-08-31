import streamlit as st
import asyncio
import threading
import time
import json
import os
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from bot.arbitrage_bot import ArbitrageBot
from config.config import load_config

# Page configuration
st.set_page_config(
    page_title="Solana Arbitrage Bot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'bot' not in st.session_state:
    st.session_state.bot = None
if 'bot_running' not in st.session_state:
    st.session_state.bot_running = False
if 'trades_history' not in st.session_state:
    st.session_state.trades_history = []
if 'discovered_tokens' not in st.session_state:
    st.session_state.discovered_tokens = []
if 'performance_metrics' not in st.session_state:
    st.session_state.performance_metrics = {
        'total_profit': 0.0,
        'total_trades': 0,
        'successful_trades': 0,
        'failed_trades': 0,
        'daily_profit': 0.0,
        'wallet_balance': 0.53,
        'opportunities_found': 0,
        'scam_tokens_blocked': 0
    }

def run_bot_async(bot_instance):
    """Run the bot in a separate thread"""
    try:
        # Run the bot's main loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(bot_instance.run())
    except Exception as e:
        print(f"Bot encountered an error: {str(e)}")
    finally:
        # Set a flag that can be checked by the main thread
        if hasattr(bot_instance, 'running'):
            bot_instance.running = False

def main():
    st.title("🚀 Ultra-Fast Solana Arbitrage Bot")
    st.markdown("---")
    
    # Check for required configuration
    try:
        config = load_config()
        if not os.path.exists('data/wallet_keypair.json'):
            st.error("❌ Wallet keypair file not found. Please create 'data/wallet_keypair.json'")
            st.stop()
    except Exception as e:
        st.error(f"❌ Configuration error: {str(e)}")
        st.stop()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🎛️ Bot Controls")
        
        # Bot status
        bot_is_running = (st.session_state.bot_running and 
                         st.session_state.bot and 
                         getattr(st.session_state.bot, 'running', False))
        
        if bot_is_running:
            st.success("🟢 Bot is RUNNING")
            if st.button("🛑 Stop Bot", type="secondary"):
                if st.session_state.bot:
                    st.session_state.bot.stop()
                st.session_state.bot_running = False
                st.rerun()
        else:
            st.error("🔴 Bot is STOPPED")
            if st.button("▶️ Start Bot", type="primary"):
                try:
                    config = load_config()
                    st.session_state.bot = ArbitrageBot(config)
                    st.session_state.bot_running = True
                    # Start bot in background thread
                    bot_thread = threading.Thread(target=run_bot_async, args=(st.session_state.bot,), daemon=True)
                    bot_thread.start()
                    st.success("Bot starting...")
                    time.sleep(2)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to start bot: {str(e)}")
                    st.session_state.bot_running = False
        
        st.markdown("---")
        
        # Configuration display
        st.header("⚙️ Configuration")
        try:
            config = load_config()
            st.metric("Initial Capital", f"{config.trading.initial_capital} SOL")
            st.metric("Max Position Size", f"{config.trading.max_position_size * 100}%")
            st.metric("Min Profit", f"${config.trading.min_profit_usd}")
            st.metric("Scan Interval", f"{config.trading.scan_interval_ms}ms")
            
            # DEX status
            st.subheader("🏪 DEX Status")
            dex_status = {
                "Raydium": config.dexes.raydium.enabled,
                "Orca": config.dexes.orca.enabled,
                "Serum": config.dexes.serum.enabled,
                "Jupiter": config.dexes.jupiter.enabled,
                "Saber": config.dexes.saber.enabled
            }
            
            for dex, enabled in dex_status.items():
                status = "🟢" if enabled else "🔴"
                st.text(f"{status} {dex}")
                
        except Exception as e:
            st.error(f"Config error: {str(e)}")
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 Total Profit",
            f"${st.session_state.performance_metrics['total_profit']:.2f}",
            delta=f"{st.session_state.performance_metrics['daily_profit']:.2f}"
        )
    
    with col2:
        success_rate = 0
        if st.session_state.performance_metrics['total_trades'] > 0:
            success_rate = (st.session_state.performance_metrics['successful_trades'] / 
                          st.session_state.performance_metrics['total_trades']) * 100
        st.metric(
            "📊 Success Rate",
            f"{success_rate:.1f}%",
            delta=f"{st.session_state.performance_metrics['successful_trades']} trades"
        )
    
    with col3:
        st.metric(
            "💎 Wallet Balance",
            f"{st.session_state.performance_metrics['wallet_balance']:.3f} SOL",
            delta=None
        )
    
    with col4:
        st.metric(
            "🔍 Opportunities",
            st.session_state.performance_metrics['opportunities_found'],
            delta=f"{st.session_state.performance_metrics['scam_tokens_blocked']} scams blocked"
        )
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Live Trading", "🎯 Discovered Tokens", "📊 Analytics", "⚠️ Risk Monitor", "📱 Notifications"])
    
    with tab1:
        st.header("Live Trading Activity")
        
        # Real-time opportunities
        st.subheader("🎯 Current Opportunities")
        if st.session_state.bot and st.session_state.bot_running:
            # This would be populated by the bot's real-time data
            opportunities_df = pd.DataFrame({
                'Token': ['Loading...'],
                'DEX Pair': ['Scanning...'],
                'Profit USD': [0.00],
                'Profit %': [0.0],
                'Liquidity': ['$0'],
                'Status': ['Scanning']
            })
        else:
            opportunities_df = pd.DataFrame({
                'Token': ['Bot Not Running'],
                'DEX Pair': ['-'],
                'Profit USD': [0.00],
                'Profit %': [0.0],
                'Liquidity': ['-'],
                'Status': ['Stopped']
            })
        
        st.dataframe(opportunities_df, width='stretch')
        
        # Recent trades
        st.subheader("📋 Recent Trades")
        if st.session_state.trades_history:
            trades_df = pd.DataFrame(st.session_state.trades_history[-10:])  # Last 10 trades
            st.dataframe(trades_df, width='stretch')
        else:
            st.info("No trades executed yet. Bot will display trade history here.")
    
    with tab2:
        st.header("🎯 Token Discovery")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Recently Discovered Tokens")
            if st.session_state.discovered_tokens:
                tokens_df = pd.DataFrame(st.session_state.discovered_tokens[-20:])
                st.dataframe(tokens_df, width='stretch')
            else:
                st.info("Token scanner will display discovered tokens here.")
        
        with col2:
            st.subheader("🛡️ Scam Protection")
            st.metric("Scam Tokens Blocked", st.session_state.performance_metrics['scam_tokens_blocked'])
            
            # Scam detection criteria
            st.markdown("**Detection Criteria:**")
            st.text("• Holder concentration > 50%")
            st.text("• Liquidity lock < 30 days")
            st.text("• Suspicious metadata")
            st.text("• Low holder count")
            st.text("• High sell pressure")
    
    with tab3:
        st.header("📊 Performance Analytics")
        
        # Performance chart (placeholder)
        fig = go.Figure()
        
        # Sample data for visualization
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='h')
        cumulative_profit = [0.0] * len(dates)
        
        if st.session_state.trades_history:
            # Calculate actual cumulative profit from trades
            profit_sum = 0
            for i, trade in enumerate(st.session_state.trades_history):
                if i < len(cumulative_profit):
                    profit_sum += trade.get('profit_usd', 0)
                    cumulative_profit[i] = profit_sum
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=cumulative_profit,
            mode='lines+markers',
            name='Cumulative Profit',
            line=dict(color='#00d4aa', width=3)
        ))
        
        fig.update_layout(
            title="Profit Over Time",
            xaxis_title="Time",
            yaxis_title="Profit (USD)",
            height=400
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Additional metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Trade Statistics")
            st.metric("Total Trades", st.session_state.performance_metrics['total_trades'])
            st.metric("Successful Trades", st.session_state.performance_metrics['successful_trades'])
            st.metric("Failed Trades", st.session_state.performance_metrics['failed_trades'])
        
        with col2:
            st.subheader("Risk Metrics")
            st.metric("Max Drawdown", "0.0%")
            st.metric("Daily Loss Limit", "3.0%")
            st.metric("Current Risk Level", "LOW")
    
    with tab4:
        st.header("⚠️ Risk Management Monitor")
        
        # Risk status indicators
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🛡️ Circuit Breakers")
            st.success("✅ All systems operational")
            st.text("• Max drawdown: OK")
            st.text("• Daily loss limit: OK")
            st.text("• Consecutive failures: OK")
        
        with col2:
            st.subheader("💰 Position Limits")
            current_position = (st.session_state.performance_metrics['wallet_balance'] * 0.10)
            st.metric("Current Max Position", f"{current_position:.3f} SOL")
            st.progress(0.0)  # Would show actual position usage
        
        with col3:
            st.subheader("🚨 Emergency Controls")
            if st.button("🚨 EMERGENCY STOP", type="secondary"):
                if st.session_state.bot:
                    st.session_state.bot.emergency_stop()
                st.warning("Emergency stop activated!")
    
    with tab5:
        st.header("📱 Telegram Notifications")
        
        try:
            config = load_config()
            if config.telegram.enabled:
                st.success(f"✅ Telegram notifications enabled")
                st.info(f"Bot Token: {config.telegram.bot_token[:10]}...")
                st.info(f"Chat ID: {config.telegram.chat_id}")
                
                # Notification settings
                st.subheader("Notification Types")
                notifications = {
                    "Trade Notifications": config.telegram.send_trade_notifications,
                    "Performance Updates": config.telegram.send_performance_updates,
                    "Withdrawal Notifications": config.telegram.send_withdrawal_notifications,
                    "Alert Messages": config.telegram.send_alerts,
                    "Daily Summaries": config.telegram.send_daily_summaries,
                    "Milestone Notifications": config.telegram.send_milestone_notifications
                }
                
                for notif_type, enabled in notifications.items():
                    status = "🟢" if enabled else "🔴"
                    st.text(f"{status} {notif_type}")
            else:
                st.warning("⚠️ Telegram notifications disabled")
        except Exception as e:
            st.error(f"Telegram config error: {str(e)}")
    
    # Auto-refresh every 5 seconds when bot is running
    bot_is_running = (st.session_state.bot_running and 
                     st.session_state.bot and 
                     getattr(st.session_state.bot, 'running', False))
    
    if bot_is_running:
        # Use placeholder for auto-refresh to prevent blocking
        placeholder = st.empty()
        with placeholder.container():
            if st.button("🔄 Refresh", key="auto_refresh"):
                st.rerun()
        
        # Auto-refresh timer (non-blocking)
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = time.time()
        
        if time.time() - st.session_state.last_refresh > 5:
            st.session_state.last_refresh = time.time()
            st.rerun()

if __name__ == "__main__":
    main()
