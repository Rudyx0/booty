import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from solana.rpc.async_api import AsyncClient
from solders.pubkey import Pubkey as PublicKey

class ScamDetector:
    def __init__(self, config, solana_client: AsyncClient):
        self.config = config
        self.client = solana_client
        self.logger = logging.getLogger(__name__)
        
        # Caches for performance
        self.token_safety_cache: Dict[str, Dict] = {}
        self.blacklisted_tokens: Set[str] = set()
        self.verified_tokens: Set[str] = set()
        
        # Initialize with known scam indicators
        self.initialize_known_lists()
        
    def initialize_known_lists(self):
        """Initialize with known safe and unsafe tokens"""
        # Known safe tokens
        safe_tokens = [
            'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',  # USDC
            'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB',  # USDT
            'mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So',  # mSOL
            '7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs',  # ETH
            '3NZ9JMVBmGAqocybic2c7LQCJScmgsAZ6vQqTDzcqmJh',  # BTC
            'So11111111111111111111111111111111111111112',    # SOL
        ]
        
        self.verified_tokens.update(safe_tokens)
        self.logger.info(f"Initialized with {len(safe_tokens)} verified safe tokens")
        
    async def is_token_safe(self, token_address: str) -> bool:
        """Main method to check if a token is safe to trade"""
        try:
            # Check cache first
            if token_address in self.token_safety_cache:
                cache_entry = self.token_safety_cache[token_address]
                if datetime.now() - cache_entry['timestamp'] < timedelta(hours=1):
                    return cache_entry['is_safe']
            
            # Check blacklist
            if token_address in self.blacklisted_tokens:
                return False
                
            # Check whitelist
            if token_address in self.verified_tokens:
                return True
                
            # Perform comprehensive safety check
            safety_result = await self.perform_safety_checks(token_address)
            
            # Cache result
            self.token_safety_cache[token_address] = {
                'is_safe': safety_result['is_safe'],
                'timestamp': datetime.now(),
                'risk_score': safety_result['risk_score'],
                'flags': safety_result['flags']
            }
            
            # Update blacklist if token is unsafe
            if not safety_result['is_safe']:
                self.blacklisted_tokens.add(token_address)
                
            return safety_result['is_safe']
            
        except Exception as e:
            self.logger.error(f"Token safety check error for {token_address}: {str(e)}")
            return False  # Err on the side of caution
            
    async def perform_safety_checks(self, token_address: str) -> Dict:
        """Perform comprehensive safety checks on a token"""
        risk_score = 0.0
        flags = []
        
        try:
            # Run all safety checks in parallel
            tasks = [
                self.check_holder_distribution(token_address),
                self.check_liquidity_lock(token_address),
                self.check_metadata_validity(token_address),
                self.check_mint_authority(token_address),
                self.check_freeze_authority(token_address),
                self.check_trading_volume(token_address),
                self.check_price_manipulation(token_address),
                self.check_honeypot_indicators(token_address)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Analyze results
            for i, result in enumerate(results):
                if isinstance(result, dict):
                    risk_score += result.get('risk_score', 0)
                    if result.get('flags'):
                        flags.extend(result['flags'])
                elif isinstance(result, Exception):
                    self.logger.warning(f"Safety check {i} failed: {str(result)}")
                    risk_score += 0.1  # Small penalty for failed checks
                    
            # Determine if token is safe based on risk score
            is_safe = risk_score < 0.5  # Threshold for safety
            
            return {
                'is_safe': is_safe,
                'risk_score': risk_score,
                'flags': flags
            }
            
        except Exception as e:
            self.logger.error(f"Safety checks failed for {token_address}: {str(e)}")
            return {
                'is_safe': False,
                'risk_score': 1.0,
                'flags': ['safety_check_failed']
            }
            
    async def check_holder_distribution(self, token_address: str) -> Dict:
        """Check token holder concentration"""
        risk_score = 0.0
        flags = []
        
        try:
            # Get token accounts for this mint
            response = await self.client.get_token_accounts_by_mint(PublicKey(token_address))
            
            if response.value:
                accounts = response.value
                total_accounts = len(accounts)
                
                # Get account balances
                balances = []
                for account in accounts[:100]:  # Check top 100 holders
                    try:
                        balance_response = await self.client.get_token_account_balance(
                            PublicKey(account.pubkey)
                        )
                        if balance_response.value:
                            balances.append(float(balance_response.value.amount))
                    except:
                        continue
                        
                if balances:
                    balances.sort(reverse=True)
                    total_supply = sum(balances)
                    
                    if total_supply > 0:
                        # Check top holder concentration
                        top_1_percent = balances[0] / total_supply * 100
                        top_10_percent = sum(balances[:min(10, len(balances))]) / total_supply * 100
                        
                        # Risk scoring
                        if top_1_percent > 50:
                            risk_score += 0.4
                            flags.append('top_holder_concentration_high')
                        elif top_1_percent > 30:
                            risk_score += 0.2
                            flags.append('top_holder_concentration_medium')
                            
                        if top_10_percent > 80:
                            risk_score += 0.3
                            flags.append('top_10_holders_concentration_high')
                            
                        # Check for insufficient holders
                        if total_accounts < 100:
                            risk_score += 0.2
                            flags.append('low_holder_count')
                            
        except Exception as e:
            self.logger.debug(f"Holder distribution check error: {str(e)}")
            risk_score += 0.1
            flags.append('holder_check_failed')
            
        return {
            'risk_score': risk_score,
            'flags': flags
        }
        
    async def check_liquidity_lock(self, token_address: str) -> Dict:
        """Check if liquidity is locked"""
        risk_score = 0.0
        flags = []
        
        try:
            # This would check if liquidity tokens are locked
            # Implementation depends on specific lock mechanisms used
            
            # For now, we'll check basic liquidity availability
            async with aiohttp.ClientSession() as session:
                url = f"https://price.jup.ag/v4/price?ids={token_address}"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        token_data = data.get('data', {}).get(token_address)
                        
                        if token_data:
                            liquidity = float(token_data.get('liquidity', 0))
                            
                            # Check liquidity levels
                            if liquidity < 10000:  # Less than $10k liquidity
                                risk_score += 0.3
                                flags.append('low_liquidity')
                            elif liquidity < 50000:  # Less than $50k liquidity
                                risk_score += 0.1
                                flags.append('medium_liquidity')
                                
        except Exception as e:
            self.logger.debug(f"Liquidity lock check error: {str(e)}")
            risk_score += 0.05
            flags.append('liquidity_check_failed')
            
        return {
            'risk_score': risk_score,
            'flags': flags
        }
        
    async def check_metadata_validity(self, token_address: str) -> Dict:
        """Check token metadata for suspicious indicators"""
        risk_score = 0.0
        flags = []
        
        try:
            # Get token metadata
            metadata = await self.get_token_metadata(token_address)
            
            if metadata:
                name = metadata.get('name', '').lower()
                symbol = metadata.get('symbol', '').lower()
                
                # Check for suspicious naming patterns
                suspicious_patterns = [
                    'test', 'fake', 'scam', 'rug', 'moon', 'safe',
                    'pump', 'dump', 'x100', 'x1000', 'gem'
                ]
                
                for pattern in suspicious_patterns:
                    if pattern in name or pattern in symbol:
                        risk_score += 0.1
                        flags.append(f'suspicious_name_{pattern}')
                        
                # Check for missing or invalid metadata
                if not name or not symbol:
                    risk_score += 0.2
                    flags.append('missing_metadata')
                    
                # Check for extremely long names (potential spam)
                if len(name) > 50 or len(symbol) > 10:
                    risk_score += 0.1
                    flags.append('suspicious_name_length')
                    
            else:
                risk_score += 0.3
                flags.append('no_metadata')
                
        except Exception as e:
            self.logger.debug(f"Metadata check error: {str(e)}")
            risk_score += 0.05
            flags.append('metadata_check_failed')
            
        return {
            'risk_score': risk_score,
            'flags': flags
        }
        
    async def check_mint_authority(self, token_address: str) -> Dict:
        """Check if mint authority is renounced or suspicious"""
        risk_score = 0.0
        flags = []
        
        try:
            # Get mint account info
            mint_info = await self.client.get_account_info(PublicKey(token_address))
            
            if mint_info.value:
                # Parse mint data to check authority
                # This is a simplified check - actual implementation would parse the account data
                
                # For safety, assume mint authority exists if we can't verify
                risk_score += 0.05
                flags.append('mint_authority_unknown')
                
        except Exception as e:
            self.logger.debug(f"Mint authority check error: {str(e)}")
            risk_score += 0.1
            flags.append('mint_check_failed')
            
        return {
            'risk_score': risk_score,
            'flags': flags
        }
        
    async def check_freeze_authority(self, token_address: str) -> Dict:
        """Check freeze authority status"""
        risk_score = 0.0
        flags = []
        
        try:
            # Similar to mint authority check
            # This would parse the mint account data for freeze authority
            
            # Placeholder implementation
            risk_score += 0.05
            flags.append('freeze_authority_unknown')
            
        except Exception as e:
            self.logger.debug(f"Freeze authority check error: {str(e)}")
            risk_score += 0.05
            flags.append('freeze_check_failed')
            
        return {
            'risk_score': risk_score,
            'flags': flags
        }
        
    async def check_trading_volume(self, token_address: str) -> Dict:
        """Check trading volume patterns for manipulation"""
        risk_score = 0.0
        flags = []
        
        try:
            # Get volume data from Jupiter
            async with aiohttp.ClientSession() as session:
                url = f"https://price.jup.ag/v4/price?ids={token_address}"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        token_data = data.get('data', {}).get(token_address)
                        
                        if token_data:
                            volume_24h = float(token_data.get('volume24h', 0))
                            liquidity = float(token_data.get('liquidity', 0))
                            
                            # Check volume/liquidity ratio
                            if liquidity > 0:
                                volume_ratio = volume_24h / liquidity
                                
                                # Very high volume relative to liquidity might indicate manipulation
                                if volume_ratio > 10:
                                    risk_score += 0.2
                                    flags.append('high_volume_ratio')
                                elif volume_ratio > 5:
                                    risk_score += 0.1
                                    flags.append('medium_volume_ratio')
                                    
                            # Check for extremely low volume
                            if volume_24h < 1000:  # Less than $1k daily volume
                                risk_score += 0.1
                                flags.append('low_trading_volume')
                                
        except Exception as e:
            self.logger.debug(f"Trading volume check error: {str(e)}")
            risk_score += 0.05
            flags.append('volume_check_failed')
            
        return {
            'risk_score': risk_score,
            'flags': flags
        }
        
    async def check_price_manipulation(self, token_address: str) -> Dict:
        """Check for signs of price manipulation"""
        risk_score = 0.0
        flags = []
        
        try:
            # This would analyze price movements over time
            # For now, we'll do basic checks
            
            # Get current price data
            async with aiohttp.ClientSession() as session:
                url = f"https://price.jup.ag/v4/price?ids={token_address}"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        token_data = data.get('data', {}).get(token_address)
                        
                        if token_data:
                            # Check for extreme price volatility indicators
                            # This is a simplified check
                            price = float(token_data.get('price', 0))
                            
                            # Very low or very high prices might indicate manipulation
                            if price < 0.000001:  # Extremely low price
                                risk_score += 0.1
                                flags.append('extremely_low_price')
                                
        except Exception as e:
            self.logger.debug(f"Price manipulation check error: {str(e)}")
            risk_score += 0.05
            flags.append('price_check_failed')
            
        return {
            'risk_score': risk_score,
            'flags': flags
        }
        
    async def check_honeypot_indicators(self, token_address: str) -> Dict:
        """Check for honeypot indicators"""
        risk_score = 0.0
        flags = []
        
        try:
            # This would attempt small test transactions to check if selling is possible
            # For now, we'll implement basic checks
            
            # Check if token has any DEX listings
            has_dex_listing = await self.check_dex_listings(token_address)
            
            if not has_dex_listing:
                risk_score += 0.3
                flags.append('no_dex_listings')
                
        except Exception as e:
            self.logger.debug(f"Honeypot check error: {str(e)}")
            risk_score += 0.05
            flags.append('honeypot_check_failed')
            
        return {
            'risk_score': risk_score,
            'flags': flags
        }
        
    async def check_dex_listings(self, token_address: str) -> bool:
        """Check if token is listed on major DEXs"""
        try:
            # Check Jupiter for routes
            async with aiohttp.ClientSession() as session:
                url = f"https://quote-api.jup.ag/v6/quote"
                params = {
                    'inputMint': 'So11111111111111111111111111111111111111112',  # SOL
                    'outputMint': token_address,
                    'amount': 1000000  # 0.001 SOL
                }
                
                async with session.get(url, params=params) as response:
                    return response.status == 200
                    
        except Exception as e:
            self.logger.debug(f"DEX listing check error: {str(e)}")
            return False
            
    async def get_token_metadata(self, token_address: str) -> Optional[Dict]:
        """Get token metadata from various sources"""
        try:
            # Try Jupiter token list first
            async with aiohttp.ClientSession() as session:
                async with session.get("https://token.jup.ag/all") as response:
                    if response.status == 200:
                        tokens = await response.json()
                        
                        for token in tokens:
                            if token.get('address') == token_address:
                                return {
                                    'name': token.get('name', ''),
                                    'symbol': token.get('symbol', ''),
                                    'decimals': token.get('decimals', 9),
                                    'logoURI': token.get('logoURI', '')
                                }
                                
        except Exception as e:
            self.logger.debug(f"Metadata fetch error: {str(e)}")
            
        return None
        
    def get_safety_report(self, token_address: str) -> Optional[Dict]:
        """Get cached safety report for a token"""
        return self.token_safety_cache.get(token_address)
        
    def get_blacklisted_tokens(self) -> Set[str]:
        """Get set of blacklisted token addresses"""
        return self.blacklisted_tokens.copy()
        
    def get_verified_tokens(self) -> Set[str]:
        """Get set of verified safe token addresses"""
        return self.verified_tokens.copy()
        
    def add_to_blacklist(self, token_address: str, reason: str = "manual"):
        """Manually add token to blacklist"""
        self.blacklisted_tokens.add(token_address)
        self.logger.info(f"Token {token_address} blacklisted: {reason}")
        
    def add_to_verified(self, token_address: str, reason: str = "manual"):
        """Manually add token to verified list"""
        self.verified_tokens.add(token_address)
        self.logger.info(f"Token {token_address} verified: {reason}")
        
    def clear_cache(self):
        """Clear safety cache"""
        self.token_safety_cache.clear()
        self.logger.info("Safety cache cleared")

