import asyncio
import aiohttp
import json
import logging
from typing import List, Dict, Optional, Set, Any
from datetime import datetime, timedelta
from solana.rpc.async_api import AsyncClient
from solders.pubkey import Pubkey as PublicKey

class TokenScanner:
    def __init__(self, config, solana_client: AsyncClient):
        self.config = config
        self.client = solana_client
        self.logger = logging.getLogger(__name__)
        
        # Cache for discovered tokens
        self.known_tokens: Set[str] = set()
        self.active_tokens: List[Dict[str, Any]] = []
        self.last_scan_time = datetime.now() - timedelta(minutes=10)
        
        # Token sources
        self.token_sources = [
            "https://token.jup.ag/all",
            "https://registry.saber.so/data/llama.mainnet.json",
            "https://api.raydium.io/v2/sdk/liquidity/mainnet.json"
        ]
        
        # Initialize with popular tokens
        self.initialize_popular_tokens()
        
    def initialize_popular_tokens(self):
        """Initialize with popular Solana tokens"""
        popular_tokens = [
            {
                'address': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',  # USDC
                'symbol': 'USDC',
                'name': 'USD Coin',
                'liquidity': 1000000000,  # High liquidity
                'volume': 100000000,
                'source': 'popular'
            },
            {
                'address': 'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB',  # USDT
                'symbol': 'USDT',
                'name': 'Tether USD',
                'liquidity': 500000000,
                'volume': 80000000,
                'source': 'popular'
            },
            {
                'address': 'mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So',  # mSOL
                'symbol': 'mSOL',
                'name': 'Marinade staked SOL',
                'liquidity': 100000000,
                'volume': 20000000,
                'source': 'popular'
            },
            {
                'address': '7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs',  # ETH
                'symbol': 'ETH',
                'name': 'Ethereum (Wormhole)',
                'liquidity': 200000000,
                'volume': 50000000,
                'source': 'popular'
            },
            {
                'address': '3NZ9JMVBmGAqocybic2c7LQCJScmgsAZ6vQqTDzcqmJh',  # BTC
                'symbol': 'BTC',
                'name': 'Bitcoin (Wormhole)',
                'liquidity': 150000000,
                'volume': 40000000,
                'source': 'popular'
            }
        ]
        
        for token in popular_tokens:
            self.known_tokens.add(token['address'])
            self.active_tokens.append(token)
            
        self.logger.info(f"Initialized with {len(popular_tokens)} popular tokens")
        
    async def discover_new_tokens(self) -> List[Dict[str, Any]]:
        """Discover new tokens from various sources"""
        new_tokens = []
        
        try:
            # Scan different sources in parallel
            tasks = [
                self.scan_jupiter_tokens(),
                self.scan_raydium_pools(),
                self.scan_new_pairs(),
                self.scan_trending_tokens()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, list):
                    new_tokens.extend(result)
                elif isinstance(result, Exception):
                    self.logger.warning(f"Token discovery error: {str(result)}")
                    
        except Exception as e:
            self.logger.error(f"Token discovery failed: {str(e)}")
            
        # Filter and validate new tokens
        validated_tokens = []
        for token in new_tokens:
            if await self.validate_new_token(token):
                validated_tokens.append(token)
                self.known_tokens.add(token['address'])
                
        self.logger.info(f"Discovered {len(validated_tokens)} new tokens")
        return validated_tokens
        
    async def scan_jupiter_tokens(self) -> List[Dict[str, Any]]:
        """Scan Jupiter token list for new tokens"""
        new_tokens = []
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://token.jup.ag/all") as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for token in data:
                            address = token.get('address')
                            if address and address not in self.known_tokens:
                                # Get additional token info
                                token_info = await self.get_token_info(address)
                                if token_info:
                                    new_tokens.append({
                                        'address': address,
                                        'symbol': token.get('symbol', 'UNKNOWN'),
                                        'name': token.get('name', ''),
                                        'decimals': token.get('decimals', 9),
                                        'liquidity': token_info.get('liquidity', 0),
                                        'volume': token_info.get('volume_24h', 0),
                                        'source': 'jupiter',
                                        'discovered_at': datetime.now()
                                    })
                                    
        except Exception as e:
            self.logger.error(f"Jupiter token scan error: {str(e)}")
            
        return new_tokens[:10]  # Limit to 10 new tokens per scan
        
    async def scan_raydium_pools(self) -> List[Dict[str, Any]]:
        """Scan Raydium pools for new tokens"""
        new_tokens = []
        
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.raydium.io/v2/sdk/liquidity/mainnet.json"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for pool in data.get('official', []):
                            base_mint = pool.get('baseMint')
                            quote_mint = pool.get('quoteMint')
                            
                            # Check both base and quote tokens
                            for mint in [base_mint, quote_mint]:
                                if mint and mint not in self.known_tokens:
                                    # Skip SOL and common tokens
                                    if mint == 'So11111111111111111111111111111111111111112':
                                        continue
                                        
                                    new_tokens.append({
                                        'address': mint,
                                        'symbol': pool.get('baseSymbol' if mint == base_mint else 'quoteSymbol', 'UNKNOWN'),
                                        'name': '',
                                        'liquidity': float(pool.get('liquidity', 0)),
                                        'volume': float(pool.get('volume24h', 0)),
                                        'source': 'raydium',
                                        'pool_id': pool.get('id'),
                                        'discovered_at': datetime.now()
                                    })
                                    
        except Exception as e:
            self.logger.error(f"Raydium pool scan error: {str(e)}")
            
        return new_tokens[:10]
        
    async def scan_new_pairs(self) -> List[Dict[str, Any]]:
        """Scan for newly created trading pairs"""
        new_tokens = []
        
        try:
            # This would integrate with DEX program logs to find new pair creation events
            # For now, we'll use a simplified approach
            
            # Get recent transactions from known DEX programs
            recent_sigs = await self.get_recent_dex_transactions()
            
            for sig in recent_sigs:
                token_info = await self.extract_token_from_transaction(sig)
                if token_info and token_info['address'] not in self.known_tokens:
                    new_tokens.append(token_info)
                    
        except Exception as e:
            self.logger.error(f"New pairs scan error: {str(e)}")
            
        return new_tokens[:5]
        
    async def scan_trending_tokens(self) -> List[Dict[str, Any]]:
        """Scan for trending tokens based on volume spikes"""
        trending_tokens = []
        
        try:
            # This would analyze recent volume data to identify trending tokens
            # Implementation would depend on available APIs
            pass
            
        except Exception as e:
            self.logger.error(f"Trending tokens scan error: {str(e)}")
            
        return trending_tokens
        
    async def get_token_info(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Get additional information about a token"""
        try:
            # Try to get token info from Jupiter Price API
            async with aiohttp.ClientSession() as session:
                url = f"https://price.jup.ag/v4/price?ids={token_address}"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        token_data = data.get('data', {}).get(token_address)
                        
                        if token_data:
                            return {
                                'liquidity': float(token_data.get('liquidity', 0)),
                                'volume_24h': float(token_data.get('volume24h', 0)),
                                'price': float(token_data.get('price', 0)),
                                'market_cap': float(token_data.get('marketCap', 0))
                            }
                            
        except Exception as e:
            self.logger.debug(f"Token info fetch error for {token_address}: {str(e)}")
            
        return None
        
    async def validate_new_token(self, token: Dict[str, Any]) -> bool:
        """Validate if a new token meets our criteria"""
        try:
            address = token.get('address')
            
            # Basic validation
            if not address or len(address) < 32:
                return False
                
            # Liquidity requirements
            min_liquidity = self.config.optimization.min_trade_size * 1000  # 1000x min trade
            if token.get('liquidity', 0) < min_liquidity:
                return False
                
            # Volume requirements
            min_volume = token.get('liquidity', 0) * 0.1  # 10% of liquidity as daily volume
            if token.get('volume', 0) < min_volume:
                return False
                
            # Check if token exists on chain
            token_account = await self.get_token_account_info(address)
            if not token_account:
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Token validation error: {str(e)}")
            return False
            
    async def get_token_account_info(self, token_address: str) -> Optional[Dict]:
        """Get token account information from Solana"""
        try:
            pubkey = PublicKey(token_address)
            account_info = await self.client.get_account_info(pubkey)
            
            if account_info.value:
                return {
                    'exists': True,
                    'owner': str(account_info.value.owner),
                    'lamports': account_info.value.lamports
                }
                
        except Exception as e:
            self.logger.debug(f"Token account info error for {token_address}: {str(e)}")
            
        return None
        
    async def get_recent_dex_transactions(self) -> List[str]:
        """Get recent transactions from DEX programs"""
        signatures = []
        
        try:
            # Get signatures for known DEX programs
            dex_programs = [
                self.config.dexes.raydium.program_id,
                self.config.dexes.orca.program_id,
                self.config.dexes.jupiter.program_id
            ]
            
            for program_id in dex_programs:
                try:
                    pubkey = PublicKey(program_id)
                    sigs = await self.client.get_signatures_for_address(
                        pubkey,
                        limit=10
                    )
                    
                    for sig in sigs.value:
                        signatures.append(sig.signature)
                        
                except Exception as e:
                    self.logger.debug(f"Error getting signatures for {program_id}: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Recent transactions error: {str(e)}")
            
        return signatures[:20]  # Limit to 20 recent transactions
        
    async def extract_token_from_transaction(self, signature: str) -> Optional[Dict[str, Any]]:
        """Extract token information from a transaction"""
        try:
            # Get transaction details
            tx = await self.client.get_transaction(signature)
            
            if tx.value:
                # Parse transaction to find new token mints
                # This is a simplified implementation
                message = tx.value.transaction.message
                
                # Look for new token accounts in the transaction
                # Implementation would parse instruction data for token creation
                pass
                
        except Exception as e:
            self.logger.debug(f"Transaction extraction error for {signature}: {str(e)}")
            
        return None
        
    async def get_active_tokens(self) -> List[str]:
        """Get list of currently active tokens for arbitrage scanning"""
        # Update active tokens list periodically
        if datetime.now() - self.last_scan_time > timedelta(minutes=5):
            await self.update_active_tokens()
            
        # Return addresses of active tokens
        return [token['address'] for token in self.active_tokens if token.get('liquidity', 0) > 0]
        
    async def update_active_tokens(self):
        """Update the list of active tokens based on current market conditions"""
        try:
            updated_tokens = []
            
            for token in self.active_tokens:
                # Get current token info
                current_info = await self.get_token_info(token['address'])
                
                if current_info:
                    # Update token data
                    token.update(current_info)
                    
                    # Keep tokens with sufficient liquidity and volume
                    if (token.get('liquidity', 0) >= self.config.dexes.raydium.min_liquidity and
                        token.get('volume_24h', 0) > 0):
                        updated_tokens.append(token)
                        
            # Sort by liquidity (highest first)
            self.active_tokens = sorted(
                updated_tokens,
                key=lambda x: x.get('liquidity', 0),
                reverse=True
            )[:50]  # Keep top 50 tokens
            
            self.last_scan_time = datetime.now()
            self.logger.info(f"Updated active tokens: {len(self.active_tokens)} tokens")
            
        except Exception as e:
            self.logger.error(f"Active tokens update error: {str(e)}")
            
    def get_token_stats(self) -> Dict[str, int]:
        """Get statistics about discovered tokens"""
        return {
            'total_known': len(self.known_tokens),
            'active_tokens': len(self.active_tokens),
            'high_liquidity': len([t for t in self.active_tokens if t.get('liquidity', 0) > 100000]),
            'popular_tokens': len([t for t in self.active_tokens if t.get('source') == 'popular'])
        }
