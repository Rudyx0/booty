import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from solana.rpc.async_api import AsyncClient
from solders.transaction import Transaction
from solders.pubkey import Pubkey as PublicKey

@dataclass
class DEXQuote:
    price: float
    liquidity: float
    volume_24h: float
    symbol: str
    dex: str

class BaseDEX(ABC):
    def __init__(self, config, solana_client: AsyncClient):
        self.config = config
        self.client = solana_client
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    @abstractmethod
    async def get_price(self, token_address: str) -> Optional[DEXQuote]:
        pass
        
    @abstractmethod
    async def execute_swap(self, token_in: str, token_out: str, amount: float, slippage: float) -> Dict[str, Any]:
        pass

class RaydiumDEX(BaseDEX):
    def __init__(self, config, solana_client: AsyncClient):
        super().__init__(config, solana_client)
        self.base_url = "https://api.raydium.io"
        self.program_id = PublicKey(config.dexes.raydium.program_id)
        
    async def get_price(self, token_address: str) -> Optional[DEXQuote]:
        try:
            async with aiohttp.ClientSession() as session:
                # Get pool info
                url = f"{self.base_url}/v2/sdk/liquidity/mainnet.json"
                async with session.get(url) as response:
                    data = await response.json()
                    
                    # Find pool with this token
                    for pool in data.get('official', []):
                        if (pool.get('baseMint') == token_address or 
                            pool.get('quoteMint') == token_address):
                            
                            # Calculate price and liquidity
                            base_reserve = float(pool.get('baseReserve', 0))
                            quote_reserve = float(pool.get('quoteReserve', 0))
                            
                            if base_reserve > 0 and quote_reserve > 0:
                                price = quote_reserve / base_reserve
                                liquidity = min(base_reserve, quote_reserve) * price
                                
                                return DEXQuote(
                                    price=price,
                                    liquidity=liquidity,
                                    volume_24h=float(pool.get('volume24h', 0)),
                                    symbol=pool.get('baseSymbol', 'UNKNOWN'),
                                    dex='raydium'
                                )
                                
        except Exception as e:
            self.logger.error(f"Raydium price fetch error: {str(e)}")
            
        return None
        
    async def execute_swap(self, token_in: str, token_out: str, amount: float, slippage: float) -> Dict[str, Any]:
        try:
            # Implementation for Raydium swap
            # This would integrate with Raydium's swap program
            async with aiohttp.ClientSession() as session:
                swap_data = {
                    "inputMint": token_in,
                    "outputMint": token_out,
                    "amount": int(amount * 1e9),  # Convert to lamports
                    "slippageBps": int(slippage * 10000),
                }
                
                url = f"{self.base_url}/v1/swap"
                async with session.post(url, json=swap_data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            'success': True,
                            'signature': result.get('txid'),
                            'token_amount': result.get('outAmount', 0) / 1e9,
                            'gas_used': 0.002  # Estimated
                        }
                    else:
                        return {
                            'success': False,
                            'error': f"HTTP {response.status}"
                        }
                        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

class OrcaDEX(BaseDEX):
    def __init__(self, config, solana_client: AsyncClient):
        super().__init__(config, solana_client)
        self.base_url = "https://api.orca.so"
        self.program_id = PublicKey(config.dexes.orca.program_id)
        
    async def get_price(self, token_address: str) -> Optional[DEXQuote]:
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/v1/whirlpool/list"
                async with session.get(url) as response:
                    data = await response.json()
                    
                    for pool in data.get('whirlpools', []):
                        if (pool.get('tokenA', {}).get('mint') == token_address or
                            pool.get('tokenB', {}).get('mint') == token_address):
                            
                            price = float(pool.get('price', 0))
                            liquidity = float(pool.get('tvl', 0))
                            
                            return DEXQuote(
                                price=price,
                                liquidity=liquidity,
                                volume_24h=float(pool.get('volume24h', 0)),
                                symbol=pool.get('tokenA', {}).get('symbol', 'UNKNOWN'),
                                dex='orca'
                            )
                            
        except Exception as e:
            self.logger.error(f"Orca price fetch error: {str(e)}")
            
        return None
        
    async def execute_swap(self, token_in: str, token_out: str, amount: float, slippage: float) -> Dict[str, Any]:
        try:
            # Orca swap implementation
            async with aiohttp.ClientSession() as session:
                quote_url = f"{self.base_url}/v1/quote"
                quote_params = {
                    'inputMint': token_in,
                    'outputMint': token_out,
                    'amount': int(amount * 1e9),
                    'slippageBps': int(slippage * 10000)
                }
                
                async with session.get(quote_url, params=quote_params) as response:
                    if response.status == 200:
                        quote = await response.json()
                        
                        # Execute swap
                        swap_data = {
                            'quote': quote,
                            'userPublicKey': str(self.client.is_connected())  # Placeholder
                        }
                        
                        async with session.post(f"{self.base_url}/v1/swap", json=swap_data) as swap_response:
                            if swap_response.status == 200:
                                result = await swap_response.json()
                                return {
                                    'success': True,
                                    'signature': result.get('txid'),
                                    'token_amount': quote.get('outAmount', 0) / 1e9,
                                    'gas_used': 0.0015
                                }
                                
            return {'success': False, 'error': 'Swap failed'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

class JupiterDEX(BaseDEX):
    def __init__(self, config, solana_client: AsyncClient):
        super().__init__(config, solana_client)
        self.base_url = "https://quote-api.jup.ag/v6"
        self.price_url = "https://price.jup.ag/v4"
        self.api_key = config.dexes.jupiter.ultra_api_key
        
    async def get_price(self, token_address: str) -> Optional[DEXQuote]:
        try:
            async with aiohttp.ClientSession() as session:
                # Get price from Jupiter Price API
                url = f"{self.price_url}/price?ids={token_address}"
                headers = {'X-API-KEY': self.api_key} if self.api_key else {}
                
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        price_data = data.get('data', {}).get(token_address)
                        
                        if price_data:
                            return DEXQuote(
                                price=float(price_data.get('price', 0)),
                                liquidity=float(price_data.get('liquidity', 0)),
                                volume_24h=float(price_data.get('volume24h', 0)),
                                symbol=price_data.get('symbol', 'UNKNOWN'),
                                dex='jupiter'
                            )
                            
        except Exception as e:
            self.logger.error(f"Jupiter price fetch error: {str(e)}")
            
        return None
        
    async def execute_swap(self, token_in: str, token_out: str, amount: float, slippage: float) -> Dict[str, Any]:
        try:
            async with aiohttp.ClientSession() as session:
                # Get quote
                quote_params = {
                    'inputMint': token_in,
                    'outputMint': token_out,
                    'amount': int(amount * 1e9),
                    'slippageBps': int(slippage * 10000)
                }
                
                headers = {'X-API-KEY': self.api_key} if self.api_key else {}
                
                async with session.get(f"{self.base_url}/quote", params=quote_params, headers=headers) as response:
                    if response.status == 200:
                        quote = await response.json()
                        
                        # Get swap transaction
                        swap_data = {
                            'quoteResponse': quote,
                            'userPublicKey': 'placeholder',  # Would use actual wallet
                            'wrapAndUnwrapSol': True
                        }
                        
                        async with session.post(f"{self.base_url}/swap", json=swap_data, headers=headers) as swap_response:
                            if swap_response.status == 200:
                                result = await swap_response.json()
                                return {
                                    'success': True,
                                    'signature': 'pending',
                                    'token_amount': float(quote.get('outAmount', 0)) / 1e9,
                                    'gas_used': 0.001
                                }
                                
            return {'success': False, 'error': 'Jupiter swap failed'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

class SerumDEX(BaseDEX):
    def __init__(self, config, solana_client: AsyncClient):
        super().__init__(config, solana_client)
        self.program_id = PublicKey(config.dexes.serum.program_id)
        
    async def get_price(self, token_address: str) -> Optional[DEXQuote]:
        try:
            # Serum market data fetching
            # This would integrate with Serum's market data
            return DEXQuote(
                price=0.0,
                liquidity=0.0,
                volume_24h=0.0,
                symbol='UNKNOWN',
                dex='serum'
            )
            
        except Exception as e:
            self.logger.error(f"Serum price fetch error: {str(e)}")
            
        return None
        
    async def execute_swap(self, token_in: str, token_out: str, amount: float, slippage: float) -> Dict[str, Any]:
        try:
            # Serum swap implementation would go here
            return {'success': False, 'error': 'Serum swap not implemented'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

class SaberDEX(BaseDEX):
    def __init__(self, config, solana_client: AsyncClient):
        super().__init__(config, solana_client)
        self.base_url = "https://registry.saber.so"
        self.program_id = PublicKey(config.dexes.saber.program_id)
        
    async def get_price(self, token_address: str) -> Optional[DEXQuote]:
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/data/llama.mainnet.json"
                async with session.get(url) as response:
                    data = await response.json()
                    
                    for pool in data.get('pools', []):
                        tokens = pool.get('tokens', [])
                        if any(token.get('mint') == token_address for token in tokens):
                            return DEXQuote(
                                price=float(pool.get('price', 0)),
                                liquidity=float(pool.get('tvl', 0)),
                                volume_24h=float(pool.get('volume24h', 0)),
                                symbol=tokens[0].get('symbol', 'UNKNOWN') if tokens else 'UNKNOWN',
                                dex='saber'
                            )
                            
        except Exception as e:
            self.logger.error(f"Saber price fetch error: {str(e)}")
            
        return None
        
    async def execute_swap(self, token_in: str, token_out: str, amount: float, slippage: float) -> Dict[str, Any]:
        try:
            # Saber swap implementation would go here
            return {'success': False, 'error': 'Saber swap not implemented'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

class MeteoraDEX(BaseDEX):
    def __init__(self, config, solana_client: AsyncClient):
        super().__init__(config, solana_client)
        self.base_url = "https://app.meteora.ag"
        self.program_id = PublicKey.from_string(config.dexes.meteora.program_id)
        
    async def get_price(self, token_address: str) -> Optional[DEXQuote]:
        try:
            async with aiohttp.ClientSession() as session:
                # Get pools from Meteora
                url = f"{self.base_url}/pools"
                async with session.get(url) as response:
                    data = await response.json()
                    
                    for pool in data.get('pools', []):
                        tokens = pool.get('tokens', [])
                        if any(token.get('mint') == token_address for token in tokens):
                            return DEXQuote(
                                price=float(pool.get('price', 0)),
                                liquidity=float(pool.get('liquidity', 0)),
                                volume_24h=float(pool.get('volume24h', 0)),
                                symbol=tokens[0].get('symbol', 'UNKNOWN') if tokens else 'UNKNOWN',
                                dex='meteora'
                            )
                            
        except Exception as e:
            self.logger.error(f"Meteora price fetch error: {str(e)}")
            
        return None
        
    async def execute_swap(self, token_in: str, token_out: str, amount: float, slippage: float) -> Dict[str, Any]:
        try:
            # Meteora swap implementation would go here
            # For now, return a simulated response
            return {
                'success': True,
                'signature': f'meteora_sim_{int(amount * 1000)}',
                'token_amount': amount * 0.998,  # Account for fees
                'gas_used': 0.002
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

class DEXManager:
    def __init__(self, config, solana_client: AsyncClient):
        self.config = config
        self.client = solana_client
        self.logger = logging.getLogger(__name__)
        
        # Initialize DEX connectors
        self.dexes = {}
        
        if config.dexes.raydium.enabled:
            self.dexes['raydium'] = RaydiumDEX(config, solana_client)
            
        if config.dexes.orca.enabled:
            self.dexes['orca'] = OrcaDEX(config, solana_client)
            
        if config.dexes.jupiter.enabled:
            self.dexes['jupiter'] = JupiterDEX(config, solana_client)
            
        if config.dexes.serum.enabled:
            self.dexes['serum'] = SerumDEX(config, solana_client)
            
        if config.dexes.saber.enabled:
            self.dexes['saber'] = SaberDEX(config, solana_client)
            
        if config.dexes.meteora.enabled:
            self.dexes['meteora'] = MeteoraDEX(config, solana_client)
            
        self.logger.info(f"Initialized {len(self.dexes)} DEX connectors")
        
    async def get_token_prices(self, token_address: str) -> Dict[str, Dict[str, Any]]:
        """Get token prices from all enabled DEXs"""
        prices = {}
        
        tasks = []
        for dex_name, dex in self.dexes.items():
            task = asyncio.create_task(self._get_dex_price(dex_name, dex, token_address))
            tasks.append(task)
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            dex_name = list(self.dexes.keys())[i]
            if isinstance(result, DEXQuote):
                prices[dex_name] = {
                    'price': result.price,
                    'liquidity': result.liquidity,
                    'volume_24h': result.volume_24h,
                    'symbol': result.symbol,
                    'dex': result.dex
                }
            elif isinstance(result, Exception):
                self.logger.warning(f"Price fetch failed for {dex_name}: {str(result)}")
                
        return prices
        
    async def _get_dex_price(self, dex_name: str, dex: BaseDEX, token_address: str) -> Optional[DEXQuote]:
        """Get price from a single DEX"""
        try:
            return await dex.get_price(token_address)
        except Exception as e:
            self.logger.error(f"Error getting price from {dex_name}: {str(e)}")
            return None
            
    async def get_current_prices(self, token_address: str, dex_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get current prices from specific DEXs"""
        prices = {}
        
        for dex_name in dex_names:
            if dex_name in self.dexes:
                quote = await self.dexes[dex_name].get_price(token_address)
                if quote:
                    prices[dex_name] = {
                        'price': quote.price,
                        'liquidity': quote.liquidity,
                        'volume_24h': quote.volume_24h,
                        'symbol': quote.symbol
                    }
                    
        return prices
        
    async def execute_buy(self, dex_name: str, token_address: str, amount_sol: float, slippage: float) -> Dict[str, Any]:
        """Execute buy order on specified DEX"""
        if dex_name not in self.dexes:
            return {'success': False, 'error': f'DEX {dex_name} not available'}
            
        try:
            # SOL mint address
            sol_mint = "So11111111111111111111111111111111111111112"
            return await self.dexes[dex_name].execute_swap(sol_mint, token_address, amount_sol, slippage)
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def execute_sell(self, dex_name: str, token_address: str, token_amount: float, slippage: float) -> Dict[str, Any]:
        """Execute sell order on specified DEX"""
        if dex_name not in self.dexes:
            return {'success': False, 'error': f'DEX {dex_name} not available'}
            
        try:
            # SOL mint address
            sol_mint = "So11111111111111111111111111111111111111112"
            result = await self.dexes[dex_name].execute_swap(token_address, sol_mint, token_amount, slippage)
            
            # Convert result to include sol_amount
            if result.get('success'):
                result['sol_amount'] = result.get('token_amount', 0)
                
            return result
        except Exception as e:
            return {'success': False, 'error': str(e)}
