import asyncio
import aiohttp
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from solana.rpc.async_api import AsyncClient
from solders.pubkey import Pubkey

@dataclass
class DEXQuote:
    price: float
    liquidity: float
    volume_24h: float
    symbol: str
    dex: str

class BaseDEX:
    def __init__(self, config, solana_client: AsyncClient):
        self.config = config
        self.client = solana_client
        self.logger = logging.getLogger(__name__)
        self.last_request_time = 0
        self.rate_limit_delay = 1.0  # 1 second between requests for primary DEXs
        self.request_count = 0
        self.session = None
        self.consecutive_failures = 0
        self.backoff_until = 0

    async def _rate_limit(self):
        """Implement enhanced rate limiting with exponential backoff"""
        current_time = time.time()

        # Check if we're in backoff period
        if current_time < self.backoff_until:
            wait_time = self.backoff_until - current_time
            self.logger.info(f"Backing off for {wait_time:.1f}s due to failures")
            await asyncio.sleep(wait_time)

        # Apply normal rate limiting
        time_since_last = current_time - self.last_request_time
        effective_delay = self.rate_limit_delay * (2 ** min(self.consecutive_failures, 3))  # Max 8x delay

        if time_since_last < effective_delay:
            await asyncio.sleep(effective_delay - time_since_last)

        self.last_request_time = time.time()
        self.request_count += 1

    async def _get_session(self):
        """Get or create aiohttp session with timeout"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=15, connect=8)
            headers = {
                'User-Agent': 'SolanaArbitrageBot/1.0',
                'Accept': 'application/json',
                'Connection': 'keep-alive'
            }
            self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self.session

    def _handle_success(self):
        """Reset failure counters on successful request"""
        self.consecutive_failures = 0
        self.backoff_until = 0

    def _handle_failure(self, error_type: str = "general"):
        """Handle request failure with exponential backoff"""
        self.consecutive_failures += 1
        if error_type == "rate_limit":
            # Longer backoff for rate limiting
            backoff_time = min(30, 5 * (2 ** self.consecutive_failures))
        else:
            backoff_time = min(10, 1 * (2 ** self.consecutive_failures))

        self.backoff_until = time.time() + backoff_time
        self.logger.warning(f"Backing off for {backoff_time}s after {self.consecutive_failures} failures")

    async def _check_rate_limit(self, dex_name: str) -> bool:
        """Check if we can make a request to this DEX"""
        current_time = time.time()
        
        # Check if we're in backoff period
        if current_time < self.backoff_until:
            return False

        # Apply normal rate limiting
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            return False

        self.last_request_time = current_time
        return True

    def _get_backoff_time(self, dex_name: str) -> float:
        """Calculate backoff time with proper minimum"""
        backoff_time = max(0, self.backoff_until - time.time())
        return max(backoff_time, self.rate_limit_delay)  # Ensure minimum delay

    def _reset_backoff(self, dex_name: str):
        """Reset backoff on successful request"""
        self.consecutive_failures = 0
        self.backoff_until = 0

    async def get_price(self, token_address: str) -> Optional[DEXQuote]:
        raise NotImplementedError

    async def execute_swap(self, token_in: str, token_out: str, amount: float, slippage: float) -> Dict[str, Any]:
        raise NotImplementedError

class RaydiumDEX(BaseDEX):
    def __init__(self, config, solana_client: AsyncClient):
        super().__init__(config, solana_client)
        self.base_url = "https://api.raydium.io"
        self.program_id = Pubkey.from_string(config.dexes.raydium.program_id)
        self.dex_name = 'raydium' # Added for consistent logging and rate limiting

    async def get_price(self, token_address: str) -> Optional[DEXQuote]:
        try:
            if not self.config.dexes.raydium.enabled:
                return None

            # Check rate limiting with exponential backoff
            if not await self._check_rate_limit(self.dex_name):
                wait_time = self._get_backoff_time(self.dex_name)
                self.logger.warning(f"Rate limited by Raydium, waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
                return None

            session = await self._get_session()

            # Use simpler, more reliable endpoints
            endpoints = [
                "https://api-v3.raydium.io/pools/info/list",
                f"{self.base_url}/v2/main/pairs"
            ]

            for url in endpoints:
                try:
                    async with session.get(url) as response:
                        if response.status == 429:
                            self._handle_failure("rate_limit")
                            self.logger.warning(f"Rate limited by Raydium, backing off...")
                            return None

                        if response.status != 200:
                            self.logger.debug(f"Raydium returned status {response.status}")
                            continue

                        data = await response.json()
                        self._handle_success()

                        # Handle different response formats
                        pools = data.get('data', data.get('official', []))
                        if isinstance(pools, dict):
                            pools = list(pools.values())

                        for pool in pools:
                            if (pool.get('baseMint') == token_address or 
                                pool.get('quoteMint') == token_address):
                                # Reset backoff on successful request
                                self._reset_backoff(self.dex_name)

                                return DEXQuote(
                                    price=float(pool.get('price', pool.get('currentPrice', 0))),
                                    liquidity=float(pool.get('tvl', pool.get('liquidity', 0))),
                                    volume_24h=float(pool.get('volume24h', pool.get('volume', 0))),
                                    symbol=pool.get('baseSymbol', pool.get('symbol', 'UNKNOWN')),
                                    dex='raydium'
                                )
                        break
                except aiohttp.ClientResponseError as e:
                    if e.status == 429:
                        # Increment failure count and set backoff
                        self.consecutive_failures[self.dex_name] = self.consecutive_failures.get(self.dex_name, 0) + 1
                        backoff_time = self._get_backoff_time(self.dex_name)
                        self.logger.warning(f"Rate limited by Raydium, backing off for {backoff_time}s")
                        return None
                    else:
                        self.logger.error(f"Raydium HTTP error: {e.status}")
                        return None
                except asyncio.TimeoutError:
                    self._handle_failure()
                    self.logger.warning(f"Raydium timeout for {url}")
                    continue
                except Exception as e:
                    self.logger.debug(f"Raydium endpoint {url} failed: {str(e)}")
                    # Increment failure count for any error
                    self.consecutive_failures[self.dex_name] = self.consecutive_failures.get(self.dex_name, 0) + 1
                    continue

        except Exception as e:
            self._handle_failure()
            self.logger.error(f"Raydium price fetch error: {str(e)}")
        return None

    async def execute_swap(self, token_in: str, token_out: str, amount: float, slippage: float) -> Dict[str, Any]:
        try:
            # Simulated swap for now
            return {
                'success': True,
                'signature': f'raydium_sim_{int(amount * 1000)}',
                'token_amount': amount * 0.997,  # Account for fees
                'gas_used': 0.0025
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

class OrcaDEX(BaseDEX):
    def __init__(self, config, solana_client: AsyncClient):
        super().__init__(config, solana_client)
        self.base_url = "https://api.orca.so"
        self.program_id = Pubkey.from_string(config.dexes.orca.program_id)
        self.dex_name = 'orca' # Added for consistent logging and rate limiting

    async def get_price(self, token_address: str) -> Optional[DEXQuote]:
        try:
            # Check rate limiting with exponential backoff
            if not await self._check_rate_limit(self.dex_name):
                wait_time = self._get_backoff_time(self.dex_name)
                self.logger.warning(f"Rate limited by Orca, waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
                return None
                
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/v1/whirlpool/list"
                async with session.get(url) as response:
                    if response.status == 429:
                        self._handle_failure("rate_limit")
                        return None
                    if response.status != 200:
                        self.logger.debug(f"Orca returned status {response.status}")
                        return None
                        
                    data = await response.json()
                    self._handle_success()

                    for pool in data.get('whirlpools', []):
                        if (pool.get('tokenA', {}).get('mint') == token_address or
                            pool.get('tokenB', {}).get('mint') == token_address):

                            price = float(pool.get('price', 0))
                            liquidity = float(pool.get('tvl', 0))

                            # Reset backoff on successful request
                            self._reset_backoff(self.dex_name)

                            return DEXQuote(
                                price=price,
                                liquidity=liquidity,
                                volume_24h=float(pool.get('volume24h', 0)),
                                symbol=pool.get('tokenA', {}).get('symbol', 'UNKNOWN'),
                                dex='orca'
                            )
        except asyncio.TimeoutError:
            self._handle_failure()
            self.logger.warning(f"Orca timeout for {url}")
        except aiohttp.ClientResponseError as e:
            if e.status == 429:
                self._handle_failure("rate_limit")
            else:
                self.logger.error(f"Orca HTTP error: {e.status}")
        except Exception as e:
            self.logger.error(f"Orca price fetch error: {str(e)}")
            # Increment failure count for any error
            self.consecutive_failures[self.dex_name] = self.consecutive_failures.get(self.dex_name, 0) + 1
        return None

    async def execute_swap(self, token_in: str, token_out: str, amount: float, slippage: float) -> Dict[str, Any]:
        try:
            return {
                'success': True,
                'signature': f'orca_sim_{int(amount * 1000)}',
                'token_amount': amount * 0.998,
                'gas_used': 0.002
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

class JupiterDEX(BaseDEX):
    def __init__(self, config, solana_client: AsyncClient):
        super().__init__(config, solana_client)
        self.base_url = "https://quote-api.jup.ag"
        self.program_id = Pubkey.from_string(config.dexes.jupiter.program_id)
        self.dex_name = 'jupiter' # Added for consistent logging and rate limiting

    async def get_price(self, token_address: str) -> Optional[DEXQuote]:
        try:
            # Check rate limiting with exponential backoff
            if not await self._check_rate_limit(self.dex_name):
                wait_time = self._get_backoff_time(self.dex_name)
                self.logger.warning(f"Rate limited by Jupiter, waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
                return None
                
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/v6/quote?inputMint={token_address}&outputMint=So11111111111111111111111111111111111111112&amount=1000000"
                async with session.get(url) as response:
                    if response.status == 429:
                        self._handle_failure("rate_limit")
                        return None
                    if response.status != 200:
                        self.logger.debug(f"Jupiter returned status {response.status}")
                        return None

                    data = await response.json()
                    self._handle_success()

                    if 'outAmount' in data:
                        price = float(data['outAmount']) / 1000000
                        # Reset backoff on successful request
                        self._reset_backoff(self.dex_name)
                        
                        return DEXQuote(
                            price=price,
                            liquidity=float(data.get('contextSlot', 0)),
                            volume_24h=0.0,  # Not available from quote API
                            symbol='UNKNOWN',
                            dex='jupiter'
                        )
        except asyncio.TimeoutError:
            self._handle_failure()
            self.logger.warning(f"Jupiter timeout for {url}")
        except aiohttp.ClientResponseError as e:
            if e.status == 429:
                self._handle_failure("rate_limit")
            else:
                self.logger.error(f"Jupiter HTTP error: {e.status}")
        except Exception as e:
            self.logger.error(f"Jupiter price fetch error: {str(e)}")
            # Increment failure count for any error
            self.consecutive_failures[self.dex_name] = self.consecutive_failures.get(self.dex_name, 0) + 1
        return None

    async def execute_swap(self, token_in: str, token_out: str, amount: float, slippage: float) -> Dict[str, Any]:
        try:
            return {
                'success': True,
                'signature': f'jupiter_sim_{int(amount * 1000)}',
                'token_amount': amount * 0.999,
                'gas_used': 0.001
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

class DexscreenerDEX(BaseDEX):
    def __init__(self, config, solana_client: AsyncClient):
        super().__init__(config, solana_client)
        self.base_url = "https://api.dexscreener.com"
        self.rate_limit_delay = 2.0  # Be more conservative with Dexscreener
        self.dex_name = 'dexscreener' # Added for consistent logging and rate limiting

    async def get_price(self, token_address: str) -> Optional[DEXQuote]:
        try:
            # Check rate limiting with exponential backoff
            if not await self._check_rate_limit(self.dex_name):
                wait_time = self._get_backoff_time(self.dex_name)
                self.logger.warning(f"Rate limited by Dexscreener, waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
                return None

            session = await self._get_session()

            url = f"{self.base_url}/latest/dex/tokens/{token_address}"
            async with session.get(url) as response:
                if response.status == 429:
                    self._handle_failure("rate_limit")
                    return None
                if response.status != 200:
                    self.logger.debug(f"Dexscreener returned status {response.status}")
                    return None

                data = await response.json()
                self._handle_success()

                pairs = data.get('pairs', [])

                # Find best pair by liquidity
                best_pair = None
                max_liquidity = 0

                for pair in pairs:
                    if pair.get('chainId') == 'solana':
                        liquidity = float(pair.get('liquidity', {}).get('usd', 0))
                        if liquidity > max_liquidity:
                            max_liquidity = liquidity
                            best_pair = pair

                if best_pair:
                    # Reset backoff on successful request
                    self._reset_backoff(self.dex_name)

                    return DEXQuote(
                        price=float(best_pair.get('priceUsd', 0)),
                        liquidity=max_liquidity,
                        volume_24h=float(best_pair.get('volume', {}).get('h24', 0)),
                        symbol=best_pair.get('baseToken', {}).get('symbol', 'UNKNOWN'),
                        dex='dexscreener'
                    )
        except asyncio.TimeoutError:
            self._handle_failure()
            self.logger.warning(f"Dexscreener timeout for {url}")
        except aiohttp.ClientResponseError as e:
            if e.status == 429:
                self._handle_failure("rate_limit")
            else:
                self.logger.error(f"Dexscreener HTTP error: {e.status}")
        except Exception as e:
            self.logger.debug(f"Dexscreener price fetch error: {str(e)}")
            # Increment failure count for any error
            self.consecutive_failures[self.dex_name] = self.consecutive_failures.get(self.dex_name, 0) + 1
        return None

    async def execute_swap(self, token_in: str, token_out: str, amount: float, slippage: float) -> Dict[str, Any]:
        # Dexscreener is for price data only
        return {'success': False, 'error': 'Dexscreener is price data only'}

class MoonshotDEX(BaseDEX):
    def __init__(self, config, solana_client: AsyncClient):
        super().__init__(config, solana_client)
        self.base_url = "https://api.moonshot.cc"
        self.rate_limit_delay = 1.5
        self.dex_name = 'moonshot' # Added for consistent logging and rate limiting

    async def get_price(self, token_address: str) -> Optional[DEXQuote]:
        try:
            # Check rate limiting with exponential backoff
            if not await self._check_rate_limit(self.dex_name):
                wait_time = self._get_backoff_time(self.dex_name)
                self.logger.warning(f"Rate limited by Moonshot, waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
                return None

            session = await self._get_session()

            url = f"{self.base_url}/token/v1/solana/{token_address}"
            async with session.get(url) as response:
                if response.status == 429:
                    self._handle_failure("rate_limit")
                    return None
                if response.status != 200:
                    self.logger.debug(f"Moonshot returned status {response.status}")
                    return None

                data = await response.json()
                self._handle_success()

                # Reset backoff on successful request
                self._reset_backoff(self.dex_name)

                return DEXQuote(
                    price=float(data.get('price_usd', 0)),
                    liquidity=float(data.get('liquidity', 0)),
                    volume_24h=float(data.get('volume_24h', 0)),
                    symbol=data.get('symbol', 'UNKNOWN'),
                    dex='moonshot'
                )
        except asyncio.TimeoutError:
            self._handle_failure()
            self.logger.warning(f"Moonshot timeout for {url}")
        except aiohttp.ClientResponseError as e:
            if e.status == 429:
                self._handle_failure("rate_limit")
            else:
                self.logger.error(f"Moonshot HTTP error: {e.status}")
        except Exception as e:
            self.logger.debug(f"Moonshot price fetch error: {str(e)}")
            # Increment failure count for any error
            self.consecutive_failures[self.dex_name] = self.consecutive_failures.get(self.dex_name, 0) + 1
        return None

    async def execute_swap(self, token_in: str, token_out: str, amount: float, slippage: float) -> Dict[str, Any]:
        try:
            return {
                'success': True,
                'signature': f'moonshot_sim_{int(amount * 1000)}',
                'token_amount': amount * 0.995,
                'gas_used': 0.003
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

class PumpfunDEX(BaseDEX):
    def __init__(self, config, solana_client: AsyncClient):
        super().__init__(config, solana_client)
        self.base_url = "https://frontend-api.pump.fun"
        self.rate_limit_delay = 1.0
        self.dex_name = 'pumpfun' # Added for consistent logging and rate limiting

    async def get_price(self, token_address: str) -> Optional[DEXQuote]:
        try:
            # Check rate limiting with exponential backoff
            if not await self._check_rate_limit(self.dex_name):
                wait_time = self._get_backoff_time(self.dex_name)
                self.logger.warning(f"Rate limited by Pump.fun, waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
                return None

            session = await self._get_session()

            url = f"{self.base_url}/coins/{token_address}"
            async with session.get(url) as response:
                if response.status == 429:
                    self._handle_failure("rate_limit")
                    return None
                if response.status != 200:
                    self.logger.debug(f"Pump.fun returned status {response.status}")
                    return None

                data = await response.json()
                self._handle_success()

                # Reset backoff on successful request
                self._reset_backoff(self.dex_name)

                return DEXQuote(
                    price=float(data.get('usd_market_cap', 0)) / float(data.get('total_supply', 1)),
                    liquidity=float(data.get('virtual_sol_reserves', 0)) * 200,  # Approximate SOL price
                    volume_24h=float(data.get('volume_24h', 0)),
                    symbol=data.get('symbol', 'UNKNOWN'),
                    dex='pumpfun'
                )
        except asyncio.TimeoutError:
            self._handle_failure()
            self.logger.warning(f"Pump.fun timeout for {url}")
        except aiohttp.ClientResponseError as e:
            if e.status == 429:
                self._handle_failure("rate_limit")
            else:
                self.logger.error(f"Pump.fun HTTP error: {e.status}")
        except Exception as e:
            self.logger.debug(f"Pump.fun price fetch error: {str(e)}")
            # Increment failure count for any error
            self.consecutive_failures[self.dex_name] = self.consecutive_failures.get(self.dex_name, 0) + 1
        return None

    async def execute_swap(self, token_in: str, token_out: str, amount: float, slippage: float) -> Dict[str, Any]:
        try:
            return {
                'success': True,
                'signature': f'pumpfun_sim_{int(amount * 1000)}',
                'token_amount': amount * 0.993,
                'gas_used': 0.004
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

class FluxbeamDEX(BaseDEX):
    def __init__(self, config, solana_client: AsyncClient):
        super().__init__(config, solana_client)
        self.base_url = "https://api.fluxbeam.xyz"
        self.rate_limit_delay = 1.0
        self.dex_name = 'fluxbeam' # Added for consistent logging and rate limiting

    async def get_price(self, token_address: str) -> Optional[DEXQuote]:
        try:
            # Check rate limiting with exponential backoff
            if not await self._check_rate_limit(self.dex_name):
                wait_time = self._get_backoff_time(self.dex_name)
                self.logger.warning(f"Rate limited by Fluxbeam, waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
                return None

            session = await self._get_session()

            url = f"{self.base_url}/v1/tokens/{token_address}/price"
            async with session.get(url) as response:
                if response.status == 429:
                    self._handle_failure("rate_limit")
                    return None
                if response.status != 200:
                    self.logger.debug(f"Fluxbeam returned status {response.status}")
                    return None

                data = await response.json()
                self._handle_success()

                # Reset backoff on successful request
                self._reset_backoff(self.dex_name)

                return DEXQuote(
                    price=float(data.get('price', 0)),
                    liquidity=float(data.get('liquidity', 0)),
                    volume_24h=float(data.get('volume24h', 0)),
                    symbol=data.get('symbol', 'UNKNOWN'),
                    dex='fluxbeam'
                )
        except asyncio.TimeoutError:
            self._handle_failure()
            self.logger.warning(f"Fluxbeam timeout for {url}")
        except aiohttp.ClientResponseError as e:
            if e.status == 429:
                self._handle_failure("rate_limit")
            else:
                self.logger.error(f"Fluxbeam HTTP error: {e.status}")
        except Exception as e:
            self.logger.debug(f"Fluxbeam price fetch error: {str(e)}")
            # Increment failure count for any error
            self.consecutive_failures[self.dex_name] = self.consecutive_failures.get(self.dex_name, 0) + 1
        return None

    async def execute_swap(self, token_in: str, token_out: str, amount: float, slippage: float) -> Dict[str, Any]:
        try:
            return {
                'success': True,
                'signature': f'fluxbeam_sim_{int(amount * 1000)}',
                'token_amount': amount * 0.996,
                'gas_used': 0.0025
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

class DEXManager:
    def __init__(self, config, solana_client: AsyncClient):
        self.config = config
        self.client = solana_client
        self.logger = logging.getLogger(__name__)
        self.current_dex_index = 0

        # Initialize DEX connectors with priorities
        self.dexes = {}
        self.primary_dexes = []
        self.secondary_dexes = []

        # Rate limiting and backoff tracking per DEX
        self.rate_limits = {}
        self.last_requests = {}
        self.backoff_times = {}  # Track backoff times for each DEX
        self.consecutive_failures = {}  # Track consecutive failures

        if config.dexes.raydium.enabled:
            dex = RaydiumDEX(config, solana_client)
            self.dexes['raydium'] = dex
            self.primary_dexes.append('raydium')

        if config.dexes.orca.enabled:
            dex = OrcaDEX(config, solana_client)
            self.dexes['orca'] = dex
            self.primary_dexes.append('orca')

        if config.dexes.jupiter.enabled:
            dex = JupiterDEX(config, solana_client)
            self.dexes['jupiter'] = dex
            self.primary_dexes.append('jupiter')

        # Add secondary DEXs for more coverage
        self.dexes['dexscreener'] = DexscreenerDEX(config, solana_client)
        self.dexes['moonshot'] = MoonshotDEX(config, solana_client)
        self.dexes['pumpfun'] = PumpfunDEX(config, solana_client)
        self.dexes['fluxbeam'] = FluxbeamDEX(config, solana_client)

        self.secondary_dexes = ['dexscreener', 'moonshot', 'pumpfun', 'fluxbeam']

        self.logger.info(f"Initialized {len(self.primary_dexes)} primary and {len(self.secondary_dexes)} secondary DEXs")

    async def _check_rate_limit(self, dex_name: str) -> bool:
        """Check if we can make a request to this DEX"""
        current_time = time.time()
        last_request = self.last_requests.get(dex_name, 0)

        # Check if we're in backoff period
        backoff_until = self.backoff_times.get(dex_name, 0)
        if current_time < backoff_until:
            return False

        # Minimum time between requests (varies by DEX)
        min_interval = self._get_min_interval(dex_name)
        if current_time - last_request < min_interval:
            return False

        self.last_requests[dex_name] = current_time
        return True

    def _get_min_interval(self, dex_name: str) -> float:
        """Get minimum interval between requests for each DEX"""
        intervals = {
            'raydium': 2.0,  # Reduced from 5.0 to 2.0
            'orca': 1.5,
            'meteora': 1.5,
            'serum': 1.0,
            'jupiter': 1.0,
            'saber': 1.0,
            'dexscreener': 2.0,
            'moonshot': 1.5,
            'pumpfun': 1.0,
            'fluxbeam': 1.0
        }
        return intervals.get(dex_name, 1.5)

    def _get_backoff_time(self, dex_name: str) -> float:
        """Calculate exponential backoff time"""
        failures = self.consecutive_failures.get(dex_name, 0)
        base_time = 2.0
        max_time = 60.0  # Max 1 minute backoff

        backoff = min(base_time * (2 ** failures), max_time)
        self.backoff_times[dex_name] = time.time() + backoff
        return backoff

    def _reset_backoff(self, dex_name: str):
        """Reset backoff on successful request"""
        self.consecutive_failures[dex_name] = 0
        self.backoff_times[dex_name] = 0


    async def get_token_prices(self, token_address: str) -> Dict[str, Dict]:
        """Get token prices using smart distribution to avoid rate limits"""
        prices = {}

        # Try primary DEXs first with proper rate limiting
        for dex_name in self.primary_dexes:
            if dex_name in self.dexes and await self._check_rate_limit(dex_name):
                try:
                    self.logger.debug(f"Querying {dex_name} for token {token_address[:8]}...")
                    quote = await self.dexes[dex_name].get_price(token_address)
                    if quote:
                        prices[dex_name] = {
                            'price': quote.price,
                            'liquidity': quote.liquidity,
                            'volume_24h': quote.volume_24h,
                            'symbol': quote.symbol,
                            'dex': quote.dex
                        }
                        self._reset_backoff(dex_name)
                        self.logger.debug(f"{dex_name} returned price: ${quote.price:.6f}")
                        # If we get a good price from primary DEX, also try one secondary
                        break
                    else:
                        self.logger.debug(f"{dex_name} returned no price data")
                except Exception as e:
                    self.consecutive_failures[dex_name] = self.consecutive_failures.get(dex_name, 0) + 1
                    self.logger.debug(f"Primary DEX {dex_name} failed: {str(e)}")

                # Short delay between DEX attempts
                await asyncio.sleep(0.5)
            else:
                self.logger.debug(f"Rate limited for {dex_name}, skipping")

        # Try one secondary DEX for additional coverage
        if len(prices) > 0:  # Only if we got at least one price
            secondary_dex = self.secondary_dexes[self.current_dex_index % len(self.secondary_dexes)]
            self.current_dex_index += 1

            try:
                quote = await self.dexes[secondary_dex].get_price(token_address)
                if quote:
                    prices[secondary_dex] = {
                        'price': quote.price,
                        'liquidity': quote.liquidity,
                        'volume_24h': quote.volume_24h,
                        'symbol': quote.symbol,
                        'dex': quote.dex
                    }
            except Exception as e:
                self.logger.debug(f"Secondary DEX {secondary_dex} failed: {str(e)}")

        return prices

    async def get_current_prices(self, token_address: str, dex_names: List[str]) -> Dict[str, Dict]:
        """Get current prices from specific DEXs"""
        prices = {}

        for dex_name in dex_names:
            if dex_name in self.dexes:
                try:
                    quote = await self.dexes[dex_name].get_price(token_address)
                    if quote:
                        prices[dex_name] = {
                            'price': quote.price,
                            'liquidity': quote.liquidity
                        }
                except Exception as e:
                    self.logger.error(f"Error getting current price from {dex_name}: {str(e)}")

        return prices

    async def execute_buy(self, dex_name: str, token_address: str, amount_sol: float, slippage: float) -> Dict[str, Any]:
        """Execute buy order on specified DEX"""
        if dex_name not in self.dexes:
            return {'success': False, 'error': f'DEX {dex_name} not available'}

        try:
            return await self.dexes[dex_name].execute_swap(
                'So11111111111111111111111111111111111111112',  # SOL
                token_address,
                amount_sol,
                slippage
            )
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def execute_sell(self, dex_name: str, token_address: str, token_amount: float, slippage: float) -> Dict[str, Any]:
        """Execute sell order on specified DEX"""
        if dex_name not in self.dexes:
            return {'success': False, 'error': f'DEX {dex_name} not available'}

        try:
            return await self.dexes[dex_name].execute_swap(
                token_address,
                'So11111111111111111111111111111111111111112',  # SOL
                token_amount,
                slippage
            )
        except Exception as e:
            return {'success': False, 'error': str(e)}