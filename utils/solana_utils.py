import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Commitment
from solders.pubkey import Pubkey as PublicKey
from solders.keypair import Keypair
from solders.transaction import Transaction
from solders.system_program import TransferParams, transfer
from solana.rpc.types import TxOpts
import base58
import time

class SolanaUtils:
    def __init__(self, client: AsyncClient, config):
        self.client = client
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Cache for frequently accessed data
        self.account_cache: Dict[str, Dict] = {}
        self.cache_ttl = config.network.cache_duration_seconds
        
    async def get_wallet_balance(self, pubkey: PublicKey) -> float:
        """Get SOL balance for a wallet"""
        try:
            response = await self.client.get_balance(pubkey)
            if response.value is not None:
                return response.value / 1e9  # Convert lamports to SOL
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to get wallet balance: {str(e)}")
            return 0.0
            
    async def get_token_balance(self, pubkey: PublicKey, mint: PublicKey) -> float:
        """Get token balance for a specific mint"""
        try:
            # Get token accounts for this mint
            response = await self.client.get_token_accounts_by_mint(mint)
            
            if response.value:
                for account in response.value:
                    # Check if this account belongs to our wallet
                    account_info = await self.client.get_account_info(
                        PublicKey(account.pubkey)
                    )
                    
                    if account_info.value:
                        # Parse token account data to check owner
                        # Simplified implementation - would need proper parsing
                        balance_response = await self.client.get_token_account_balance(
                            PublicKey(account.pubkey)
                        )
                        
                        if balance_response.value:
                            return float(balance_response.value.amount) / (10 ** balance_response.value.decimals)
                            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to get token balance: {str(e)}")
            return 0.0
            
    async def send_transaction(self, transaction: Transaction, 
                             signers: List[Keypair],
                             max_retries: int = 3) -> Optional[str]:
        """Send transaction with retry logic"""
        for attempt in range(max_retries):
            try:
                # Get recent blockhash
                blockhash_response = await self.client.get_latest_blockhash()
                if not blockhash_response.value:
                    raise Exception("Failed to get recent blockhash")
                    
                transaction.recent_blockhash = blockhash_response.value.blockhash
                
                # Sign transaction
                transaction.sign(*signers)
                
                # Send transaction
                response = await self.client.send_transaction(
                    transaction,
                    opts=TxOpts(
                        skip_confirmation=False,
                        skip_preflight=False,
                        max_retries=1
                    )
                )
                
                if response.value:
                    # Confirm transaction
                    confirmed = await self.confirm_transaction(response.value)
                    if confirmed:
                        return response.value
                        
            except Exception as e:
                self.logger.warning(f"Transaction attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)  # Wait before retry
                    
        return None
        
    async def confirm_transaction(self, signature: str, 
                                timeout: int = 30) -> bool:
        """Confirm transaction with timeout"""
        try:
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                response = await self.client.get_signature_statuses([signature])
                
                if response.value and response.value[0]:
                    status = response.value[0]
                    
                    if status.confirmation_status in ['confirmed', 'finalized']:
                        if status.err:
                            self.logger.error(f"Transaction failed: {status.err}")
                            return False
                        return True
                        
                await asyncio.sleep(1)
                
            self.logger.warning(f"Transaction confirmation timeout: {signature}")
            return False
            
        except Exception as e:
            self.logger.error(f"Transaction confirmation error: {str(e)}")
            return False
            
    async def create_transfer_transaction(self, 
                                        from_pubkey: PublicKey,
                                        to_pubkey: PublicKey,
                                        lamports: int) -> Transaction:
        """Create a SOL transfer transaction"""
        try:
            # Create transfer instruction
            transfer_instruction = transfer(
                TransferParams(
                    from_pubkey=from_pubkey,
                    to_pubkey=to_pubkey,
                    lamports=lamports
                )
            )
            
            # Create transaction
            transaction = Transaction()
            transaction.add(transfer_instruction)
            
            return transaction
            
        except Exception as e:
            self.logger.error(f"Failed to create transfer transaction: {str(e)}")
            raise
            
    async def get_account_info_cached(self, pubkey: PublicKey) -> Optional[Dict]:
        """Get account info with caching"""
        try:
            pubkey_str = str(pubkey)
            current_time = time.time()
            
            # Check cache
            if pubkey_str in self.account_cache:
                cache_entry = self.account_cache[pubkey_str]
                if current_time - cache_entry['timestamp'] < self.cache_ttl:
                    return cache_entry['data']
                    
            # Fetch fresh data
            response = await self.client.get_account_info(pubkey)
            
            if response.value:
                account_data = {
                    'lamports': response.value.lamports,
                    'owner': str(response.value.owner),
                    'executable': response.value.executable,
                    'rent_epoch': response.value.rent_epoch,
                    'data_length': len(response.value.data) if response.value.data else 0
                }
                
                # Cache the result
                self.account_cache[pubkey_str] = {
                    'data': account_data,
                    'timestamp': current_time
                }
                
                return account_data
                
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get account info: {str(e)}")
            return None
            
    async def get_token_accounts(self, owner: PublicKey) -> List[Dict]:
        """Get all token accounts for an owner"""
        try:
            response = await self.client.get_token_accounts_by_owner(
                owner,
                mint=None  # Get all token accounts
            )
            
            accounts = []
            if response.value:
                for account in response.value:
                    try:
                        # Get balance for this token account
                        balance_response = await self.client.get_token_account_balance(
                            PublicKey(account.pubkey)
                        )
                        
                        if balance_response.value:
                            accounts.append({
                                'pubkey': account.pubkey,
                                'mint': account.account.data.parsed['info']['mint'],
                                'amount': balance_response.value.amount,
                                'decimals': balance_response.value.decimals,
                                'ui_amount': balance_response.value.ui_amount
                            })
                    except:
                        continue
                        
            return accounts
            
        except Exception as e:
            self.logger.error(f"Failed to get token accounts: {str(e)}")
            return []
            
    async def estimate_transaction_fee(self, transaction: Transaction) -> int:
        """Estimate transaction fee in lamports"""
        try:
            # Get fee rate
            fee_response = await self.client.get_fee_for_message(
                transaction.compile_message()
            )
            
            if fee_response.value is not None:
                return fee_response.value
                
            # Fallback estimation
            return 5000  # 0.000005 SOL
            
        except Exception as e:
            self.logger.debug(f"Fee estimation error: {str(e)}")
            return 5000  # Fallback fee
            
    async def get_transaction_details(self, signature: str) -> Optional[Dict]:
        """Get detailed transaction information"""
        try:
            response = await self.client.get_transaction(signature)
            
            if response.value:
                return {
                    'signature': signature,
                    'block_time': response.value.block_time,
                    'slot': response.value.slot,
                    'fee': response.value.meta.fee if response.value.meta else None,
                    'success': not bool(response.value.meta.err) if response.value.meta else False,
                    'log_messages': response.value.meta.log_messages if response.value.meta else []
                }
                
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get transaction details: {str(e)}")
            return None
            
    async def get_multiple_accounts(self, pubkeys: List[PublicKey]) -> List[Optional[Dict]]:
        """Get multiple account info in a single request"""
        try:
            response = await self.client.get_multiple_accounts(pubkeys)
            
            accounts = []
            if response.value:
                for account in response.value:
                    if account:
                        accounts.append({
                            'lamports': account.lamports,
                            'owner': str(account.owner),
                            'executable': account.executable,
                            'rent_epoch': account.rent_epoch
                        })
                    else:
                        accounts.append(None)
            else:
                accounts = [None] * len(pubkeys)
                
            return accounts
            
        except Exception as e:
            self.logger.error(f"Failed to get multiple accounts: {str(e)}")
            return [None] * len(pubkeys)
            
    async def simulate_transaction(self, transaction: Transaction) -> Dict:
        """Simulate transaction execution"""
        try:
            response = await self.client.simulate_transaction(transaction)
            
            if response.value:
                return {
                    'success': not bool(response.value.err),
                    'error': response.value.err,
                    'logs': response.value.logs or [],
                    'accounts': response.value.accounts,
                    'units_consumed': response.value.units_consumed
                }
            else:
                return {
                    'success': False,
                    'error': 'No simulation result',
                    'logs': [],
                    'accounts': None,
                    'units_consumed': None
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'logs': [],
                'accounts': None,
                'units_consumed': None
            }
            
    def sol_to_lamports(self, sol: float) -> int:
        """Convert SOL to lamports"""
        return int(sol * 1e9)
        
    def lamports_to_sol(self, lamports: int) -> float:
        """Convert lamports to SOL"""
        return lamports / 1e9
        
    async def get_slot(self) -> int:
        """Get current slot"""
        try:
            response = await self.client.get_slot()
            return response.value if response.value is not None else 0
        except Exception as e:
            self.logger.error(f"Failed to get slot: {str(e)}")
            return 0
            
    async def get_block_height(self) -> int:
        """Get current block height"""
        try:
            response = await self.client.get_block_height()
            return response.value if response.value is not None else 0
        except Exception as e:
            self.logger.error(f"Failed to get block height: {str(e)}")
            return 0
            
    async def get_epoch_info(self) -> Optional[Dict]:
        """Get current epoch information"""
        try:
            response = await self.client.get_epoch_info()
            
            if response.value:
                return {
                    'epoch': response.value.epoch,
                    'slot_index': response.value.slot_index,
                    'slots_in_epoch': response.value.slots_in_epoch,
                    'absolute_slot': response.value.absolute_slot,
                    'block_height': response.value.block_height,
                    'transaction_count': response.value.transaction_count
                }
                
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get epoch info: {str(e)}")
            return None
            
    def clear_cache(self):
        """Clear account info cache"""
        self.account_cache.clear()
        self.logger.debug("Account cache cleared")
        
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'cached_accounts': len(self.account_cache),
            'cache_ttl': self.cache_ttl
        }
        
    async def health_check(self) -> bool:
        """Check if RPC connection is healthy"""
        try:
            response = await self.client.get_health()
            return response == "ok"
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return False
            
    async def get_cluster_nodes(self) -> List[Dict]:
        """Get cluster node information"""
        try:
            response = await self.client.get_cluster_nodes()
            
            nodes = []
            if response.value:
                for node in response.value:
                    nodes.append({
                        'pubkey': node.pubkey,
                        'gossip': node.gossip,
                        'tpu': node.tpu,
                        'rpc': node.rpc,
                        'version': node.version
                    })
                    
            return nodes
            
        except Exception as e:
            self.logger.error(f"Failed to get cluster nodes: {str(e)}")
            return []
