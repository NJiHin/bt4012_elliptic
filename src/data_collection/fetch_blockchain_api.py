"""
Fetch Bitcoin Transaction Data from Blockchain.com API

Fetches transaction details (sender/receiver addresses, values, timestamps)
from the blockchain for the Elliptic dataset.

Features:
- Rate limiting (configurable)
- Automatic checkpointing (resume from failures)
- Progress tracking
- Error handling and retry logic

Usage:
    python src/data_collection/fetch_blockchain_api.py

Input:  raw_data/Result.csv (automatically detected)
Output: processed_data/api_outputs.csv (created if doesn't exist)
"""

# ============================================================================
# CONFIGURATION - Edit these values as needed
# ============================================================================

# Input/Output files
INPUT_FILE = 'raw_data/timestep_37_48_ivan.csv'          # CHANGE THIS TO YOUR FILE CSV with txId and transaction columns
OUTPUT_FILE = 'processed_data/api_outputs.csv'  # Where to save results

# API Settings
RATE_LIMIT_PER_MINUTE = 50                 # Max requests per minute (default: 100)
CHECKPOINT_INTERVAL = 50                   # Save progress every N transactions

# ============================================================================

import requests
import time
import pandas as pd
import argparse
import json
from datetime import datetime
from tqdm import tqdm
from pathlib import Path


class BlockchainAPIFetcher:
    """Fetches Bitcoin transaction data from Blockchain.com API."""

    def __init__(self, rate_limit_per_minute=100):
        """
        Initialize fetcher with rate limiting.

        Args:
            rate_limit_per_minute: Maximum requests per minute (default: 50 for safety)
        """
        self.base_url = "https://blockchain.info/rawtx/"
        self.rate_limit = rate_limit_per_minute
        self.request_times = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Research Project - Bitcoin Fraud Detection)'
        })

    def _wait_for_rate_limit(self):
        """Enforce rate limiting by waiting if necessary."""
        now = time.time()

        # Remove requests older than 60 seconds
        self.request_times = [t for t in self.request_times if now - t < 60]

        # If at limit, wait
        if len(self.request_times) >= self.rate_limit:
            sleep_time = 60 - (now - self.request_times[0]) + 0.5  # Small buffer
            if sleep_time > 0:
                time.sleep(sleep_time)
                # Clean old requests again
                now = time.time()
                self.request_times = [t for t in self.request_times if now - t < 60]

        # Record this request
        self.request_times.append(time.time())

    def fetch_transaction(self, tx_hash, retry_count=3):
        """
        Fetch transaction details from Blockchain.com API.

        Args:
            tx_hash: Bitcoin transaction hash
            retry_count: Number of retries on failure

        Returns:
            dict: Transaction data including inputs, outputs, timestamp, value
        """
        for attempt in range(retry_count):
            try:
                # Rate limiting
                self._wait_for_rate_limit()

                # Make request
                url = f"{self.base_url}{tx_hash}"
                response = self.session.get(url, timeout=10)

                # Handle rate limiting
                if response.status_code == 429:
                    wait_time = 60 * (2 ** attempt)  # Exponential backoff
                    print(f"\nRate limited for {tx_hash}. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                # Handle other errors
                response.raise_for_status()

                # Parse response
                data = response.json()

                # Extract input addresses
                input_addresses = []
                input_values = []
                for inp in data.get('inputs', []):
                    prev_out = inp.get('prev_out', {})
                    addr = prev_out.get('addr')
                    value = prev_out.get('value', 0)  # in satoshis
                    if addr:
                        input_addresses.append(addr)
                        input_values.append(value)

                # Extract output addresses
                output_addresses = []
                output_values = []
                for out in data.get('out', []):
                    addr = out.get('addr')
                    value = out.get('value', 0)  # in satoshis
                    if addr:
                        output_addresses.append(addr)
                        output_values.append(value)

                # Calculate total values (convert satoshis to BTC)
                total_input = sum(input_values) / 1e8
                total_output = sum(output_values) / 1e8
                fee = total_input - total_output

                return {
                    'transaction': tx_hash,
                    'success': True,
                    'timestamp': data.get('time'),
                    'block_height': data.get('block_height'),
                    'num_inputs': len(input_addresses),
                    'num_outputs': len(output_addresses),
                    'input_addresses': '|'.join(input_addresses),  # Pipe-separated
                    'output_addresses': '|'.join(output_addresses),
                    'input_values': '|'.join(map(str, input_values)),
                    'output_values': '|'.join(map(str, output_values)),
                    'total_input_btc': total_input,
                    'total_output_btc': total_output,
                    'fee_btc': fee,
                    'fetched_at': datetime.now().isoformat(),
                    'error': None
                }

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    # Transaction not found
                    return {
                        'transaction': tx_hash,
                        'success': False,
                        'error': 'Transaction not found (404)',
                        'fetched_at': datetime.now().isoformat()
                    }
                elif attempt < retry_count - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                else:
                    return {
                        'transaction': tx_hash,
                        'success': False,
                        'error': f'HTTP Error: {e}',
                        'fetched_at': datetime.now().isoformat()
                    }

            except Exception as e:
                if attempt < retry_count - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                else:
                    return {
                        'transaction': tx_hash,
                        'success': False,
                        'error': f'Error: {str(e)}',
                        'fetched_at': datetime.now().isoformat()
                    }

        return {
            'transaction': tx_hash,
            'success': False,
            'error': 'Max retries exceeded',
            'fetched_at': datetime.now().isoformat()
        }


def load_transaction_hashes(input_file):
    """
    Load transaction hashes from CSV file.

    Expects CSV with 'txId' and 'transaction' columns (Result.csv format).

    Args:
        input_file: Path to CSV file

    Returns:
        DataFrame: Transaction hashes with txId mapping
    """
    df = pd.read_csv(input_file)

    if 'txId' not in df.columns or 'transaction' not in df.columns:
        raise ValueError("CSV must have 'txId' and 'transaction' columns")

    print(f"Loaded {len(df)} transactions with txId mapping")
    return df[['txId', 'transaction']].copy()


def process_transactions(tx_df, output_file, checkpoint_interval=100, rate_limit=100):
    """
    Process all transactions with checkpointing.

    Args:
        tx_df: DataFrame with 'txId' and 'transaction' columns
        output_file: Path to output CSV file
        checkpoint_interval: Save progress every N transactions
        rate_limit: Requests per minute
    """
    output_path = Path(output_file)
    checkpoint_path = output_path.with_suffix('.checkpoint.csv')

    # Load checkpoint if exists
    processed_hashes = set()
    results = []

    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        df_checkpoint = pd.read_csv(checkpoint_path)
        processed_hashes = set(df_checkpoint['transaction'].astype(str))
        results = df_checkpoint.to_dict('records')
        print(f"Resuming: {len(processed_hashes)} already processed")

    # Filter out already processed
    remaining_df = tx_df[~tx_df['transaction'].isin(processed_hashes)].copy()
    print(f"Remaining to process: {len(remaining_df)}")

    if len(remaining_df) == 0:
        print("All transactions already processed!")
        return

    # Initialize fetcher
    fetcher = BlockchainAPIFetcher(rate_limit_per_minute=rate_limit)

    # Process transactions
    print(f"\nFetching transaction data from Blockchain.com API...")
    print(f"Rate limit: {fetcher.rate_limit} requests/minute")
    print(f"Estimated time: {len(remaining_df) / fetcher.rate_limit:.1f} minutes\n")

    start_time = time.time()

    with tqdm(total=len(remaining_df), desc="Fetching transactions") as pbar:
        for i, row in remaining_df.iterrows():
            tx_hash = row['transaction']
            result = fetcher.fetch_transaction(tx_hash)

            # Add txId to result
            result['txId'] = row['txId']

            results.append(result)
            pbar.update(1)

            # Update progress bar with success rate
            success_count = sum(1 for r in results if r.get('success'))
            success_rate = success_count / len(results) * 100 if results else 0
            pbar.set_postfix({
                'Success': f'{success_rate:.1f}%',
                'Total': len(results)
            })

            # Checkpoint every N transactions
            if len(results) % checkpoint_interval == 0:
                df_temp = pd.DataFrame(results)
                cols = ['txId', 'transaction'] + [c for c in df_temp.columns if c not in ['txId', 'transaction']]
                df_temp = df_temp[cols]
                df_temp.to_csv(checkpoint_path, index=False)

    # Final save
    df_final = pd.DataFrame(results)
    cols = ['txId', 'transaction'] + [c for c in df_final.columns if c not in ['txId', 'transaction']]
    df_final = df_final[cols]
    df_final.to_csv(output_path, index=False)

    # Remove checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # Statistics
    elapsed_time = time.time() - start_time
    successful = df_final[df_final['success'] == True]
    failed = df_final[df_final['success'] == False]

    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Total processed: {len(df_final)}")
    print(f"Successful: {len(successful)} ({len(successful)/len(df_final)*100:.1f}%)")
    print(f"Failed: {len(failed)} ({len(failed)/len(df_final)*100:.1f}%)")
    print(f"Time elapsed: {elapsed_time/60:.1f} minutes")
    print(f"Average rate: {len(df_final)/elapsed_time*60:.1f} requests/minute")
    print(f"\nOutput saved to: {output_path}")

    # Show common errors
    if len(failed) > 0:
        print(f"\nMost common errors:")
        error_counts = failed['error'].value_counts().head(5)
        for error, count in error_counts.items():
            print(f"  {error}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description='Fetch Bitcoin transaction data from Blockchain.com API for Elliptic dataset'
    )
    parser.add_argument(
        '--rate-limit',
        type=int,
        default=RATE_LIMIT_PER_MINUTE,
        help=f'Requests per minute (default: {RATE_LIMIT_PER_MINUTE})'
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=CHECKPOINT_INTERVAL,
        help=f'Save checkpoint every N transactions (default: {CHECKPOINT_INTERVAL})'
    )

    args = parser.parse_args()

    # Use configuration constants (can be overridden by command line args)
    input_file = INPUT_FILE
    output_file = OUTPUT_FILE

    # Create output directory if doesn't exist
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load transaction hashes
    print(f"Loading transaction hashes from {input_file}")
    tx_df = load_transaction_hashes(input_file)
    print(f"Loaded {len(tx_df)} transaction hashes")

    # Process
    process_transactions(
        tx_df,
        output_file,
        checkpoint_interval=args.checkpoint_interval,
        rate_limit=args.rate_limit
    )


if __name__ == "__main__":
    main()
