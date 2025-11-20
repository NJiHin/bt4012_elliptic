"""
Compute Address Node Features

Creates feature vectors for each address using address-specific activity metrics.

Usage:
    python src/graph/address_features.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# Paths
RAW_DATA_DIR = Path("raw_data/elliptic_bitcoin_dataset")
PROCESSED_DATA_DIR = Path("processed_data")

API_COMBINED_FILE = PROCESSED_DATA_DIR / "api_combined.csv"
FEATURES_FILE = RAW_DATA_DIR / "elliptic_txs_features.csv"
ADDRESS_EDGELIST_FILE = PROCESSED_DATA_DIR / "address_graphs" / "address_edgelist.csv"
ADDRESS_MAPPING_FILE = PROCESSED_DATA_DIR / "address_graphs" / "address_to_idx.csv"

OUTPUT_FILE = PROCESSED_DATA_DIR / "address_graphs" / "address_features.csv"


def load_data():
    """Load necessary datasets."""
    print("Loading datasets...")

    # Load API data
    print(f"  Loading {API_COMBINED_FILE}...")
    api_df = pd.read_csv(API_COMBINED_FILE)
    print(f"    {len(api_df)} transactions")

    # Load features
    print(f"  Loading {FEATURES_FILE}...")
    features_df = pd.read_csv(FEATURES_FILE, header=None)
    features_df.columns = ['txId', 'timestep'] + [f'feature_{i}' for i in range(1, 166)]
    print(f"    {len(features_df)} transactions with features")

    # Load address mapping
    print(f"  Loading {ADDRESS_MAPPING_FILE}...")
    address_mapping = pd.read_csv(ADDRESS_MAPPING_FILE)
    print(f"    {len(address_mapping)} addresses")

    # Load edgelist
    print(f"  Loading {ADDRESS_EDGELIST_FILE}...")
    edges_df = pd.read_csv(ADDRESS_EDGELIST_FILE)
    print(f"    {len(edges_df)} edges")

    # Merge API with features
    merged_df = api_df.merge(features_df, on='txId', how='inner')

    return merged_df, address_mapping, edges_df


def build_address_to_transactions(merged_df):
    """Map each address to all transactions it appears in."""
    print("\nBuilding address-to-transaction mapping...")

    address_txs = defaultdict(list)

    # STRATEGY 2A: Use itertuples instead of iterrows (2-5x faster)
    for row in tqdm(merged_df.itertuples(index=False), total=len(merged_df), desc="Processing"):
        if not row.success:
            continue

        txId = row.txId

        # Input addresses
        if pd.notna(row.input_addresses) and row.input_addresses:
            input_addrs = [addr.strip() for addr in str(row.input_addresses).split('|') if addr.strip()]
            for addr in input_addrs:
                address_txs[addr].append({
                    'txId': txId,
                    'role': 'input',
                    'timestep': row.timestep,
                    'total_input_btc': row.total_input_btc,
                    'total_output_btc': row.total_output_btc,
                    'num_inputs': row.num_inputs,
                    'num_outputs': row.num_outputs,
                    'fee_btc': row.fee_btc
                })

        # Output addresses
        if pd.notna(row.output_addresses) and row.output_addresses:
            output_addrs = [addr.strip() for addr in str(row.output_addresses).split('|') if addr.strip()]
            for addr in output_addrs:
                address_txs[addr].append({
                    'txId': txId,
                    'role': 'output',
                    'timestep': row.timestep,
                    'total_input_btc': row.total_input_btc,
                    'total_output_btc': row.total_output_btc,
                    'num_inputs': row.num_inputs,
                    'num_outputs': row.num_outputs,
                    'fee_btc': row.fee_btc
                })

    print(f"  Mapped {len(address_txs)} addresses to transactions")
    return address_txs


def compute_features_for_address(addr, tx_list, source_groups, target_groups):
    """
    Compute feature vector for a single address.

    Features include:
    - Activity metrics (num transactions, BTC sent/received, etc.)
    - Temporal features (first/last timestep, activity duration)
    - Graph features (degree, unique counterparties)
    - Transaction structure features (input/output ratios, fees, etc.)

    Args:
        addr: Address to compute features for
        tx_list: List of transaction dicts for this address
        source_groups: Pre-indexed dict {source_addr: [target_addrs]}
        target_groups: Pre-indexed dict {target_addr: [source_addrs]}
    """
    features = {}

    # Separate input and output transactions
    input_txs = [tx for tx in tx_list if tx['role'] == 'input']
    output_txs = [tx for tx in tx_list if tx['role'] == 'output']

    # Activity metrics
    features['num_transactions'] = len(tx_list)
    features['num_as_input'] = len(input_txs)
    features['num_as_output'] = len(output_txs)

    # BTC amounts
    total_sent = 0
    total_received = 0
    for tx_info in tx_list:
        if tx_info['role'] == 'input':
            total_sent += tx_info['total_input_btc'] if pd.notna(tx_info['total_input_btc']) else 0
        else:
            total_received += tx_info['total_output_btc'] if pd.notna(tx_info['total_output_btc']) else 0

    features['total_btc_sent'] = total_sent
    features['total_btc_received'] = total_received
    features['net_btc_flow'] = total_received - total_sent

    # Average BTC per transaction
    features['avg_btc_per_input_tx'] = total_sent / features['num_as_input'] if features['num_as_input'] > 0 else 0
    features['avg_btc_per_output_tx'] = total_received / features['num_as_output'] if features['num_as_output'] > 0 else 0

    # Transaction structure features (from input transactions)
    if len(input_txs) > 0:
        total_num_inputs = sum(tx['num_inputs'] for tx in input_txs if pd.notna(tx['num_inputs']))
        total_num_outputs = sum(tx['num_outputs'] for tx in input_txs if pd.notna(tx['num_outputs']))
        total_fees = sum(tx['fee_btc'] for tx in input_txs if pd.notna(tx['fee_btc']))

        features['avg_num_inputs_per_tx'] = total_num_inputs / len(input_txs)
        features['avg_num_outputs_per_tx'] = total_num_outputs / len(input_txs)
        features['input_output_ratio'] = total_num_inputs / total_num_outputs if total_num_outputs > 0 else 0
        features['total_fees_paid'] = total_fees
        features['fee_to_volume_ratio'] = total_fees / total_sent if total_sent > 0 else 0
    else:
        # Address only received, never sent
        features['avg_num_inputs_per_tx'] = 0
        features['avg_num_outputs_per_tx'] = 0
        features['input_output_ratio'] = 0
        features['total_fees_paid'] = 0
        features['fee_to_volume_ratio'] = 0

    # Temporal features
    timesteps = [tx_info['timestep'] for tx_info in tx_list]
    features['first_timestep'] = min(timesteps)
    features['last_timestep'] = max(timesteps)
    features['activity_duration'] = max(timesteps) - min(timesteps) + 1

    # STRATEGY 1: Graph features using pre-indexed lookups (10-100x faster)
    outgoing = source_groups.get(addr, [])
    incoming = target_groups.get(addr, [])

    features['out_degree'] = len(outgoing)
    features['in_degree'] = len(incoming)
    features['unique_counterparties'] = len(set(outgoing) | set(incoming))

    return features


def compute_all_features(address_mapping, address_txs, edges_df):
    """Compute features for all addresses."""
    print("\nComputing features for all addresses...")

    # STRATEGY 1: Pre-index edges once (10-100x speedup for graph features)
    print("  Building edge indexes...")
    source_groups = edges_df.groupby('source')['target'].apply(list).to_dict()
    target_groups = edges_df.groupby('target')['source'].apply(list).to_dict()
    print(f"    Indexed {len(source_groups)} source addresses")
    print(f"    Indexed {len(target_groups)} target addresses")

    all_features = []

    # STRATEGY 2A: Use itertuples instead of iterrows
    for row in tqdm(address_mapping.itertuples(index=False), total=len(address_mapping), desc="Computing features"):
        addr = row.address
        idx = row.idx

        if addr not in address_txs:
            # Address has no transaction data (shouldn't happen, but handle it)
            continue

        tx_list = address_txs[addr]
        features = compute_features_for_address(addr, tx_list, source_groups, target_groups)

        # Flatten features
        feature_vector = {
            'address': addr,
            'idx': idx
        }

        # Add activity metrics
        feature_vector['num_transactions'] = features['num_transactions']
        feature_vector['num_as_input'] = features['num_as_input']
        feature_vector['num_as_output'] = features['num_as_output']
        feature_vector['total_btc_sent'] = features['total_btc_sent']
        feature_vector['total_btc_received'] = features['total_btc_received']
        feature_vector['net_btc_flow'] = features['net_btc_flow']
        feature_vector['avg_btc_per_input_tx'] = features['avg_btc_per_input_tx']
        feature_vector['avg_btc_per_output_tx'] = features['avg_btc_per_output_tx']

        # Add transaction structure features
        feature_vector['avg_num_inputs_per_tx'] = features['avg_num_inputs_per_tx']
        feature_vector['avg_num_outputs_per_tx'] = features['avg_num_outputs_per_tx']
        feature_vector['input_output_ratio'] = features['input_output_ratio']
        feature_vector['total_fees_paid'] = features['total_fees_paid']
        feature_vector['fee_to_volume_ratio'] = features['fee_to_volume_ratio']

        # Add temporal features
        feature_vector['first_timestep'] = features['first_timestep']
        feature_vector['last_timestep'] = features['last_timestep']
        feature_vector['activity_duration'] = features['activity_duration']

        # Add graph features
        feature_vector['out_degree'] = features['out_degree']
        feature_vector['in_degree'] = features['in_degree']
        feature_vector['unique_counterparties'] = features['unique_counterparties']

        all_features.append(feature_vector)

    features_df = pd.DataFrame(all_features)
    return features_df


def main():
    """Main execution function."""
    print("="*60)
    print("ADDRESS FEATURE COMPUTATION")
    print("="*60)

    # Load data
    merged_df, address_mapping, edges_df = load_data()

    # Build address-to-transaction mapping
    address_txs = build_address_to_transactions(merged_df)

    # Compute features
    features_df = compute_all_features(address_mapping, address_txs, edges_df)

    # Save features
    print(f"\nSaving features to {OUTPUT_FILE}...")
    features_df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Addresses processed: {len(features_df)}")
    print(f"Feature dimensions:  {len(features_df.columns) - 2}")  # Exclude address and idx
    print(f"\nFeature categories:")
    print(f"  - Activity metrics: 8 features")
    print(f"  - Transaction structure: 5 features")
    print(f"  - Temporal: 3 features")
    print(f"  - Graph: 3 features")
    print(f"  Total: 19 features")
    print(f"\nOutput file:         {OUTPUT_FILE}")
    print("="*60)


if __name__ == "__main__":
    main()
