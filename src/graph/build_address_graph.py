"""
Build Address-Based Graph from Transaction Data

Transforms the transaction-based Elliptic dataset into an address-based graph:
- Nodes: Bitcoin addresses (extracted from API data)
- Edges: Address-to-address flows (via transactions)
- Features: Aggregated transaction features per address
- Labels: Propagated from transaction labels

Usage:
    python src/graph/build_address_graph.py
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
CLASSES_FILE = RAW_DATA_DIR / "elliptic_txs_classes.csv"

OUTPUT_DIR = PROCESSED_DATA_DIR / "address_graphs"
ADDRESS_FEATURES_FILE = OUTPUT_DIR / "address_features.csv"
ADDRESS_LABELS_FILE = OUTPUT_DIR / "address_labels.csv"
ADDRESS_EDGELIST_FILE = OUTPUT_DIR / "address_edgelist.csv"
ADDRESS_MAPPING_FILE = OUTPUT_DIR / "address_to_idx.csv"
TIMESTEP_STATS_FILE = OUTPUT_DIR / "timestep_stats.csv"


def load_data():
    """Load all necessary datasets."""
    print("Loading datasets...")

    # Load API data with address information
    print(f"  Loading {API_COMBINED_FILE}...")
    api_df = pd.read_csv(API_COMBINED_FILE)
    print(f"    {len(api_df)} transactions with API data")

    # Load features (no header, columns: txId, timestep, feature_1...feature_165)
    print(f"  Loading {FEATURES_FILE}...")
    features_df = pd.read_csv(FEATURES_FILE, header=None)
    features_df.columns = ['txId', 'timestep'] + [f'feature_{i}' for i in range(1, 166)]
    print(f"    {len(features_df)} transactions with features")

    # Load classes
    print(f"  Loading {CLASSES_FILE}...")
    classes_df = pd.read_csv(CLASSES_FILE)
    print(f"    {len(classes_df)} transactions with labels")

    # Merge datasets
    print("\n  Merging datasets...")
    merged_df = api_df.merge(features_df[['txId', 'timestep']], on='txId', how='inner')
    merged_df = merged_df.merge(features_df, on='txId', how='inner', suffixes=('', '_dup'))
    merged_df = merged_df.merge(classes_df, on='txId', how='left')

    # Drop duplicate timestep column
    if 'timestep_dup' in merged_df.columns:
        merged_df = merged_df.drop(columns=['timestep_dup'])

    print(f"    {len(merged_df)} transactions in merged dataset")
    print(f"    Timestep range: {merged_df['timestep'].min()} to {merged_df['timestep'].max()}")

    return merged_df, features_df


def extract_addresses(merged_df):
    """
    Extract all unique addresses and their transaction associations.

    Returns:
        address_txs: dict mapping address -> list of (txId, role, timestep, btc_amount)
                     role is 'input' or 'output'
    """
    print("\nExtracting addresses from transactions...")
    address_txs = defaultdict(list)

    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Processing transactions"):
        if not row['success']:  # Skip failed transactions
            continue

        txId = row['txId']
        timestep = row['timestep']

        # Parse input addresses
        if pd.notna(row['input_addresses']) and row['input_addresses']:
            input_addrs = str(row['input_addresses']).split('|')
            input_vals = str(row['input_values']).split('|') if pd.notna(row['input_values']) else []

            for i, addr in enumerate(input_addrs):
                addr = addr.strip()
                if addr:
                    btc_amount = float(input_vals[i]) / 1e8 if i < len(input_vals) else 0.0  # Convert satoshi to BTC
                    address_txs[addr].append({
                        'txId': txId,
                        'role': 'input',
                        'timestep': timestep,
                        'btc_amount': btc_amount
                    })

        # Parse output addresses
        if pd.notna(row['output_addresses']) and row['output_addresses']:
            output_addrs = str(row['output_addresses']).split('|')
            output_vals = str(row['output_values']).split('|') if pd.notna(row['output_values']) else []

            for i, addr in enumerate(output_addrs):
                addr = addr.strip()
                if addr:
                    btc_amount = float(output_vals[i]) / 1e8 if i < len(output_vals) else 0.0
                    address_txs[addr].append({
                        'txId': txId,
                        'role': 'output',
                        'timestep': timestep,
                        'btc_amount': btc_amount
                    })

    print(f"  Extracted {len(address_txs)} unique addresses")
    print(f"  Average transactions per address: {np.mean([len(txs) for txs in address_txs.values()]):.2f}")

    return address_txs


def build_edges(merged_df):
    """
    Build address-to-address edges from transactions.
    Each transaction creates edges: input_address -> output_address(es)

    Returns:
        edges: list of dicts with source, target, txId, timestep, btc_amount
    """
    print("\nBuilding address-to-address edges...")
    edges = []

    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Creating edges"):
        if not row['success']:
            continue

        txId = row['txId']
        timestep = row['timestep']

        # Parse addresses
        input_addrs = []
        if pd.notna(row['input_addresses']) and row['input_addresses']:
            input_addrs = [addr.strip() for addr in str(row['input_addresses']).split('|') if addr.strip()]

        output_addrs = []
        output_vals = []
        if pd.notna(row['output_addresses']) and row['output_addresses']:
            output_addrs = [addr.strip() for addr in str(row['output_addresses']).split('|') if addr.strip()]
            if pd.notna(row['output_values']) and row['output_values']:
                output_vals = [float(v) / 1e8 for v in str(row['output_values']).split('|')]

        # Create edges: each input -> each output
        for input_addr in input_addrs:
            for i, output_addr in enumerate(output_addrs):
                btc_amount = output_vals[i] if i < len(output_vals) else 0.0
                edges.append({
                    'source': input_addr,
                    'target': output_addr,
                    'txId': txId,
                    'timestep': timestep,
                    'btc_amount': btc_amount
                })

    print(f"  Created {len(edges)} address-to-address edges")
    return edges


def save_address_mapping(address_txs):
    """Save address to index mapping."""
    print("\nSaving address mapping...")
    addresses = sorted(address_txs.keys())
    mapping_df = pd.DataFrame({
        'address': addresses,
        'idx': range(len(addresses))
    })
    mapping_df.to_csv(ADDRESS_MAPPING_FILE, index=False)
    print(f"  Saved {len(addresses)} addresses to {ADDRESS_MAPPING_FILE}")


def compute_timestep_stats(address_txs, edges):
    """Compute statistics per timestep."""
    print("\nComputing timestep statistics...")

    stats = []
    timesteps = set()
    for addr_data in address_txs.values():
        for tx_info in addr_data:
            timesteps.add(tx_info['timestep'])

    for t in sorted(timesteps):
        # Count addresses active in this timestep
        active_addresses = set()
        for addr, tx_list in address_txs.items():
            if any(tx['timestep'] == t for tx in tx_list):
                active_addresses.add(addr)

        # Count edges in this timestep
        t_edges = [e for e in edges if e['timestep'] == t]

        stats.append({
            'timestep': t,
            'num_addresses': len(active_addresses),
            'num_edges': len(t_edges),
            'total_btc': sum(e['btc_amount'] for e in t_edges)
        })

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(TIMESTEP_STATS_FILE, index=False)
    print(f"  Saved timestep statistics to {TIMESTEP_STATS_FILE}")
    return stats_df


def main():
    """Main execution function."""
    print("="*60)
    print("ADDRESS-BASED GRAPH CONSTRUCTION")
    print("="*60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    merged_df, features_df = load_data()

    # Extract addresses and their transaction associations
    address_txs = extract_addresses(merged_df)

    # Build edges
    edges = build_edges(merged_df)

    # Save address mapping
    save_address_mapping(address_txs)

    # Save edges
    print(f"\nSaving edges to {ADDRESS_EDGELIST_FILE}...")
    edges_df = pd.DataFrame(edges)
    edges_df.to_csv(ADDRESS_EDGELIST_FILE, index=False)
    print(f"  Saved {len(edges_df)} edges")

    # Compute and save timestep statistics
    stats_df = compute_timestep_stats(address_txs, edges)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total unique addresses:     {len(address_txs)}")
    print(f"Total edges:                {len(edges)}")
    print(f"Timesteps:                  {stats_df['timestep'].min()} to {stats_df['timestep'].max()}")
    print(f"Avg addresses per timestep: {stats_df['num_addresses'].mean():.0f}")
    print(f"Avg edges per timestep:     {stats_df['num_edges'].mean():.0f}")
    print("\n" + "="*60)
    print("Next steps:")
    print("  1. Run address_features.py to compute node features")
    print("  2. Run address_labels.py to propagate labels")
    print("="*60)


if __name__ == "__main__":
    main()
