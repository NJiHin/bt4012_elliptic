"""
Propagate Labels from Transactions to Addresses

Assigns labels to addresses based on ONLY the transactions they INITIATED (input role).
Per Elliptic dataset paper: "A transaction is deemed licit/illicit if the entity
controlling the input addresses belongs to a licit/illicit category."

Uses majority voting: The most common label among input transactions wins.

Usage:
    python src/graph/address_labels.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter

# Paths
RAW_DATA_DIR = Path("raw_data/elliptic_bitcoin_dataset")
PROCESSED_DATA_DIR = Path("processed_data")

API_COMBINED_FILE = PROCESSED_DATA_DIR / "api_combined.csv"
FEATURES_FILE = RAW_DATA_DIR / "elliptic_txs_features.csv"
CLASSES_FILE = RAW_DATA_DIR / "elliptic_txs_classes.csv"
ADDRESS_MAPPING_FILE = PROCESSED_DATA_DIR / "address_graphs" / "address_to_idx.csv"

OUTPUT_FILE = PROCESSED_DATA_DIR / "address_graphs" / "address_labels.csv"


def load_data():
    """Load necessary datasets."""
    print("Loading datasets...")

    # Load API data
    print(f"  Loading {API_COMBINED_FILE}...")
    api_df = pd.read_csv(API_COMBINED_FILE, low_memory=False)
    print(f"    {len(api_df)} transactions")

    # Load features (for timestep info)
    print(f"  Loading {FEATURES_FILE}...")
    features_df = pd.read_csv(FEATURES_FILE, header=None)
    features_df.columns = ['txId', 'timestep'] + [f'feature_{i}' for i in range(1, 166)]

    # Load classes
    print(f"  Loading {CLASSES_FILE}...")
    classes_df = pd.read_csv(CLASSES_FILE)
    print(f"    {len(classes_df)} transactions with labels")

    # Load address mapping
    print(f"  Loading {ADDRESS_MAPPING_FILE}...")
    address_mapping = pd.read_csv(ADDRESS_MAPPING_FILE)
    print(f"    {len(address_mapping)} addresses")

    # Merge datasets
    merged_df = api_df.merge(features_df[['txId', 'timestep']], on='txId', how='inner')
    merged_df = merged_df.merge(classes_df, on='txId', how='left')

    return merged_df, address_mapping


def build_address_labels_mapping(merged_df):
    """
    Map each address to transaction labels.

    IMPORTANT: Only INPUT addresses inherit the transaction label, as they
    are the entities that initiated/controlled the transaction. Output addresses
    are tracked separately for statistics but do not inherit labels.
    """
    print("\nBuilding address-to-labels mapping...")

    address_labels = defaultdict(list)

    # Rename 'class' column to avoid Python keyword conflict with itertuples
    merged_df = merged_df.rename(columns={'class': 'tx_class'})

    for row in tqdm(merged_df.itertuples(index=False), total=len(merged_df), desc="Processing"):
        if not row.success:
            continue

        txId = row.txId
        label = row.tx_class if pd.notna(row.tx_class) else 'unknown'

        # INPUT addresses - These INITIATED the transaction, inherit the label
        if pd.notna(row.input_addresses) and row.input_addresses:
            input_addrs = [addr.strip() for addr in str(row.input_addresses).split('|') if addr.strip()]
            for addr in input_addrs:
                address_labels[addr].append({
                    'txId': txId,
                    'label': label,
                    'role': 'input'  # Entity controlling this address initiated this tx
                })

        # OUTPUT addresses - These RECEIVED funds, tracked for stats only
        if pd.notna(row.output_addresses) and row.output_addresses:
            output_addrs = [addr.strip() for addr in str(row.output_addresses).split('|') if addr.strip()]
            for addr in output_addrs:
                address_labels[addr].append({
                    'txId': txId,
                    'label': label,
                    'role': 'output'  # Only received funds, does NOT inherit label
                })

    print(f"  Mapped {len(address_labels)} addresses to labels")
    return address_labels


def compute_label_majority(label_list):
    """
    Majority voting strategy.
    Returns the most common label among INPUT transactions only.
    """
    # ONLY consider transactions where address was INPUT (initiated transaction)
    input_labels = [item['label'] for item in label_list if item['role'] == 'input']

    if not input_labels:
        return 'unknown', 0.0  # Address never initiated any transaction

    counter = Counter(input_labels)

    # If all unknown, return unknown
    if counter.most_common(1)[0][0] == 'unknown' and len(counter) == 1:
        return 'unknown', 0.0

    # Remove 'unknown' from consideration
    filtered = [l for l in input_labels if l != 'unknown']
    if not filtered:
        return 'unknown', 0.0

    counter_filtered = Counter(filtered)
    most_common_label = counter_filtered.most_common(1)[0][0]
    confidence = counter_filtered[most_common_label] / len(filtered)

    return most_common_label, confidence


def compute_labels(address_mapping, address_labels):
    """Compute labels for all addresses using majority voting."""
    print("\nComputing labels using majority voting...")
    print("  Note: Only INPUT transactions are used for labeling (addresses that initiated txs)")

    results = []

    for row in tqdm(address_mapping.itertuples(index=False), total=len(address_mapping), desc="Labeling"):
        addr = row.address
        idx = row.idx

        if addr not in address_labels:
            # No label data (shouldn't happen)
            label = 'unknown'
            confidence = 0.0
            num_transactions = 0
            num_input_txs = 0
            num_output_txs = 0
            num_illicit_input_txs = 0
            num_licit_input_txs = 0
            num_illicit_output_txs = 0
            num_licit_output_txs = 0
        else:
            label_list = address_labels[addr]
            label, confidence = compute_label_majority(label_list)

            # Statistics: separate input vs output
            num_transactions = len(label_list)
            num_input_txs = sum(1 for item in label_list if item['role'] == 'input')
            num_output_txs = sum(1 for item in label_list if item['role'] == 'output')

            num_illicit_input_txs = sum(1 for item in label_list if item['role'] == 'input' and item['label'] == '1')
            num_licit_input_txs = sum(1 for item in label_list if item['role'] == 'input' and item['label'] == '2')
            num_illicit_output_txs = sum(1 for item in label_list if item['role'] == 'output' and item['label'] == '1')
            num_licit_output_txs = sum(1 for item in label_list if item['role'] == 'output' and item['label'] == '2')

        results.append({
            'address': addr,
            'idx': idx,
            'label': label,
            'confidence': confidence,
            'num_transactions': num_transactions,
            'num_input_txs': num_input_txs,
            'num_output_txs': num_output_txs,
            'num_illicit_input_txs': num_illicit_input_txs,
            'num_licit_input_txs': num_licit_input_txs,
            'num_illicit_output_txs': num_illicit_output_txs,
            'num_licit_output_txs': num_licit_output_txs
        })

    labels_df = pd.DataFrame(results)
    return labels_df


def main():
    """Main execution function."""
    print("="*60)
    print("ADDRESS LABEL PROPAGATION (Majority Voting)")
    print("="*60)

    # Load data
    merged_df, address_mapping = load_data()

    # Build address-to-labels mapping
    address_labels = build_address_labels_mapping(merged_df)

    # Compute labels
    labels_df = compute_labels(address_mapping, address_labels)

    # Save labels
    print(f"\nSaving labels to {OUTPUT_FILE}...")
    labels_df.to_csv(OUTPUT_FILE, index=False)

    # Print statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total addresses: {len(labels_df)}")
    print(f"\nLabel distribution (based on INPUT transactions only):")
    print(labels_df['label'].value_counts())
    print(f"\nAverage confidence: {labels_df['confidence'].mean():.4f}")

    print(f"\nAddresses that INITIATED illicit transactions:  {(labels_df['num_illicit_input_txs'] > 0).sum()}")
    print(f"Addresses that INITIATED licit transactions:    {(labels_df['num_licit_input_txs'] > 0).sum()}")
    print(f"\nAddresses that RECEIVED from illicit transactions: {(labels_df['num_illicit_output_txs'] > 0).sum()}")
    print(f"Addresses that RECEIVED from licit transactions:   {(labels_df['num_licit_output_txs'] > 0).sum()}")

    print(f"\nAddresses that only received (never sent): {(labels_df['num_input_txs'] == 0).sum()}")

    print(f"\nOutput file: {OUTPUT_FILE}")
    print("="*60)


if __name__ == "__main__":
    main()
