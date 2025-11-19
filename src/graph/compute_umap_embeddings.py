"""
Compute UMAP embeddings for Elliptic Bitcoin transactions.

This script computes 2D UMAP coordinates globally across all timesteps,
allowing for consistent visualization and temporal comparison.

Usage:
    python src/graph/compute_umap_embeddings.py
"""

from pathlib import Path
import pandas as pd
import umap
import numpy as np
from tqdm import tqdm

# Paths
RAW_DATA_DIR = Path("raw_data/elliptic_bitcoin_dataset")
OUTPUT_DIR = Path("src/graph")

FEATURES_FILE = RAW_DATA_DIR / "elliptic_txs_features.csv"
CLASSES_FILE = RAW_DATA_DIR / "elliptic_txs_classes.csv"
OUTPUT_FILE = OUTPUT_DIR / "umap_coordinates.csv"


def load_data():
    """Load transaction features and classes."""
    print("Loading transaction features...")
    features_df = pd.read_csv(FEATURES_FILE, header=None)

    # Columns: txId (0), timestep (1), features (2-167)
    features_df.columns = ['txId', 'time_step'] + [f'feature_{i}' for i in range(1, 166)]

    print(f"Loaded {len(features_df)} transactions")
    print(f"Timesteps range: {features_df['time_step'].min()} to {features_df['time_step'].max()}")

    print("\nLoading transaction classes...")
    classes_df = pd.read_csv(CLASSES_FILE)

    print(f"Class distribution:")
    print(classes_df['class'].value_counts())

    # Merge features with classes
    merged_df = features_df.merge(classes_df, on='txId', how='left')

    return merged_df


def compute_umap_embeddings(df):
    """
    Compute UMAP embeddings globally across all timesteps.

    Args:
        df: DataFrame with transaction features and metadata

    Returns:
        DataFrame with added umap_x and umap_y columns
    """
    print("\n" + "="*60)
    print("Computing UMAP embeddings (this may take a few minutes)...")
    print("="*60)

    # Extract feature columns (columns 2-167 in original file)
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    X = df[feature_cols].values

    print(f"Feature matrix shape: {X.shape}")
    print(f"Computing 2D UMAP projection...")

    # Initialize UMAP with parameters suitable for graph visualization
    umap_reducer = umap.UMAP(
        n_components=2,           # 2D for visualization
        n_neighbors=15,           # Balance local and global structure
        min_dist=0.1,             # Minimum distance between points
        metric='euclidean',       # Distance metric
        random_state=42,          # Reproducibility
        verbose=True
    )

    # Fit and transform
    embeddings = umap_reducer.fit_transform(X)

    print(f"UMAP embeddings computed: {embeddings.shape}")
    print(f"X-coordinate range: [{embeddings[:, 0].min():.2f}, {embeddings[:, 0].max():.2f}]")
    print(f"Y-coordinate range: [{embeddings[:, 1].min():.2f}, {embeddings[:, 1].max():.2f}]")

    # Add UMAP coordinates to dataframe
    df['umap_x'] = embeddings[:, 0]
    df['umap_y'] = embeddings[:, 1]

    return df


def save_results(df):
    """Save UMAP coordinates and metadata to CSV."""
    print("\nSaving results...")

    # Select relevant columns
    output_df = df[['txId', 'time_step', 'umap_x', 'umap_y', 'class']]

    # Save to CSV
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✓ Saved to: {OUTPUT_FILE}")
    print(f"  Total transactions: {len(output_df)}")

    # Print summary statistics by class
    print("\nUMAP coordinate statistics by class:")
    for class_label in ['1', '2', 'unknown']:
        class_data = output_df[output_df['class'] == class_label]
        if len(class_data) > 0:
            print(f"\n{class_label} ({len(class_data)} transactions):")
            print(f"  X: mean={class_data['umap_x'].mean():.2f}, std={class_data['umap_x'].std():.2f}")
            print(f"  Y: mean={class_data['umap_y'].mean():.2f}, std={class_data['umap_y'].std():.2f}")


def main():
    """Main execution function."""
    print("="*60)
    print("UMAP Embedding Computation for Elliptic Dataset")
    print("="*60)

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data()

    # Compute UMAP embeddings globally
    df = compute_umap_embeddings(df)

    # Save results
    save_results(df)

    print("\n" + "="*60)
    print("✓ UMAP computation complete!")
    print("="*60)
    print(f"\nNext step: Run visualize_graph.py to generate visualizations")


if __name__ == "__main__":
    main()
