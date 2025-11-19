"""
Analyze timestep distribution and class percentages in Elliptic Bitcoin Dataset
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'raw_data' / 'elliptic_bitcoin_dataset'
OUTPUT_DIR = BASE_DIR / 'src' / 'eda' / 'plots'

# Load data
print("Loading data...")
features_df = pd.read_csv(DATA_DIR / 'elliptic_txs_features.csv', header=None)
classes_df = pd.read_csv(DATA_DIR / 'elliptic_txs_classes.csv')

# Name the columns for features (txId and timestep)
features_df.columns = ['txId', 'timestep'] + [f'feature_{i}' for i in range(1, 166)]

# Merge features with classes
df = features_df[['txId', 'timestep']].merge(classes_df, on='txId', how='left')

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Analyze timestep distribution
print("\n=== Timestep Analysis ===")
timestep_stats = df.groupby('timestep').agg({
    'txId': 'count',
    'class': lambda x: (x == '1').sum(),  # illicit
}).rename(columns={'txId': 'total_count', 'class': 'illicit_count'})

# Add class counts
timestep_stats['licit_count'] = df[df['class'] == '2'].groupby('timestep').size()
timestep_stats['unknown_count'] = df[df['class'] == 'unknown'].groupby('timestep').size()

# Fill NaN with 0 (timesteps with no illicit/unknown transactions)
timestep_stats = timestep_stats.fillna(0).astype(int)

# Calculate percentages
timestep_stats['licit_pct'] = (timestep_stats['licit_count'] / timestep_stats['total_count'] * 100).round(2)
timestep_stats['illicit_pct'] = (timestep_stats['illicit_count'] / timestep_stats['total_count'] * 100).round(2)
timestep_stats['unknown_pct'] = (timestep_stats['unknown_count'] / timestep_stats['total_count'] * 100).round(2)

# Save summary statistics
summary_path = OUTPUT_DIR / 'timestep_summary.csv'
timestep_stats.to_csv(summary_path)
print(f"\nSummary saved to: {summary_path}")
print(f"\nTimestep statistics:\n{timestep_stats}")

# Create visualizations
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Total records per timestep
axes[0, 0].bar(timestep_stats.index, timestep_stats['total_count'], color='skyblue', edgecolor='black')
axes[0, 0].set_xlabel('Timestep', fontsize=12)
axes[0, 0].set_ylabel('Number of Transactions', fontsize=12)
axes[0, 0].set_title('Total Transactions per Timestep', fontsize=14, fontweight='bold')
axes[0, 0].grid(axis='y', alpha=0.3)

# 2. Stacked bar chart - absolute counts
timestep_stats[['licit_count', 'illicit_count', 'unknown_count']].plot(
    kind='bar', stacked=True, ax=axes[0, 1],
    color=['#2ecc71', '#e74c3c', '#95a5a6'],
    edgecolor='black', linewidth=0.5
)
axes[0, 1].set_xlabel('Timestep', fontsize=12)
axes[0, 1].set_ylabel('Number of Transactions', fontsize=12)
axes[0, 1].set_title('Transaction Class Distribution per Timestep (Absolute)', fontsize=14, fontweight='bold')
axes[0, 1].legend(['Licit', 'Illicit', 'Unknown'], loc='upper right')
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. Stacked bar chart - percentages
timestep_stats[['licit_pct', 'illicit_pct', 'unknown_pct']].plot(
    kind='bar', stacked=True, ax=axes[1, 0],
    color=['#2ecc71', '#e74c3c', '#95a5a6'],
    edgecolor='black', linewidth=0.5
)
axes[1, 0].set_xlabel('Timestep', fontsize=12)
axes[1, 0].set_ylabel('Percentage (%)', fontsize=12)
axes[1, 0].set_title('Transaction Class Distribution per Timestep (Percentage)', fontsize=14, fontweight='bold')
axes[1, 0].legend(['Licit %', 'Illicit %', 'Unknown %'], loc='upper right')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].set_ylim([0, 100])

# 4. Line plot showing class percentages over time
axes[1, 1].plot(timestep_stats.index, timestep_stats['licit_pct'], marker='o', label='Licit', color='#2ecc71', linewidth=2)
axes[1, 1].plot(timestep_stats.index, timestep_stats['illicit_pct'], marker='s', label='Illicit', color='#e74c3c', linewidth=2)
axes[1, 1].plot(timestep_stats.index, timestep_stats['unknown_pct'], marker='^', label='Unknown', color='#95a5a6', linewidth=2)
axes[1, 1].set_xlabel('Timestep', fontsize=12)
axes[1, 1].set_ylabel('Percentage (%)', fontsize=12)
axes[1, 1].set_title('Class Percentage Trends over Timesteps', fontsize=14, fontweight='bold')
axes[1, 1].legend(loc='upper right')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim([0, 105])

plt.tight_layout()
plot_path = OUTPUT_DIR / 'timestep_analysis.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {plot_path}")

# Print summary statistics
print("\n=== Overall Summary ===")
print(f"Total timesteps: {len(timestep_stats)}")
print(f"Total transactions: {timestep_stats['total_count'].sum()}")
print(f"Overall licit: {timestep_stats['licit_count'].sum()} ({timestep_stats['licit_count'].sum() / timestep_stats['total_count'].sum() * 100:.2f}%)")
print(f"Overall illicit: {timestep_stats['illicit_count'].sum()} ({timestep_stats['illicit_count'].sum() / timestep_stats['total_count'].sum() * 100:.2f}%)")
print(f"Overall unknown: {timestep_stats['unknown_count'].sum()} ({timestep_stats['unknown_count'].sum() / timestep_stats['total_count'].sum() * 100:.2f}%)")

plt.show()
