"""
Generate interactive graph visualization with timestep slider.

This script creates a single HTML file with all timesteps and a slider
to navigate between them dynamically.

Usage:
    python src/graph/visualize_graph_slider.py
"""

from pathlib import Path
import pandas as pd
import json
from tqdm import tqdm


# Paths
RAW_DATA_DIR = Path("raw_data/elliptic_bitcoin_dataset")
GRAPH_DIR = Path("src/graph")

UMAP_FILE = GRAPH_DIR / "umap_coordinates.csv"
EDGELIST_FILE = RAW_DATA_DIR / "elliptic_txs_edgelist.csv"
OUTPUT_DIR = GRAPH_DIR / "visualizations"

# Color scheme
COLORS = {
    '1': '#FF4444',      # Illicit - Red
    '2': '#4444FF',      # Licit - Blue
    'unknown': '#FFDD44' # Unknown - Yellow
}

CLASS_LABELS = {
    '1': 'Illicit',
    '2': 'Licit',
    'unknown': 'Unknown'
}


def load_data():
    """Load UMAP coordinates and edgelist."""
    print("Loading UMAP coordinates...")
    if not UMAP_FILE.exists():
        raise FileNotFoundError(
            f"UMAP coordinates not found at {UMAP_FILE}\n"
            f"Please run compute_umap_embeddings.py first."
        )

    coords_df = pd.read_csv(UMAP_FILE)
    print(f"✓ Loaded {len(coords_df)} transaction coordinates")

    print("\nLoading transaction edgelist...")
    edges_df = pd.read_csv(EDGELIST_FILE)
    print(f"✓ Loaded {len(edges_df)} edges")

    return coords_df, edges_df


def prepare_timestep_data(coords_df, edges_df):
    """Prepare all timestep data as JSON."""
    print("\nPreparing timestep data...")

    timesteps_data = {}
    scale_factor = 1000

    available_timesteps = sorted(coords_df['time_step'].unique())

    for timestep in tqdm(available_timesteps):
        # Filter nodes for this timestep
        timestep_nodes = coords_df[coords_df['time_step'] == timestep].copy()
        node_ids = set(timestep_nodes['txId'].values)

        # Filter edges: both source and target must be in this timestep
        timestep_edges = edges_df[
            edges_df['txId1'].isin(node_ids) & edges_df['txId2'].isin(node_ids)
        ]

        # Prepare nodes data
        nodes = []
        for _, row in timestep_nodes.iterrows():
            class_label = str(row['class'])
            nodes.append({
                'id': str(row['txId']),
                'x': float(row['umap_x'] * scale_factor),
                'y': float(row['umap_y'] * scale_factor),
                'group': CLASS_LABELS.get(class_label, 'Unknown'),
                'class': class_label,
                'umap_x': float(row['umap_x']),
                'umap_y': float(row['umap_y'])
            })

        # Prepare edges data
        edges = []
        for _, row in timestep_edges.iterrows():
            edges.append({
                'from': str(row['txId1']),
                'to': str(row['txId2'])
            })

        # Statistics
        num_illicit = len(timestep_nodes[timestep_nodes['class'] == '1'])
        num_licit = len(timestep_nodes[timestep_nodes['class'] == '2'])
        num_unknown = len(timestep_nodes[timestep_nodes['class'] == 'unknown'])

        timesteps_data[int(timestep)] = {
            'nodes': nodes,
            'edges': edges,
            'stats': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'illicit': int(num_illicit),
                'licit': int(num_licit),
                'unknown': int(num_unknown)
            }
        }

    return timesteps_data, available_timesteps


def generate_html(timesteps_data, available_timesteps):
    """Generate HTML with slider."""
    min_timestep = min(available_timesteps)
    max_timestep = max(available_timesteps)

    # Convert to JSON
    timesteps_json = json.dumps(timesteps_data)

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Elliptic Bitcoin Transaction Graph - Interactive Timestep Viewer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css" />
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            background-color: #1a1a1a;
            color: white;
        }}

        #controls {{
            background-color: #222222;
            padding: 20px;
            border-bottom: 2px solid #444;
        }}

        h1 {{
            margin: 0 0 20px 0;
            font-size: 24px;
        }}

        #slider-container {{
            margin: 20px 0;
        }}

        #timestep-slider {{
            width: 100%;
            height: 30px;
            background: #444;
            outline: none;
            -webkit-appearance: none;
        }}

        #timestep-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 30px;
            background: #4444FF;
            cursor: pointer;
        }}

        #timestep-slider::-moz-range-thumb {{
            width: 20px;
            height: 30px;
            background: #4444FF;
            cursor: pointer;
        }}

        #timestep-display {{
            font-size: 20px;
            font-weight: bold;
            color: #4444FF;
            margin-bottom: 10px;
        }}

        #stats {{
            margin: 15px 0;
            font-size: 14px;
        }}

        .stat-item {{
            display: inline-block;
            margin-right: 30px;
        }}

        #filter-controls {{
            margin: 15px 0;
        }}

        #filter-controls label {{
            margin-right: 20px;
            cursor: pointer;
        }}

        #network {{
            width: 100%;
            height: calc(100vh - 250px);
            background-color: #222222;
        }}

        .color-box {{
            display: inline-block;
            width: 15px;
            height: 15px;
            margin-right: 5px;
            vertical-align: middle;
        }}
    </style>
</head>
<body>
    <div id="controls">
        <h1>Elliptic Bitcoin Transaction Graph - Interactive Timestep Viewer</h1>

        <div id="slider-container">
            <div id="timestep-display">Timestep: {min_timestep}</div>
            <input type="range" id="timestep-slider" min="{min_timestep}" max="{max_timestep}" value="{min_timestep}" step="1">
        </div>

        <div id="stats">
            <span class="stat-item"><strong>Nodes:</strong> <span id="stat-nodes">0</span></span>
            <span class="stat-item"><strong>Edges:</strong> <span id="stat-edges">0</span></span>
            <span class="stat-item">
                <span class="color-box" style="background-color: {COLORS['1']}"></span>
                <strong>Illicit:</strong> <span id="stat-illicit">0</span>
            </span>
            <span class="stat-item">
                <span class="color-box" style="background-color: {COLORS['2']}"></span>
                <strong>Licit:</strong> <span id="stat-licit">0</span>
            </span>
            <span class="stat-item">
                <span class="color-box" style="background-color: {COLORS['unknown']}"></span>
                <strong>Unknown:</strong> <span id="stat-unknown">0</span>
            </span>
        </div>

        <div id="filter-controls">
            <strong>Filter by Class:</strong>
            <label>
                <input type="checkbox" id="filter-illicit" checked>
                <span class="color-box" style="background-color: {COLORS['1']}"></span>
                Illicit
            </label>
            <label>
                <input type="checkbox" id="filter-licit" checked>
                <span class="color-box" style="background-color: {COLORS['2']}"></span>
                Licit
            </label>
            <label>
                <input type="checkbox" id="filter-unknown" checked>
                <span class="color-box" style="background-color: {COLORS['unknown']}"></span>
                Unknown
            </label>
        </div>
    </div>

    <div id="network"></div>

    <script>
        // Load all timestep data
        const timestepsData = {timesteps_json};

        // Initialize network
        const container = document.getElementById('network');
        let network = null;
        let nodesDataset = null;
        let edgesDataset = null;

        const options = {{
            interaction: {{
                hover: true,
                navigationButtons: true,
                keyboard: true
            }},
            edges: {{
                color: '#666666',
                width: 0.5,
                arrows: {{
                    to: {{ enabled: true, scaleFactor: 0.5 }}
                }},
                smooth: {{ enabled: false }}
            }},
            physics: {{
                enabled: false
            }},
            groups: {{
                'Illicit': {{ color: '{COLORS['1']}' }},
                'Licit': {{ color: '{COLORS['2']}' }},
                'Unknown': {{ color: '{COLORS['unknown']}' }}
            }},
            nodes: {{
                shape: 'dot',
                size: 10,
                font: {{ color: 'white' }}
            }}
        }};

        // Current filter state
        const filters = {{
            'Illicit': true,
            'Licit': true,
            'Unknown': true
        }};

        // Update graph for given timestep
        function updateTimestep(timestep) {{
            const data = timestepsData[timestep];
            if (!data) {{
                console.error('No data for timestep', timestep);
                return;
            }}

            // Update display
            document.getElementById('timestep-display').textContent = `Timestep: ${{timestep}}`;

            // Apply filters to nodes
            const filteredNodes = data.nodes
                .filter(node => filters[node.group])
                .map(node => ({{
                    id: node.id,
                    x: node.x,
                    y: node.y,
                    group: node.group,
                    title: `Transaction ID: ${{node.id}}\\nTimestep: ${{timestep}}\\nClass: ${{node.group}}\\nUMAP X: ${{node.umap_x.toFixed(2)}}\\nUMAP Y: ${{node.umap_y.toFixed(2)}}`,
                    label: '',
                    physics: false
                }}));

            const visibleNodeIds = new Set(filteredNodes.map(n => n.id));

            // Filter edges: only show if both endpoints are visible
            const filteredEdges = data.edges
                .filter(edge => visibleNodeIds.has(edge.from) && visibleNodeIds.has(edge.to));

            // Update or create network
            if (network === null) {{
                nodesDataset = new vis.DataSet(filteredNodes);
                edgesDataset = new vis.DataSet(filteredEdges);
                network = new vis.Network(container, {{
                    nodes: nodesDataset,
                    edges: edgesDataset
                }}, options);
            }} else {{
                nodesDataset.clear();
                edgesDataset.clear();
                nodesDataset.add(filteredNodes);
                edgesDataset.add(filteredEdges);
            }}

            // Update stats
            document.getElementById('stat-nodes').textContent = filteredNodes.length;
            document.getElementById('stat-edges').textContent = filteredEdges.length;
            document.getElementById('stat-illicit').textContent = data.stats.illicit;
            document.getElementById('stat-licit').textContent = data.stats.licit;
            document.getElementById('stat-unknown').textContent = data.stats.unknown;
        }}

        // Slider event
        document.getElementById('timestep-slider').addEventListener('input', function(e) {{
            updateTimestep(parseInt(e.target.value));
        }});

        // Filter checkboxes
        document.getElementById('filter-illicit').addEventListener('change', function(e) {{
            filters['Illicit'] = e.target.checked;
            const currentTimestep = parseInt(document.getElementById('timestep-slider').value);
            updateTimestep(currentTimestep);
        }});

        document.getElementById('filter-licit').addEventListener('change', function(e) {{
            filters['Licit'] = e.target.checked;
            const currentTimestep = parseInt(document.getElementById('timestep-slider').value);
            updateTimestep(currentTimestep);
        }});

        document.getElementById('filter-unknown').addEventListener('change', function(e) {{
            filters['Unknown'] = e.target.checked;
            const currentTimestep = parseInt(document.getElementById('timestep-slider').value);
            updateTimestep(currentTimestep);
        }});

        // Initialize with first timestep
        updateTimestep({min_timestep});
    </script>
</body>
</html>
"""

    return html


def main():
    """Main execution function."""
    print("="*60)
    print("Elliptic Graph Visualization with Timestep Slider")
    print("="*60)

    # Load data
    coords_df, edges_df = load_data()

    # Prepare all timestep data
    timesteps_data, available_timesteps = prepare_timestep_data(coords_df, edges_df)

    # Generate HTML
    print("\nGenerating HTML...")
    html = generate_html(timesteps_data, available_timesteps)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "elliptic_timestep_slider.html"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    file_size_mb = output_file.stat().st_size / (1024 * 1024)

    print("\n" + "="*60)
    print("✓ Visualization generated!")
    print("="*60)
    print(f"\nOutput file: {output_file.absolute()}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Timesteps: {min(available_timesteps)} to {max(available_timesteps)}")
    print("\nOpen the HTML file in your browser to explore!")


if __name__ == "__main__":
    main()
