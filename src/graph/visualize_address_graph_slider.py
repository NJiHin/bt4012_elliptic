"""
Generate interactive address-level graph visualization with timestep slider and Louvain community detection.

This script creates separate HTML files for each timestep with a main index file
that uses an iframe to switch between timesteps. Much faster loading than embedding all graphs.

Usage:
    python src/graph/visualize_address_graph_slider.py
"""

from pathlib import Path
import pandas as pd
import networkx as nx
from pyvis.network import Network
import community.community_louvain as louvain
from tqdm import tqdm
import json


# Paths
DATA_DIR = Path("processed_data/address_graphs")
OUTPUT_DIR = Path("src/graph/visualizations/address_graphs")

# Color scheme for labels
LABEL_COLORS = {
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
    """Load address graph data."""
    print("Loading address graph data...")

    labels_df = pd.read_csv(DATA_DIR / 'address_labels.csv')
    features_df = pd.read_csv(DATA_DIR / 'address_features.csv')
    edges_df = pd.read_csv(DATA_DIR / 'address_edgelist.csv')

    # Merge labels with features
    nodes_df = features_df.merge(labels_df, on=['address', 'idx'], how='left')

    print(f"✓ Loaded {len(nodes_df):,} addresses")
    print(f"✓ Loaded {len(edges_df):,} edges")

    return nodes_df, edges_df


def build_timestep_graph(nodes_df, edges_df, timestep, sample_size=None):
    """Build NetworkX graph for a specific timestep with optional sampling."""
    # Filter nodes active in this timestep
    timestep_nodes = nodes_df[
        (nodes_df['first_timestep'] <= timestep) &
        (nodes_df['last_timestep'] >= timestep)
    ].copy()

    # Filter edges for this timestep
    timestep_edges = edges_df[edges_df['timestep'] == timestep].copy()

    # Sample if requested (for performance)
    if sample_size and len(timestep_nodes) > sample_size:
        # Step 1: Always include ALL fraudulent nodes
        fraudulent = timestep_nodes[timestep_nodes['label'] == '1']
        fraudulent_addresses = set(fraudulent['address'])

        # Step 2: Find all nodes connected to fraudulent nodes (1-hop neighbors)
        # Get edges involving fraudulent nodes
        fraudulent_edges = timestep_edges[
            timestep_edges['source'].isin(fraudulent_addresses) |
            timestep_edges['target'].isin(fraudulent_addresses)
        ]

        # Get all addresses connected to fraudulent nodes
        connected_to_fraudulent = set(fraudulent_edges['source']) | set(fraudulent_edges['target'])
        connected_to_fraudulent -= fraudulent_addresses  # Remove fraudulent nodes themselves

        # Get the connected nodes
        neighbors = timestep_nodes[timestep_nodes['address'].isin(connected_to_fraudulent)]

        # Step 3: Combine fraudulent + neighbors
        sampled_nodes = [fraudulent, neighbors]
        remaining_slots = sample_size - len(fraudulent) - len(neighbors)

        # Step 4: Sample remaining nodes if we haven't reached the limit
        if remaining_slots > 0:
            # Get all other nodes (not fraudulent, not neighbors)
            already_included = fraudulent_addresses | connected_to_fraudulent
            remaining_nodes = timestep_nodes[~timestep_nodes['address'].isin(already_included)]

            if len(remaining_nodes) > remaining_slots:
                sampled_remaining = remaining_nodes.sample(n=remaining_slots, random_state=42)
                sampled_nodes.append(sampled_remaining)
            else:
                sampled_nodes.append(remaining_nodes)

        timestep_nodes = pd.concat(sampled_nodes)

        # Filter edges to only include sampled nodes
        sampled_addresses = set(timestep_nodes['address'])
        timestep_edges = timestep_edges[
            timestep_edges['source'].isin(sampled_addresses) &
            timestep_edges['target'].isin(sampled_addresses)
        ]

    # Build NetworkX graph
    G = nx.DiGraph()

    # Add nodes with attributes
    for _, row in timestep_nodes.iterrows():
        G.add_node(
            row['address'],
            label=row['label'],
            num_transactions=row['num_transactions_x'],
            total_btc_sent=row['total_btc_sent'],
            total_btc_received=row['total_btc_received'],
            in_degree=row['in_degree'],
            out_degree=row['out_degree']
        )

    # Add edges with weights
    for _, row in timestep_edges.iterrows():
        if G.has_node(row['source']) and G.has_node(row['target']):
            G.add_edge(
                row['source'],
                row['target'],
                weight=row['btc_amount']
            )

    # Remove isolated nodes (nodes with no edges)
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)

    return G


def detect_communities(G):
    """Detect communities using Louvain algorithm."""
    # Convert to undirected for community detection
    G_undirected = G.to_undirected()

    # Run Louvain community detection
    communities = louvain.best_partition(G_undirected, random_state=42)

    return communities


def generate_community_colors(num_communities):
    """Generate distinct colors for communities."""
    # Use a color palette
    import colorsys
    colors = []
    for i in range(num_communities):
        hue = i / num_communities
        rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        hex_color = '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_color)
    return colors


def create_pyvis_network(G, communities, timestep, color_by='label'):
    """Create PyVis network from NetworkX graph."""
    # Initialize PyVis network
    net = Network(
        height='100vh',
        width='100vw',
        bgcolor='#0f1115',
        font_color='white',
        directed=True
    )

    # Configure physics
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.08,
                "damping": 0.4,
                "avoidOverlap": 0.5
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {
                "enabled": true,
                "iterations": 100
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": true
        },
        "nodes": {
            "font": {
                "size": 0
            }
        },
        "edges": {
            "color": {
                "color": "#666666",
                "opacity": 0.3
            },
            "arrows": {
                "to": {
                    "enabled": true,
                    "scaleFactor": 0.5
                }
            },
            "smooth": false
        }
    }
    """)

    # Get community colors
    num_communities = len(set(communities.values()))
    community_colors = generate_community_colors(num_communities)

    # Add nodes
    for node in G.nodes():
        node_data = G.nodes[node]
        label = node_data.get('label', 'unknown')

        # Determine color
        if color_by == 'label':
            color = LABEL_COLORS.get(label, '#FFDD44')
        else:  # color by community
            community_id = communities.get(node, 0)
            color = community_colors[community_id % len(community_colors)]

        # Node size based on degree
        degree = G.degree(node)
        size = 10 + min(degree, 50)  # Cap size for visibility

        # Create hover title (plain text, no HTML tags)
        title = (
            f"Address: {node[:16]}...\n"
            f"Label: {CLASS_LABELS.get(label, 'Unknown')}\n"
            f"Community: {communities.get(node, 'N/A')}\n"
            f"Degree: {degree}\n"
            f"Transactions: {node_data.get('num_transactions', 0)}\n"
            f"BTC Sent: {node_data.get('total_btc_sent', 0):.4f}\n"
            f"BTC Received: {node_data.get('total_btc_received', 0):.4f}"
        )

        net.add_node(
            node,
            label='',  # Empty label to prevent text from showing when zoomed in
            color=color,
            size=size,
            title=title
        )

    # Add edges
    for source, target, data in G.edges(data=True):
        weight = data.get('weight', 0)

        # Edge width based on BTC amount
        width = 1 + min(weight / 10, 5)  # Cap width

        title = f"BTC: {weight:.4f}"

        net.add_edge(
            source,
            target,
            value=width,
            title=title
        )

    return net


def generate_timestep_files(nodes_df, edges_df, sample_size=None, color_by='label'):
    """Generate separate HTML file for each timestep."""
    if sample_size:
        print(f"\nGenerating timestep files (sampling {sample_size} nodes)...")
        print("  - Including: ALL fraudulent nodes")
        print("  - Including: ALL nodes connected to fraudulent nodes")
        print("  - Random sample of remaining nodes")
    else:
        print(f"\nGenerating timestep files (full graphs - all nodes)...")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestep_stats = {}
    available_timesteps = sorted(edges_df['timestep'].unique())

    for timestep in tqdm(available_timesteps):
        # Build graph for timestep
        G = build_timestep_graph(nodes_df, edges_df, timestep, sample_size=sample_size)

        if len(G.nodes()) == 0:
            continue

        # Detect communities
        communities = detect_communities(G)

        # Create PyVis network
        net = create_pyvis_network(G, communities, timestep, color_by=color_by)

        # Save to file
        output_file = OUTPUT_DIR / f"timestep_{timestep}.html"
        net.save_graph(str(output_file))

        # Store stats
        num_illicit = sum(1 for n in G.nodes() if G.nodes[n].get('label') == '1')
        num_licit = sum(1 for n in G.nodes() if G.nodes[n].get('label') == '2')
        num_unknown = sum(1 for n in G.nodes() if G.nodes[n].get('label') == 'unknown')

        timestep_stats[int(timestep)] = {
            'nodes': len(G.nodes()),
            'edges': len(G.edges()),
            'communities': len(set(communities.values())),
            'illicit': num_illicit,
            'licit': num_licit,
            'unknown': num_unknown
        }

    return timestep_stats, available_timesteps


def generate_index_html(timestep_stats, available_timesteps):
    """Generate main index HTML with slider that loads timestep files via iframe."""
    min_timestep = min(available_timesteps)
    max_timestep = max(available_timesteps)

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Address Graph Viewer - Louvain Communities</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0f1115;
            --panel-bg: rgba(30, 32, 38, 0.95);
            --text-primary: #ffffff;
            --text-secondary: #a0a0a0;
            --accent: #4444FF;
            --accent-hover: #5555FF;
            --border: rgba(255, 255, 255, 0.1);
        }}

        body {{
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            overflow: hidden;
        }}

        #graph-iframe {{
            width: 100vw;
            height: 100vh;
            border: none;
            position: absolute;
            top: 0;
            left: 0;
            z-index: 1;
        }}

        /* Sidebar */
        #sidebar {{
            position: absolute;
            top: 20px;
            left: 20px;
            width: 320px;
            background: var(--panel-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            z-index: 10;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            display: flex;
            flex-direction: column;
            gap: 20px;
            max-height: calc(100vh - 40px);
            overflow-y: auto;
        }}

        h1 {{
            margin: 0;
            font-size: 18px;
            font-weight: 600;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        h1 i {{ color: var(--accent); }}

        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}

        .label {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
            font-weight: 600;
        }}

        .playback-controls {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        button {{
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid var(--border);
            color: white;
            padding: 8px 12px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            font-family: 'Inter', sans-serif;
        }}

        button:hover {{
            background: rgba(255, 255, 255, 0.2);
            border-color: rgba(255, 255, 255, 0.3);
        }}

        button.primary {{
            background: var(--accent);
            border-color: var(--accent);
        }}

        button.primary:hover {{
            background: var(--accent-hover);
        }}

        input[type="range"] {{
            width: 100%;
            height: 4px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 2px;
            outline: none;
            -webkit-appearance: none;
        }}

        input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            background: var(--accent);
            border-radius: 50%;
            cursor: pointer;
            border: 2px solid white;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        }}

        #timestep-display {{
            font-size: 24px;
            font-weight: 500;
            color: var(--accent);
            font-variant-numeric: tabular-nums;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }}

        .stat-card {{
            background: rgba(255, 255, 255, 0.05);
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }}

        .stat-value {{
            display: block;
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 4px;
        }}

        .stat-label {{
            font-size: 11px;
            color: var(--text-secondary);
        }}

        .legend-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }}

        #loader {{
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(15, 17, 21, 0.9);
            z-index: 100;
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            transition: opacity 0.3s;
            pointer-events: none;
            opacity: 0;
        }}

        #loader.active {{
            opacity: 1;
            pointer-events: all;
        }}

        .spinner {{
            width: 40px;
            height: 40px;
            border: 4px solid rgba(255,255,255,0.1);
            border-left-color: var(--accent);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}

        @keyframes spin {{ 100% {{ transform: rotate(360deg); }} }}
    </style>
</head>
<body>
    <div id="loader" class="active">
        <div class="spinner"></div>
        <p style="margin-top: 20px; color: var(--text-secondary);">Loading Graph...</p>
    </div>

    <div id="sidebar">
        <h1><i class="fas fa-project-diagram"></i> Address Graph</h1>

        <div class="control-group">
            <div class="label">Timeline</div>
            <div style="display: flex; justify-content: space-between; align-items: flex-end;">
                <div id="timestep-display">T-{min_timestep}</div>
                <div class="playback-controls">
                    <button id="btn-prev"><i class="fas fa-step-backward"></i></button>
                    <button id="btn-play" class="primary"><i class="fas fa-play"></i></button>
                    <button id="btn-next"><i class="fas fa-step-forward"></i></button>
                </div>
            </div>
            <input type="range" id="timestep-slider" min="{min_timestep}" max="{max_timestep}" value="{min_timestep}" step="1">
        </div>

        <div class="control-group">
            <div class="label">Statistics</div>
            <div class="stats-grid">
                <div class="stat-card">
                    <span class="stat-value" id="stat-nodes">0</span>
                    <span class="stat-label">Nodes</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value" id="stat-edges">0</span>
                    <span class="stat-label">Edges</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value" id="stat-communities">0</span>
                    <span class="stat-label">Communities</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value" id="stat-illicit">0</span>
                    <span class="stat-label">Illicit</span>
                </div>
            </div>
        </div>

        <div class="control-group">
            <div class="label">Legend</div>
            <div style="display: flex; flex-direction: column; gap: 8px; font-size: 13px;">
                <span><span class="legend-dot" style="background: {LABEL_COLORS['1']}"></span>Illicit</span>
                <span><span class="legend-dot" style="background: {LABEL_COLORS['2']}"></span>Licit</span>
                <span><span class="legend-dot" style="background: {LABEL_COLORS['unknown']}"></span>Unknown</span>
            </div>
        </div>
    </div>

    <iframe id="graph-iframe" src=""></iframe>

    <script>
        const timestepStats = {json.dumps(timestep_stats)};
        const minTimestep = {min_timestep};
        const maxTimestep = {max_timestep};

        let currentTimestep = minTimestep;
        let isPlaying = false;
        let playInterval = null;

        const iframe = document.getElementById('graph-iframe');
        const slider = document.getElementById('timestep-slider');
        const display = document.getElementById('timestep-display');
        const btnPlay = document.getElementById('btn-play');
        const loader = document.getElementById('loader');

        function updateTimestep(timestep) {{
            currentTimestep = timestep;
            slider.value = timestep;
            display.textContent = `T-${{timestep}}`;

            // Show loader
            loader.classList.add('active');

            // Load timestep HTML in iframe
            iframe.src = `address_graphs/timestep_${{timestep}}.html`;

            // Update stats
            const stats = timestepStats[timestep.toString()];
            if (stats) {{
                document.getElementById('stat-nodes').textContent = stats.nodes.toLocaleString();
                document.getElementById('stat-edges').textContent = stats.edges.toLocaleString();
                document.getElementById('stat-communities').textContent = stats.communities;
                document.getElementById('stat-illicit').textContent = stats.illicit;
            }}
        }}

        function togglePlay() {{
            isPlaying = !isPlaying;
            const icon = btnPlay.querySelector('i');

            if (isPlaying) {{
                icon.className = 'fas fa-pause';
                playInterval = setInterval(() => {{
                    if (currentTimestep < maxTimestep) {{
                        updateTimestep(currentTimestep + 1);
                    }} else {{
                        togglePlay();
                    }}
                }}, 2000);
            }} else {{
                icon.className = 'fas fa-play';
                clearInterval(playInterval);
            }}
        }}

        slider.addEventListener('input', (e) => {{
            if (isPlaying) togglePlay();
            updateTimestep(parseInt(e.target.value));
        }});

        btnPlay.addEventListener('click', togglePlay);

        document.getElementById('btn-prev').addEventListener('click', () => {{
            if (currentTimestep > minTimestep) updateTimestep(currentTimestep - 1);
        }});

        document.getElementById('btn-next').addEventListener('click', () => {{
            if (currentTimestep < maxTimestep) updateTimestep(currentTimestep + 1);
        }});

        // Hide loader when iframe loads
        iframe.addEventListener('load', () => {{
            setTimeout(() => loader.classList.remove('active'), 300);
        }});

        // Initial load
        window.addEventListener('load', () => {{
            updateTimestep(minTimestep);
        }});
    </script>
</body>
</html>
"""
    return html


def main():
    """Main execution function."""
    print("="*60)
    print("Address Graph Visualization with Louvain Communities")
    print("="*60)

    # Load data
    nodes_df, edges_df = load_data()

    # Generate separate HTML file for each timestep
    timestep_stats, available_timesteps = generate_timestep_files(
        nodes_df,
        edges_df,
        sample_size=5000,   # Sample 5000 nodes per timestep (always includes ALL fraudulent)
        color_by='label'    # Color by 'label' or 'community'
    )

    # Generate index HTML with slider
    print("\nGenerating index HTML with slider...")
    html = generate_index_html(timestep_stats, available_timesteps)

    # Save index file
    index_file = OUTPUT_DIR.parent / "address_graph_viewer.html"
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write(html)

    # Calculate total size
    total_size = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.html"))
    total_size_mb = total_size / (1024 * 1024)

    print("\n" + "="*60)
    print("✓ Visualization generated!")
    print("="*60)
    print(f"\nIndex file: {index_file.absolute()}")
    print(f"Timestep files: {OUTPUT_DIR.absolute()}")
    print(f"Total files: {len(list(OUTPUT_DIR.glob('*.html')))} timesteps + 1 index")
    print(f"Total size: {total_size_mb:.2f} MB")
    print(f"Timesteps: {min(available_timesteps)} to {max(available_timesteps)}")
    print(f"\nFeatures:")
    print("  - Separate HTML file per timestep (faster loading)")
    print("  - Louvain community detection per timestep")
    print("  - Interactive PyVis network visualization")
    print("  - Temporal slider to navigate timesteps")
    print("  - Playback controls")
    print("  - Node hover tooltips with detailed info")
    print("  - Smart sampling: ALL fraudulent + neighbors + random (5000 total)")
    print("  - Isolated nodes removed (only showing connected nodes)")
    print(f"\nOpen {index_file.name} in your browser to explore!")


if __name__ == "__main__":
    main()
