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
    """Generate HTML with modern UI, playback, and search."""
    min_timestep = min(available_timesteps)
    max_timestep = max(available_timesteps)

    # Convert to JSON
    timesteps_json = json.dumps(timesteps_data)

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Elliptic Graph Viewer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css" />
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0f1115;
            --panel-bg: rgba(30, 32, 38, 0.85);
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

        #network {{
            width: 100vw;
            height: 100vh;
            position: absolute;
            top: 0;
            left: 0;
            z-index: 1;
        }}

        /* Floating Sidebar */
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

        /* Controls Section */
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

        /* Slider & Playback */
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

        /* Stats Grid */
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

        /* Search */
        .search-box {{
            position: relative;
        }}

        .search-box input {{
            width: 100%;
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--border);
            padding: 10px 10px 10px 35px;
            border-radius: 8px;
            color: white;
            font-family: 'Inter', sans-serif;
            box-sizing: border-box;
        }}

        .search-box i {{
            position: absolute;
            left: 12px;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-secondary);
        }}

        /* Filters */
        .filter-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}

        .filter-row:last-child {{ border-bottom: none; }}

        .toggle-switch {{
            position: relative;
            display: inline-block;
            width: 36px;
            height: 20px;
        }}

        .toggle-switch input {{ opacity: 0; width: 0; height: 0; }}

        .slider {{
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(255, 255, 255, 0.2);
            transition: .4s;
            border-radius: 20px;
        }}

        .slider:before {{
            position: absolute;
            content: "";
            height: 16px;
            width: 16px;
            left: 2px;
            bottom: 2px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }}

        input:checked + .slider {{ background-color: var(--accent); }}
        input:checked + .slider:before {{ transform: translateX(16px); }}

        /* Legend Colors */
        .legend-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }}

        /* Loading Overlay */
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

        /* Zoom Controls */
        #zoom-controls {{
            position: absolute;
            bottom: 20px;
            right: 20px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            z-index: 10;
        }}

        #zoom-controls button {{
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: var(--panel-bg);
            backdrop-filter: blur(12px);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
        }}
    </style>
</head>
<body>
    <div id="loader" class="active">
        <div class="spinner"></div>
        <p style="margin-top: 20px; color: var(--text-secondary);">Loading Graph Data...</p>
    </div>

    <div id="sidebar">
        <h1><i class="fas fa-project-diagram"></i> Elliptic Graph</h1>

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
            <div class="label">Search</div>
            <div class="search-box">
                <i class="fas fa-search"></i>
                <input type="text" id="search-input" placeholder="Search Transaction ID...">
            </div>
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
            </div>
        </div>

        <div class="control-group">
            <div class="label">Filters</div>
            <div class="filter-row">
                <span><span class="legend-dot" style="background: {COLORS['1']}"></span>Illicit</span>
                <label class="toggle-switch">
                    <input type="checkbox" id="filter-illicit" checked>
                    <span class="slider"></span>
                </label>
            </div>
            <div class="filter-row">
                <span><span class="legend-dot" style="background: {COLORS['2']}"></span>Licit</span>
                <label class="toggle-switch">
                    <input type="checkbox" id="filter-licit" checked>
                    <span class="slider"></span>
                </label>
            </div>
            <div class="filter-row">
                <span><span class="legend-dot" style="background: {COLORS['unknown']}"></span>Unknown</span>
                <label class="toggle-switch">
                    <input type="checkbox" id="filter-unknown" checked>
                    <span class="slider"></span>
                </label>
            </div>
        </div>
    </div>

    <div id="zoom-controls">
        <button id="btn-zoom-in"><i class="fas fa-plus"></i></button>
        <button id="btn-fit"><i class="fas fa-compress-arrows-alt"></i></button>
        <button id="btn-zoom-out"><i class="fas fa-minus"></i></button>
    </div>

    <div id="network"></div>

    <script>
        // Data
        const timestepsData = {timesteps_json};
        const minTimestep = {min_timestep};
        const maxTimestep = {max_timestep};
        
        // State
        let currentTimestep = minTimestep;
        let isPlaying = false;
        let playInterval = null;
        let network = null;
        let nodesDataset = new vis.DataSet();
        let edgesDataset = new vis.DataSet();
        
        // DOM Elements
        const container = document.getElementById('network');
        const slider = document.getElementById('timestep-slider');
        const display = document.getElementById('timestep-display');
        const btnPlay = document.getElementById('btn-play');
        const loader = document.getElementById('loader');

        // Network Options
        const options = {{
            interaction: {{
                hover: true,
                navigationButtons: false,
                keyboard: true,
                zoomView: true
            }},
            edges: {{
                color: {{ color: '#666666', opacity: 0.3 }},
                width: 1,
                arrows: {{ to: {{ enabled: true, scaleFactor: 0.5 }} }},
                smooth: {{ enabled: false }}
            }},
            physics: {{
                enabled: false,
                stabilization: false
            }},
            nodes: {{
                shape: 'dot',
                size: 8,
                font: {{ color: 'white', size: 12, face: 'Inter' }},
                borderWidth: 0
            }}
        }};

        // Initialize Network
        network = new vis.Network(container, {{ nodes: nodesDataset, edges: edgesDataset }}, options);

        // Filter State
        const filters = {{
            'Illicit': true,
            'Licit': true,
            'Unknown': true
        }};

        // --- Functions ---

        function updateTimestep(timestep) {{
            // Show loader if jumping far (simulated)
            // loader.classList.add('active');
            
            const data = timestepsData[timestep];
            if (!data) return;

            currentTimestep = timestep;
            slider.value = timestep;
            display.textContent = `T-${{timestep}}`;

            // Filter Nodes
            const filteredNodes = data.nodes
                .filter(node => filters[node.group])
                .map(node => ({{
                    id: node.id,
                    x: node.x,
                    y: node.y,
                    color: node.group === 'Illicit' ? '{COLORS['1']}' : 
                           node.group === 'Licit' ? '{COLORS['2']}' : '{COLORS['unknown']}',
                    title: `
                        <div style="padding: 8px; font-family: Inter; font-size: 12px;">
                            <strong>ID:</strong> ${{node.id}}<br>
                            <strong>Class:</strong> ${{node.group}}<br>
                            <strong>Timestep:</strong> ${{timestep}}
                        </div>
                    `
                }}));

            const visibleNodeIds = new Set(filteredNodes.map(n => n.id));
            const filteredEdges = data.edges
                .filter(edge => visibleNodeIds.has(edge.from) && visibleNodeIds.has(edge.to));

            // Batch update
            nodesDataset.clear();
            edgesDataset.clear();
            nodesDataset.add(filteredNodes);
            edgesDataset.add(filteredEdges);

            // Update Stats
            document.getElementById('stat-nodes').textContent = filteredNodes.length.toLocaleString();
            document.getElementById('stat-edges').textContent = filteredEdges.length.toLocaleString();
            
            // loader.classList.remove('active');
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
                        togglePlay(); // Stop at end
                    }}
                }}, 1000); // 1 second per timestep
            }} else {{
                icon.className = 'fas fa-play';
                clearInterval(playInterval);
            }}
        }}

        function searchTransaction() {{
            const query = document.getElementById('search-input').value.trim();
            if (!query) return;

            // Search in current timestep first
            let node = nodesDataset.get(query);
            
            if (node) {{
                network.focus(node.id, {{
                    scale: 1.5,
                    animation: {{ duration: 1000, easingFunction: 'easeInOutQuad' }}
                }});
                network.selectNodes([node.id]);
            }} else {{
                alert('Transaction ID not found in current timestep view.');
            }}
        }}

        // --- Event Listeners ---

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

        // Filters
        ['Illicit', 'Licit', 'Unknown'].forEach(type => {{
            document.getElementById(`filter-${{type.toLowerCase()}}`).addEventListener('change', (e) => {{
                filters[type] = e.target.checked;
                updateTimestep(currentTimestep);
            }});
        }});

        // Search
        document.getElementById('search-input').addEventListener('keypress', (e) => {{
            if (e.key === 'Enter') searchTransaction();
        }});

        // Zoom Controls
        document.getElementById('btn-zoom-in').addEventListener('click', () => network.moveTo({{ scale: network.getScale() * 1.2 }}));
        document.getElementById('btn-zoom-out').addEventListener('click', () => network.moveTo({{ scale: network.getScale() * 0.8 }}));
        document.getElementById('btn-fit').addEventListener('click', () => network.fit({{ animation: true }}));

        // Initial Load
        window.addEventListener('load', () => {{
            updateTimestep(minTimestep);
            setTimeout(() => loader.classList.remove('active'), 500);
        }});

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
