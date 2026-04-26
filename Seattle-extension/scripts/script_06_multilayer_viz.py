"""
script_06_multilayer_viz.py
===========================
Visualizes the multilayer graph geographically, highlighting the
linked Betweenness Centrality (BC) nodes found in script 05.

Produces one map:
1. A full city-scale map of both networks.

Run AFTER script_05_multilayer.py:
    python scripts/script_06_multilayer_viz.py
"""

import matplotlib
matplotlib.use("Agg")

import pickle
import pathlib
import pandas as pd
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).parent.parent
OUT = ROOT / "output"
OUT_MULTI = OUT / "multilayer"
FIG_MULTI = OUT / "figures" / "multilayer"
FIG_MULTI.mkdir(parents=True, exist_ok=True)

DPI = 300
FIG_SIZE = (12, 12)

def main():
    print("=" * 65)
    print("Script 06 — Multilayer Visualization")
    print("=" * 65)

    print("\n[1/2] Loading data...")
    with open(OUT_MULTI / "multilayer_graph.pkl", "rb") as f:
        G_multi = pickle.load(f)
    
    t_linked = pd.read_csv(OUT_MULTI / "removal_order_linked_bc_transport.csv")
    p_linked = pd.read_csv(OUT_MULTI / "removal_order_linked_bc_power.csv")

    t_bc_node = f"T_{int(t_linked.iloc[0]['node_id'])}"
    p_bc_node = f"P_{int(p_linked.iloc[0]['node_id'])}"

    # Extract coordinates in EPSG:2926 (feet)
    # Transport nodes: 'longitude', 'latitude' hold EPSG:2926 coords
    # Power nodes: 'x_ft', 'y_ft' hold EPSG:2926 coords
    t_xs, t_ys = [], []
    p_xs, p_ys = [], []
    
    for nid, data in G_multi.nodes(data=True):
        if data["layer"] == "transport":
            t_xs.append(data["longitude"])
            t_ys.append(data["latitude"])
        elif data["layer"] == "power":
            p_xs.append(data["x_ft"])
            p_ys.append(data["y_ft"])
            
    t_bc_x = G_multi.nodes[t_bc_node]["longitude"]
    t_bc_y = G_multi.nodes[t_bc_node]["latitude"]
    
    p_bc_x = G_multi.nodes[p_bc_node]["x_ft"]
    p_bc_y = G_multi.nodes[p_bc_node]["y_ft"]

    # ── Full Map ──
    print("\n[2/2] Generating full multilayer map...")
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    ax.scatter(t_xs, t_ys, s=0.5, color="#1f77b4", alpha=0.3, label="Transport Layer", rasterized=True)
    ax.scatter(p_xs, p_ys, s=0.5, color="#ff7f0e", alpha=0.3, label="Power Layer", rasterized=True)
    
    # Highlight linked BC nodes
    ax.scatter([t_bc_x], [t_bc_y], s=150, color="cyan", edgecolors="black", marker="*", zorder=5, label=f"Transport BC Node ({t_bc_node})")
    ax.scatter([p_bc_x], [p_bc_y], s=150, color="red", edgecolors="black", marker="X", zorder=5, label=f"Power BC Node ({p_bc_node})")
    
    # Draw a line between them
    ax.plot([t_bc_x, p_bc_x], [t_bc_y, p_bc_y], color="black", linestyle="--", linewidth=2, zorder=4, label="Interdependent Link")
    
    ax.set_aspect("equal")
    ax.set_title("Seattle Multilayer Network: Transport & Power\nHighlighting Linked Top BC Nodes", fontsize=14)
    ax.set_xlabel("Easting (ft, EPSG:2926)", fontsize=11)
    ax.set_ylabel("Northing (ft, EPSG:2926)", fontsize=11)
    ax.legend(loc="lower right", fontsize=10, markerscale=2)
    
    fig.tight_layout()
    full_map_path = FIG_MULTI / "multilayer_full_map.png"
    fig.savefig(full_map_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved: {full_map_path.relative_to(ROOT)}")
    
    print("\n" + "=" * 65)
    print("Done.")
    print("=" * 65)

if __name__ == "__main__":
    main()