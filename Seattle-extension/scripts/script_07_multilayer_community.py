"""
script_07_multilayer_community.py
================================
Performs multilayer community detection by creating "dual identity" pairings
between Road (Transport) and Power nodes, restricted to GEOGRAPHICALLY LINKED nodes.

Adjustments:
  1. Only considers nodes with interlayer links < 500 feet (overlapping).
  2. Produces monolayer community reports for only these linked nodes.
  3. Performs dual-identity analysis on the filtered linked subset.
  4. Highlights the BC attack nodes on the multilayer map.
  5. Plots monolayer community maps for the linked node subsets.

Run AFTER script_06_multilayer_viz.py:
    python scripts/script_07_multilayer_community.py
"""

import matplotlib
matplotlib.use("Agg")

import pickle
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from pyproj import Transformer
import numpy as np

# Threshold for "overlapping" (feet)
LINK_THRESHOLD_FT = 500.0

# Coordinate transformation: Transport (EPSG:2926) -> WGS84
_TO_WGS84 = Transformer.from_crs("EPSG:2926", "EPSG:4326", always_xy=True)

ROOT = pathlib.Path(__file__).parent.parent
OUT = ROOT / "output"
OUT_MULTI = OUT / "multilayer"
FIG_MULTI = OUT / "figures" / "multilayer"
OUT_MULTI.mkdir(parents=True, exist_ok=True)
FIG_MULTI.mkdir(parents=True, exist_ok=True)

DPI = 150
FIG_SIZE = (12, 10)

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def get_linked_data(t_asgn, p_asgn, links_df, scenario_name):
    # Filter links by distance
    linked_df = links_df[links_df["distance_ft"] <= LINK_THRESHOLD_FT].copy()
    
    # Create mapping of T -> P
    t2p_map = {}
    linked_t_nodes = set()
    linked_p_nodes = set()
    
    for _, row in linked_df.iterrows():
        if row["edge_type"] == "transport_to_power":
            t_id = int(row["source"][2:])
            p_id = int(row["target"][2:])
            t2p_map[t_id] = p_id
            linked_t_nodes.add(t_id)
            linked_p_nodes.add(p_id)
        elif row["edge_type"] == "power_to_transport":
            p_id = int(row["source"][2:])
            t_id = int(row["target"][2:])
            if t_id not in t2p_map:
                t2p_map[t_id] = p_id
            linked_t_nodes.add(t_id)
            linked_p_nodes.add(p_id)

    # --- Monolayer Analysis (Linked Nodes Only) ---
    t_linked_asgn = {nid: comm for nid, comm in t_asgn.items() if nid in linked_t_nodes}
    p_linked_asgn = {nid: comm for nid, comm in p_asgn.items() if nid in linked_p_nodes}
    
    # Save monolayer subsets
    pd.DataFrame(list(t_linked_asgn.items()), columns=["node_id", "community_id"]).to_csv(
        OUT_MULTI / f"monolayer_linked_transport_{scenario_name}.csv", index=False
    )
    pd.DataFrame(list(p_linked_asgn.items()), columns=["node_id", "community_id"]).to_csv(
        OUT_MULTI / f"monolayer_linked_power_{scenario_name}.csv", index=False
    )

    # --- Multilayer Pairing ---
    rows = []
    for t_id in linked_t_nodes:
        if t_id in t_asgn:
            p_id = t2p_map.get(t_id)
            if p_id is not None and p_id in p_asgn:
                t_comm = t_asgn[t_id]
                p_comm = p_asgn[p_id]
                rows.append({
                    "transport_node": t_id,
                    "power_node": p_id,
                    "transport_community": t_comm,
                    "power_community": p_comm,
                    "pair": (t_comm, p_comm)
                })
    
    df_multi = pd.DataFrame(rows)
    unique_pairs = sorted(df_multi["pair"].unique())
    pair_to_id = {pair: i for i, pair in enumerate(unique_pairs)}
    df_multi["multilayer_community_id"] = df_multi["pair"].map(pair_to_id)
    
    summary = {
        "scenario": scenario_name,
        "linked_t_nodes": len(linked_t_nodes),
        "linked_p_nodes": len(linked_p_nodes),
        "unique_t_communities_linked": len(set(t_linked_asgn.values())),
        "unique_p_communities_linked": len(set(p_linked_asgn.values())),
        "unique_multilayer_pairs": len(unique_pairs),
        "t_linked_asgn": t_linked_asgn,
        "p_linked_asgn": p_linked_asgn
    }
    
    return df_multi, summary

def plot_monolayer_subset_map(asgn, G_lcc, network_type, scenario_label, save_path):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    lons, lats, comms = [], [], []
    for nid, comm_id in asgn.items():
        attr = G_lcc.nodes[nid]
        if network_type == "transport":
            lon, lat = _TO_WGS84.transform(attr["longitude"], attr["latitude"])
        else:
            lon, lat = attr["longitude"], attr["latitude"]
        lons.append(lon)
        lats.append(lat)
        comms.append(comm_id)
    
    num_communities = len(set(comms))
    # Map community IDs to 0..N-1 for color mapping consistency
    unique_comms = sorted(list(set(comms)))
    comm_to_idx = {c: i for i, c in enumerate(unique_comms)}
    indices = [comm_to_idx[c] for c in comms]
    
    cmap = plt.colormaps["hsv"].resampled(num_communities)
    
    ax.scatter(lons, lats, c=indices, cmap=cmap, s=1.5, alpha=0.6, linewidths=0, rasterized=True)
    
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (WGS84)", fontsize=9)
    ax.set_ylabel("Latitude (WGS84)", fontsize=9)
    ax.set_title(f"{network_type.title()} Communities (Linked Subset) — {scenario_label}\n"
                 f"{num_communities} communities", fontsize=12)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max(num_communities-1, 1)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Community Index", fontsize=10)
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.relative_to(ROOT)}")

def plot_multilayer_map(df, G_t, scenario_label, save_path, removed_nodes_df=None):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    lons, lats = [], []
    for _, row in df.iterrows():
        t_id = row["transport_node"]
        attr = G_t.nodes[t_id]
        lon, lat = _TO_WGS84.transform(attr["longitude"], attr["latitude"])
        lons.append(lon)
        lats.append(lat)
    
    df = df.copy()
    df["longitude"] = lons
    df["latitude"] = lats
    
    num_communities = df["multilayer_community_id"].nunique()
    cmap = plt.colormaps["hsv"].resampled(num_communities)
    
    for comm_id in sorted(df["multilayer_community_id"].unique()):
        subset = df[df["multilayer_community_id"] == comm_id]
        ax.scatter(subset["longitude"], subset["latitude"],
                   c=[cmap(comm_id % num_communities)],
                   s=1.5, alpha=0.6, linewidths=0, rasterized=True)
    
    # Plot removed nodes if applicable
    if removed_nodes_df is not None and len(removed_nodes_df) > 0:
        r_lons, r_lats = _TO_WGS84.transform(
            removed_nodes_df["longitude"].values,
            removed_nodes_df["latitude"].values,
        )
        ax.scatter(r_lons, r_lats, c="red", marker="x", s=100, linewidths=2, zorder=10, label="Removed BC Nodes")
        for i, row in removed_nodes_df.iterrows():
            ax.annotate(f"#{i+1}", xy=(r_lons[i], r_lats[i]), xytext=(5,5), textcoords="offset points", 
                        color="red", fontweight="bold", fontsize=8)

    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (WGS84)", fontsize=9)
    ax.set_ylabel("Latitude (WGS84)", fontsize=9)
    ax.set_title(f"Multilayer Communities (Linked Nodes) — {scenario_label}\n"
                 f"{num_communities} pairs within {LINK_THRESHOLD_FT}ft", fontsize=12)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max(num_communities-1, 1)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Multilayer Community Index", fontsize=10)
    
    if removed_nodes_df is not None:
        ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.relative_to(ROOT)}")

if __name__ == "__main__":
    print("=" * 65)
    print("Script 07 — Multilayer Community Detection (Linked Nodes Only)")
    print("=" * 65)

    G_t = load_pkl(OUT / "transport/lcc_graph.pkl")
    G_p = load_pkl(OUT / "power/lcc_graph.pkl")
    links_df = pd.read_csv(OUT_MULTI / "interlayer_edges.csv")

    scenarios = [
        ("baseline", "Baseline (No Attack)"),
        ("linked_bc", "Linked BC Attack")
    ]

    all_summaries = []

    for suffix, label in scenarios:
        print(f"\n[Scenario: {label}]")
        
        t_asgn = load_pkl(OUT / f"transport/community_assignments_{suffix}.pkl")
        p_asgn = load_pkl(OUT / f"power/community_assignments_{suffix}.pkl")
        
        df_multi, summary = get_linked_data(t_asgn, p_asgn, links_df, suffix)
        all_summaries.append(summary)
        
        # Plot Monolayer Subsets
        print(f"  [Plotting Monolayer Subsets...]")
        plot_monolayer_subset_map(summary['t_linked_asgn'], G_t, "transport", label, 
                                 FIG_MULTI / f"map_monolayer_linked_transport_{suffix}.png")
        plot_monolayer_subset_map(summary['p_linked_asgn'], G_p, "power", label, 
                                 FIG_MULTI / f"map_monolayer_linked_power_{suffix}.png")

        # Load removal order if it's the attack scenario
        removed_df = None
        if suffix == "linked_bc":
            removed_df = pd.read_csv(OUT_MULTI / "removal_order_linked_bc_transport.csv")

        # Save multilayer assignments
        df_multi.to_csv(OUT_MULTI / f"multilayer_assignments_{suffix}.csv", index=False)
        
        # Visualize Multilayer
        print(f"  [Plotting Multilayer...]")
        plot_multilayer_map(df_multi, G_t, label, FIG_MULTI / f"map_multilayer_communities_{suffix}.png", removed_df)

    # Final summary table
    df_sum = pd.DataFrame(all_summaries).drop(columns=['t_linked_asgn', 'p_linked_asgn'])
    df_sum.to_csv(OUT_MULTI / "multilayer_community_summary.csv", index=False)
    
    print("\n" + "="*65)
    print("MONOLAYER STATS FOR LINKED NODES ONLY")
    print("="*65)
    for s in all_summaries:
        print(f"\nScenario: {s['scenario']}")
        print(f"  Transport: {s['linked_t_nodes']} linked nodes -> {s['unique_t_communities_linked']} communities")
        print(f"  Power:     {s['linked_p_nodes']} linked nodes -> {s['unique_p_communities_linked']} communities")
        print(f"  JOINT:     {s['unique_multilayer_pairs']} unique pairs")

    print("\n" + "=" * 65)
    print("Done. Linked multilayer results saved to:")
    print("  output/multilayer/ (CSV reports)")
    print("  output/figures/multilayer/ (Maps with BC markers and Monolayer subsets)")
    print("=" * 65)
