"""
script_05_multilayer.py
=======================
Creates a multilayer interdependent network by coupling the transportation
and power networks.

Links nodes between the two layers using a KD-Tree nearest neighbor search
in a projected coordinate system (EPSG:2926 - WA State Plane North, US survey feet)
to ensure accurate geographic distances.

Run AFTER script_01_build_graph.py:
    python scripts/script_05_multilayer.py
"""

import pickle
import pathlib
import pandas as pd
import networkx as nx
import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree

ROOT = pathlib.Path(__file__).parent.parent
OUT  = ROOT / "output"
OUT_MULTI = OUT / "multilayer"
OUT_MULTI.mkdir(parents=True, exist_ok=True)

# ── 1. Load Graphs ────────────────────────────────────────────────────────────
def load_lcc_graph(network: str) -> nx.Graph:
    path = OUT / network / "lcc_graph.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)

# ── 2. Build Multilayer Graph ─────────────────────────────────────────────────
def build_multilayer_graph():
    print("=" * 65)
    print("Script 05 — Multilayer Network Construction")
    print("=" * 65)

    print("\n[1/4] Loading LCC graphs...")
    G_t = load_lcc_graph("transport")
    G_p = load_lcc_graph("power")

    print(f"      Transport: {G_t.number_of_nodes():,} nodes, {G_t.number_of_edges():,} edges")
    print(f"      Power:     {G_p.number_of_nodes():,} nodes, {G_p.number_of_edges():,} edges")

    # Create a new combined graph
    G_multi = nx.Graph()

    # Add Transport nodes, prefixed with 'T_' to avoid ID collisions
    # Transport coordinates are already in EPSG:2926 (WA State Plane North, feet)
    t_nodes = []
    t_coords = []
    for nid, data in G_t.nodes(data=True):
        multi_id = f"T_{nid}"
        G_multi.add_node(multi_id, layer="transport", original_id=nid, **data)
        t_nodes.append(multi_id)
        # In script 1 & 4, the transport "longitude" and "latitude" are actually X/Y in EPSG:2926
        t_coords.append((data["longitude"], data["latitude"]))
        
    for u, v, data in G_t.edges(data=True):
        G_multi.add_edge(f"T_{u}", f"T_{v}", layer="transport", **data)

    # Add Power nodes, prefixed with 'P_'
    # Power coordinates are WGS84 (lon/lat). We need to project them to EPSG:2926
    to_epsg2926 = Transformer.from_crs("EPSG:4326", "EPSG:2926", always_xy=True)
    
    p_nodes = []
    p_coords = []
    for nid, data in G_p.nodes(data=True):
        multi_id = f"P_{nid}"
        # data["longitude"] and data["latitude"] are WGS84 for power
        x_ft, y_ft = to_epsg2926.transform(data["longitude"], data["latitude"])
        
        G_multi.add_node(multi_id, layer="power", original_id=nid, 
                         x_ft=x_ft, y_ft=y_ft, **data)
        p_nodes.append(multi_id)
        p_coords.append((x_ft, y_ft))

    for u, v, data in G_p.edges(data=True):
        G_multi.add_edge(f"P_{u}", f"P_{v}", layer="power", **data)

    # ── 3. KD-Tree Nearest Neighbor Linking ───────────────────────────────────
    print("\n[2/4] Linking layers via KD-Tree (EPSG:2926 projected coords in feet)...")
    
    tree_t = cKDTree(np.array(t_coords))
    tree_p = cKDTree(np.array(p_coords))

    interlayer_edges = []
    
    # 1. Power -> nearest Transport
    print("      Matching each Power node to its nearest Transport node...")
    dists_p2t, indices_p2t = tree_t.query(np.array(p_coords), k=1)
    for i, p_multi_id in enumerate(p_nodes):
        nearest_t_multi_id = t_nodes[indices_p2t[i]]
        dist_ft = dists_p2t[i]
        
        if not G_multi.has_edge(p_multi_id, nearest_t_multi_id):
            G_multi.add_edge(p_multi_id, nearest_t_multi_id, 
                             layer="interlayer", weight=dist_ft, 
                             edge_type="power_to_transport")
            interlayer_edges.append({
                "source": p_multi_id,
                "target": nearest_t_multi_id,
                "distance_ft": dist_ft,
                "edge_type": "power_to_transport"
            })

    # 2. Transport -> nearest Power
    print("      Matching each Transport node to its nearest Power node...")
    dists_t2p, indices_t2p = tree_p.query(np.array(t_coords), k=1)
    for i, t_multi_id in enumerate(t_nodes):
        nearest_p_multi_id = p_nodes[indices_t2p[i]]
        dist_ft = dists_t2p[i]
        
        if not G_multi.has_edge(t_multi_id, nearest_p_multi_id):
            G_multi.add_edge(t_multi_id, nearest_p_multi_id, 
                             layer="interlayer", weight=dist_ft,
                             edge_type="transport_to_power")
            interlayer_edges.append({
                "source": t_multi_id,
                "target": nearest_p_multi_id,
                "distance_ft": dist_ft,
                "edge_type": "transport_to_power"
            })
            
    print(f"      Created {len(interlayer_edges):,} interdependent links.")
    print(f"      Multilayer Graph has {G_multi.number_of_nodes():,} nodes and {G_multi.number_of_edges():,} edges.")

    # ── 4. Save Outputs ───────────────────────────────────────────────────────
    print("\n[3/4] Saving multilayer graph...")
    with open(OUT_MULTI / "multilayer_graph.pkl", "wb") as f:
        pickle.dump(G_multi, f, protocol=4)

    print("\n[4/4] Saving interlayer edges CSV...")
    df_inter = pd.DataFrame(interlayer_edges)
    df_inter.to_csv(OUT_MULTI / "interlayer_edges.csv", index=False)
    
    # ── 5. Check for links between top BC nodes ────────────────────────────────
    print("\n[5/4] Checking for links between top BC nodes...")
    
    # Read BC nodes (these files contain the top 10 nodes for targeted attacks)
    t_bc_removal_order = pd.read_csv(OUT / "transport/removal_order_bc.csv")["node_id"].tolist()
    p_bc_removal_order = pd.read_csv(OUT / "power/removal_order_bc.csv")["node_id"].tolist()

    t_bc_set = set([f"T_{n}" for n in t_bc_removal_order])
    p_bc_set = set([f"P_{n}" for n in p_bc_removal_order])

    linked_bc_nodes = []
    # Use a set of frozensets to avoid duplicates when considering undirected pairs
    seen_links = set()

    for _, row in df_inter.iterrows():
        u, v = row["source"], row["target"]
        if (u in t_bc_set and v in p_bc_set) or (v in t_bc_set and u in p_bc_set):
            pair = frozenset([u, v])
            if pair not in seen_links:
                linked_bc_nodes.append((u, v, row["distance_ft"]))
                seen_links.add(pair)

    print(f"      Found {len(linked_bc_nodes)} direct link(s) between Transport top BC nodes and Power top BC nodes.")
    
    t_linked_nodes = []
    p_linked_nodes = []
    for u, v, dist in linked_bc_nodes:
        print(f"        {u} <-> {v} (Distance: {dist:.2f} feet)")
        if u.startswith("T_"):
            t_linked_nodes.append(int(u[2:]))
            p_linked_nodes.append(int(v[2:]))
        else:
            t_linked_nodes.append(int(v[2:]))
            p_linked_nodes.append(int(u[2:]))
            
    print("\n[6/4] Saving linked BC nodes to CSV...")
    # Load original removal orders to preserve full format
    full_t_removal_order = pd.read_csv(OUT / "transport/removal_order_bc.csv")
    full_p_removal_order = pd.read_csv(OUT / "power/removal_order_bc.csv")

    # Filter for the linked nodes and ensure they retain their rank from the original file
    df_t_linked_bc = full_t_removal_order[full_t_removal_order["node_id"].isin(t_linked_nodes)].copy()
    df_p_linked_bc = full_p_removal_order[full_p_removal_order["node_id"].isin(p_linked_nodes)].copy()
    
    # Save with similar naming convention
    df_t_linked_bc.to_csv(OUT_MULTI / "removal_order_linked_bc_transport.csv", index=False)
    df_p_linked_bc.to_csv(OUT_MULTI / "removal_order_linked_bc_power.csv", index=False)
    print("      Saved removal_order_linked_bc_transport.csv and removal_order_linked_bc_power.csv")
    
    print("\n" + "=" * 65)
    print("Done. Outputs written to:")
    print("  output/multilayer/")
    print("=" * 65)

if __name__ == "__main__":
    build_multilayer_graph()
