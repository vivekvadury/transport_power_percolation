"""
script_01_build_graph.py
========================
Build LCC NetworkX graphs for BOTH networks (transportation and power)
from raw CSVs and persist each as a pickle.
Also saves the authoritative top-10 removal order for BC and DC strategies,
and computes articulation points (cut vertices) for each network.

Run from the Seattle-extension/ directory:
    python scripts/script_01_build_graph.py

Outputs (written to output/transport/ and output/power/):
    lcc_graph.pkl               - pickled nx.Graph, integer node IDs, all attrs
    removal_order_bc.csv        - rank, node_id, betweenness_centrality, longitude, latitude
    removal_order_dc.csv        - rank, node_id, degree_centrality, longitude, latitude
    articulation_points.csv     - node_id, longitude, latitude for all cut vertices
"""

import pickle
import pathlib
import pandas as pd
import networkx as nx

ROOT = pathlib.Path(__file__).parent.parent
OUT  = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)

# ── Network configuration ──────────────────────────────────────────────────────
NETWORKS = [
    {
        "name":        "transport",
        "data_dir":    ROOT / "data" / "transportion",
        "out_dir":     OUT / "transport",
        "coord_label": "EPSG:2926 — WA State Plane North (ft)",
    },
    {
        "name":        "power",
        "data_dir":    ROOT / "data" / "power",
        "out_dir":     OUT / "power",
        "coord_label": "WGS84 Longitude / Latitude (decimal degrees)",
    },
]


# ── Reusable functions (network-agnostic) ──────────────────────────────────────

def build_graph_from_csvs(node_csv_path: pathlib.Path,
                          edge_csv_path: pathlib.Path) -> nx.Graph:
    """
    Build a nx.Graph from pre-computed node and edge CSVs.

    Node attributes attached: longitude, latitude, degree_centrality,
    betweenness_centrality.  Node IDs are kept as Python int.
    """
    nodes_df = pd.read_csv(node_csv_path)
    edges_df = pd.read_csv(edge_csv_path)

    G = nx.Graph()
    for _, row in nodes_df.iterrows():
        node_id = int(row["Node ID"])
        G.add_node(node_id,
                   longitude=float(row["Longitude"]),
                   latitude=float(row["Latitude"]),
                   degree_centrality=float(row["Degree Centrality"]),
                   betweenness_centrality=float(row["Betweenness Centrality"]))

    for _, row in edges_df.iterrows():
        G.add_edge(int(row["Source"]), int(row["Target"]),
                   weight=float(row["Weight"]))
    return G


def extract_lcc(G: nx.Graph) -> nx.Graph:
    """Extract the Largest Connected Component as a standalone mutable graph."""
    largest_cc = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_cc).copy()


def load_removal_order(central_csv_path: pathlib.Path,
                       metric_col: str,
                       rank_col: str) -> pd.DataFrame:
    """
    Load the authoritative top-10 removal order from a central_nodes_*.csv.
    The CSV is already sorted descending by the centrality metric.
    """
    df = pd.read_csv(central_csv_path)
    df["Node ID"] = df["Node ID"].astype(int)
    return pd.DataFrame({
        "rank":      range(1, len(df) + 1),
        "node_id":   df["Node ID"].values,
        rank_col:    df[metric_col].values,
        "longitude": df["Longitude"].values,
        "latitude":  df["Latitude"].values,
    })


# ── Per-network processing ─────────────────────────────────────────────────────

def process_network(cfg: dict):
    name     = cfg["name"]
    data_dir = cfg["data_dir"]
    out_dir  = cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Network: {name.upper()}")
    print(f"{'=' * 60}")

    # [1/5] Build graph
    print("\n[1/5] Reading node and edge CSVs...")
    G = build_graph_from_csvs(data_dir / "node_list_dc_btwn.csv",
                              data_dir / "edge_list.csv")
    print(f"      Full graph: {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges, "
          f"{nx.number_connected_components(G)} component(s)")

    # [2/5] Extract LCC
    print("\n[2/5] Extracting Largest Connected Component...")
    G_lcc    = extract_lcc(G)
    isolated = G.number_of_nodes() - G_lcc.number_of_nodes()
    print(f"      LCC: {G_lcc.number_of_nodes():,} nodes, "
          f"{G_lcc.number_of_edges():,} edges, "
          f"{nx.number_connected_components(G_lcc)} component(s)")
    print(f"      Nodes outside LCC (dropped): {isolated}")

    # [3/5] Save LCC pickle
    print("\n[3/5] Saving LCC graph...")
    with open(out_dir / "lcc_graph.pkl", "wb") as f:
        pickle.dump(G_lcc, f, protocol=4)
    print("      Saved.")

    # [4/5] Removal orders
    print("\n[4/5] Saving removal order CSVs...")
    bc_order = load_removal_order(
        data_dir / "central_nodes_btwn.csv",
        metric_col="Betweenness Centrality",
        rank_col="betweenness_centrality"
    )
    bc_order.to_csv(out_dir / "removal_order_bc.csv", index=False)
    print(f"      BC removal order (top {len(bc_order)}):")
    for _, r in bc_order.iterrows():
        print(f"        rank {int(r['rank']):2d}: node {int(r['node_id']):6d}  "
              f"BC={r['betweenness_centrality']:.0f}")

    dc_order = load_removal_order(
        data_dir / "central_nodes_dc.csv",
        metric_col="Degree Centrality",
        rank_col="degree_centrality"
    )
    dc_order.to_csv(out_dir / "removal_order_dc.csv", index=False)
    print(f"\n      DC removal order (top {len(dc_order)}):")
    for _, r in dc_order.iterrows():
        print(f"        rank {int(r['rank']):2d}: node {int(r['node_id']):6d}  "
              f"DC={r['degree_centrality']:.1f}")

    # [5/5] Articulation points
    print("\n[5/5] Computing articulation points (cut vertices)...")
    ap_set = set(nx.articulation_points(G_lcc))
    print(f"      {len(ap_set):,} articulation points "
          f"({len(ap_set) / G_lcc.number_of_nodes():.2%} of LCC nodes)")

    ap_rows = [{"node_id":   nid,
                "longitude": G_lcc.nodes[nid]["longitude"],
                "latitude":  G_lcc.nodes[nid]["latitude"]}
               for nid in ap_set]
    ap_df = pd.DataFrame(ap_rows).sort_values("node_id").reset_index(drop=True)
    ap_df.to_csv(out_dir / "articulation_points.csv", index=False)
    print(f"      Saved articulation_points.csv ({len(ap_df):,} rows)")

    def ap_check(order_df: pd.DataFrame, label: str):
        print(f"\n      {label} removal order — cut vertex check:")
        for _, r in order_df.iterrows():
            nid = int(r["node_id"])
            tag = "[CUT VERTEX]" if nid in ap_set else "[not AP]    "
            print(f"        rank {int(r['rank']):2d}: node {nid:6d}  {tag}")

    ap_check(bc_order, "BC")
    ap_check(dc_order, "DC")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Script 01 — Build Graph (Transport + Power)")
    print("=" * 60)

    for cfg in NETWORKS:
        process_network(cfg)

    print(f"\n{'=' * 60}")
    print("Done. Outputs written to:")
    for cfg in NETWORKS:
        print(f"  output/{cfg['name']}/")
    print("=" * 60)
