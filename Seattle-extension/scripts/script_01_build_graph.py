"""
script_01_build_graph.py
========================
Build the LCC NetworkX graph from raw CSVs and persist it as a pickle.
Also saves the authoritative top-10 removal order for BC and DC strategies.

Run from the Seattle-extension/ directory:
    python scripts/script_01_build_graph.py

Outputs (written to output/):
    lcc_graph.pkl               - pickled nx.Graph, integer node IDs, all attrs
    removal_order_bc.csv        - rank, node_id, betweenness_centrality, longitude, latitude
    removal_order_dc.csv        - rank, node_id, degree_centrality, longitude, latitude
"""

import pickle
import pathlib
import pandas as pd
import networkx as nx

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent.parent   # Seattle-extension/
DATA = ROOT / "data"
OUT  = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)


def build_graph_from_csvs(node_csv_path: pathlib.Path,
                          edge_csv_path: pathlib.Path) -> nx.Graph:
    """
    Build a nx.Graph from pre-computed node and edge CSVs.

    Node attributes attached: Longitude, Latitude, Degree Centrality,
    Betweenness Centrality.  Node IDs are kept as Python int.

    Parameters
    ----------
    node_csv_path : path to node_list_dc_btwn.csv
    edge_csv_path : path to edge_list.csv

    Returns
    -------
    nx.Graph with integer node IDs and full attributes
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
    """
    Extract the Largest Connected Component as a standalone mutable graph.

    Parameters
    ----------
    G : full nx.Graph

    Returns
    -------
    G_lcc : nx.Graph (copy of the LCC subgraph)
    """
    largest_cc = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_cc).copy()


def load_removal_order(central_csv_path: pathlib.Path,
                       metric_col: str,
                       rank_col: str) -> pd.DataFrame:
    """
    Load the authoritative top-10 removal order from a central_nodes_*.csv.

    The CSV is already sorted descending by the centrality metric.
    Node IDs are cast to int for consistency with the graph.

    Parameters
    ----------
    central_csv_path : path to central_nodes_btwn.csv or central_nodes_dc.csv
    metric_col       : name of the centrality column in that CSV
    rank_col         : name for the metric column in the output DataFrame

    Returns
    -------
    pd.DataFrame with columns: rank, node_id, <rank_col>, longitude, latitude
    """
    df = pd.read_csv(central_csv_path)
    df["Node ID"] = df["Node ID"].astype(int)
    result = pd.DataFrame({
        "rank":      range(1, len(df) + 1),
        "node_id":   df["Node ID"].values,
        rank_col:    df[metric_col].values,
        "longitude": df["Longitude"].values,
        "latitude":  df["Latitude"].values,
    })
    return result


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Script 01 — Build Graph")
    print("=" * 60)

    # Build full graph
    print("\n[1/4] Reading node and edge CSVs...")
    G = build_graph_from_csvs(DATA / "node_list_dc_btwn.csv",
                               DATA / "edge_list.csv")
    print(f"      Full graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges, "
          f"{nx.number_connected_components(G)} component(s)")

    # Extract LCC
    print("\n[2/4] Extracting Largest Connected Component...")
    G_lcc = extract_lcc(G)
    print(f"      LCC: {G_lcc.number_of_nodes()} nodes, "
          f"{G_lcc.number_of_edges()} edges, "
          f"{nx.number_connected_components(G_lcc)} component(s)")

    isolated = G.number_of_nodes() - G_lcc.number_of_nodes()
    print(f"      Nodes outside LCC (dropped): {isolated}")

    # Save LCC pickle
    print("\n[3/4] Saving LCC graph to output/lcc_graph.pkl...")
    with open(OUT / "lcc_graph.pkl", "wb") as f:
        pickle.dump(G_lcc, f, protocol=4)
    print("      Saved.")

    # Load and save removal orders
    print("\n[4/4] Saving removal order CSVs...")

    bc_order = load_removal_order(
        DATA / "central_nodes_btwn.csv",
        metric_col="Betweenness Centrality",
        rank_col="betweenness_centrality"
    )
    bc_order.to_csv(OUT / "removal_order_bc.csv", index=False)
    print(f"      BC removal order (top {len(bc_order)}):")
    for _, r in bc_order.iterrows():
        print(f"        rank {int(r['rank'])}: node {int(r['node_id'])}  "
              f"BC={r['betweenness_centrality']:.0f}")

    dc_order = load_removal_order(
        DATA / "central_nodes_dc.csv",
        metric_col="Degree Centrality",
        rank_col="degree_centrality"
    )
    dc_order.to_csv(OUT / "removal_order_dc.csv", index=False)
    print(f"\n      DC removal order (top {len(dc_order)}):")
    for _, r in dc_order.iterrows():
        print(f"        rank {int(r['rank'])}: node {int(r['node_id'])}  "
              f"DC={r['degree_centrality']:.1f}")

    print("\n" + "=" * 60)
    print("Done. Outputs written to output/")
    print("  output/lcc_graph.pkl")
    print("  output/removal_order_bc.csv")
    print("  output/removal_order_dc.csv")
    print("=" * 60)
