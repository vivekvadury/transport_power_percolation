"""
script_02_percolation.py
========================
Core analysis loop: for each centrality strategy (BC and DC), incrementally
remove the top-k nodes (k=1..10) from the LCC and measure:
  - Connected components
  - LCC fraction
  - Louvain community structure + modularity

Also detects the tipping point: first k at which a new component appears.

Run AFTER script_01_build_graph.py:
    python scripts/script_02_percolation.py

Outputs (written to output/):
    percolation_bc.csv              - per-k metrics for BC strategy
    percolation_dc.csv              - per-k metrics for DC strategy
    community_assignments_bc.pkl    - {k: {node_id: community_index}}
    community_assignments_dc.pkl    - same for DC
    tipping_points.csv              - first k where components > 1 for each strategy
"""

import json
import pickle
import pathlib
import pandas as pd
import networkx as nx
import networkx.algorithms.community as nx_comm

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent.parent
OUT  = ROOT / "output"


def run_percolation_loop(G_lcc: nx.Graph,
                         removal_order: list[int],
                         strategy_name: str,
                         seed: int = 42) -> tuple[pd.DataFrame, dict, int | None]:
    """
    Iteratively remove the top-k nodes and measure community structure.

    The graph is mutated incrementally (one node removed per step), which
    means the state at step k reflects exactly k cumulative removals.

    Parameters
    ----------
    G_lcc          : the clean LCC graph (will be deep-copied internally)
    removal_order  : list of node IDs ranked 1st..10th for removal
    strategy_name  : 'bc' or 'dc' (used only for console labels)
    seed           : random seed for Louvain reproducibility

    Returns
    -------
    results_df          : pd.DataFrame with per-k metrics
    community_assignments : dict  {k (int): {node_id (int): community_index (int)}}
    tipping_point       : int k at which num_components first > 1, or None
    """
    N = G_lcc.number_of_nodes()
    G_attacked = G_lcc.copy()

    records = []
    community_assignments = {}
    tipping_point = None

    label = strategy_name.upper()

    for k, node_to_remove in enumerate(removal_order, start=1):
        # Incremental removal
        G_attacked.remove_node(node_to_remove)

        # Connected components
        num_components = nx.number_connected_components(G_attacked)
        lcc_nodes = max(nx.connected_components(G_attacked), key=len)
        lcc_size = len(lcc_nodes)
        lcc_fraction = lcc_size / N

        # Louvain community detection (handles disconnected graphs correctly)
        communities = nx_comm.louvain_communities(G_attacked, seed=seed)
        num_communities = len(communities)
        num_nontrivial = sum(1 for c in communities if len(c) >= 2)
        modularity = nx_comm.modularity(G_attacked, communities)

        # Community sizes (sorted descending)
        comm_sizes = sorted([len(c) for c in communities], reverse=True)

        # Store community assignment per node for geo visualization
        assignment = {}
        for comm_idx, comm in enumerate(communities):
            for node in comm:
                assignment[node] = comm_idx
        community_assignments[k] = assignment

        # Detect tipping point
        if tipping_point is None and num_components > 1:
            tipping_point = k

        records.append({
            "k":                       k,
            "removed_node_id":         node_to_remove,
            "num_components":          num_components,
            "lcc_size":                lcc_size,
            "lcc_fraction":            round(lcc_fraction, 6),
            "num_communities":         num_communities,
            "num_nontrivial_communities": num_nontrivial,
            "modularity":              round(modularity, 6),
            "community_sizes_json":    json.dumps(comm_sizes),
        })

        # Tipping point banner
        tip_marker = ""
        if num_components > 1 and (k == 1 or
                records[-2]["num_components"] == 1 if len(records) >= 2 else False):
            tip_marker = "  <<< TIPPING POINT"

        print(f"[{label} k={k:2d}] Removed node {node_to_remove:6d} | "
              f"Components: {num_components} | "
              f"LCC: {lcc_size} ({lcc_fraction:.4%}) | "
              f"Communities: {num_nontrivial} non-trivial | "
              f"Q={modularity:.4f}{tip_marker}")

    results_df = pd.DataFrame(records)
    return results_df, community_assignments, tipping_point


def compute_tipping_summary(results_df: pd.DataFrame,
                            strategy: str) -> dict:
    """
    Summarise the tipping point for one strategy.

    Parameters
    ----------
    results_df : output of run_percolation_loop
    strategy   : 'bc' or 'dc'

    Returns
    -------
    dict with keys: strategy, tipping_k, causal_node_id,
                    component_sizes_at_tipping_json
    """
    tipping_rows = results_df[results_df["num_components"] > 1]
    if tipping_rows.empty:
        return {
            "strategy":                      strategy,
            "tipping_k":                     None,
            "causal_node_id":                None,
            "component_sizes_at_tipping_json": json.dumps([]),
        }
    first = tipping_rows.iloc[0]
    return {
        "strategy":                      strategy,
        "tipping_k":                     int(first["k"]),
        "causal_node_id":                int(first["removed_node_id"]),
        "component_sizes_at_tipping_json": first["community_sizes_json"],
    }


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("Script 02 — Incremental Percolation + Community Detection")
    print("=" * 65)

    # Load LCC graph
    print("\n[1/5] Loading LCC graph from output/lcc_graph.pkl...")
    with open(OUT / "lcc_graph.pkl", "rb") as f:
        G_lcc = pickle.load(f)
    print(f"      {G_lcc.number_of_nodes()} nodes, {G_lcc.number_of_edges()} edges")

    # Load removal orders
    print("\n[2/5] Loading removal orders...")
    bc_order_df = pd.read_csv(OUT / "removal_order_bc.csv")
    dc_order_df = pd.read_csv(OUT / "removal_order_dc.csv")
    bc_order = bc_order_df["node_id"].astype(int).tolist()
    dc_order = dc_order_df["node_id"].astype(int).tolist()
    print(f"      BC order: {bc_order}")
    print(f"      DC order: {dc_order}")

    # ── BC strategy ────────────────────────────────────────────────────────────
    print("\n[3/5] Running BC (Betweenness Centrality) targeted removal...")
    print("-" * 65)
    bc_df, bc_assignments, bc_tip = run_percolation_loop(
        G_lcc, bc_order, strategy_name="bc", seed=42
    )

    if bc_tip is not None:
        print(f"\n      *** BC Tipping Point: k={bc_tip} ***")
        tip_row = bc_df[bc_df["k"] == bc_tip].iloc[0]
        sizes = json.loads(tip_row["community_sizes_json"])
        print(f"          Causal node: {int(tip_row['removed_node_id'])}")
        print(f"          Component sizes at k={bc_tip}: {sizes[:10]}")
    else:
        print("\n      No tipping point detected in k=1..10 for BC strategy.")

    # ── DC strategy ────────────────────────────────────────────────────────────
    print("\n[4/5] Running DC (Degree Centrality) targeted removal...")
    print("-" * 65)
    dc_df, dc_assignments, dc_tip = run_percolation_loop(
        G_lcc, dc_order, strategy_name="dc", seed=42
    )

    if dc_tip is not None:
        print(f"\n      *** DC Tipping Point: k={dc_tip} ***")
        tip_row = dc_df[dc_df["k"] == dc_tip].iloc[0]
        sizes = json.loads(tip_row["community_sizes_json"])
        print(f"          Causal node: {int(tip_row['removed_node_id'])}")
        print(f"          Component sizes at k={dc_tip}: {sizes[:10]}")
    else:
        print("\n      No tipping point detected in k=1..10 for DC strategy.")

    # ── Save all outputs ───────────────────────────────────────────────────────
    print("\n[5/5] Saving outputs...")

    bc_df.to_csv(OUT / "percolation_bc.csv", index=False)
    dc_df.to_csv(OUT / "percolation_dc.csv", index=False)
    print("      percolation_bc.csv  ✓")
    print("      percolation_dc.csv  ✓")

    with open(OUT / "community_assignments_bc.pkl", "wb") as f:
        pickle.dump(bc_assignments, f, protocol=4)
    with open(OUT / "community_assignments_dc.pkl", "wb") as f:
        pickle.dump(dc_assignments, f, protocol=4)
    print("      community_assignments_bc.pkl  ✓")
    print("      community_assignments_dc.pkl  ✓")

    tipping_records = [
        compute_tipping_summary(bc_df, "bc"),
        compute_tipping_summary(dc_df, "dc"),
    ]
    tipping_df = pd.DataFrame(tipping_records)
    tipping_df.to_csv(OUT / "tipping_points.csv", index=False)
    print("      tipping_points.csv  ✓")

    print("\n" + "=" * 65)
    print("Done.")
    print("=" * 65)
