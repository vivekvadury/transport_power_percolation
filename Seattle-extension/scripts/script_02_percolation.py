"""
script_02_percolation.py
========================
Two focused analyses — no redundancy with the original Seattle-clean work:

  PART A — Tipping Point Scan
    Incrementally remove nodes one at a time (k=1..10) for each strategy.
    Only asks: at what k does the network first fracture into two pieces?
    Runs Louvain once, at that exact tipping moment, to capture the community
    structure at the first point of real isolation.

  PART B — Full-Attack Snapshot
    Remove all 10 top-centrality nodes at once for each strategy.
    Runs Louvain on the resulting graph.
    This is the "worst-case attack" result — the final community geography
    after the most damaging targeted removal possible within k=10.

Run AFTER script_01_build_graph.py:
    python scripts/script_02_percolation.py

Outputs (written to output/):
    tipping_points.csv                    - first fracture k, causal node, component sizes
    community_assignments_bc_tipping.pkl  - {node_id: community_index} at BC tipping k
    community_assignments_dc_tipping.pkl  - same for DC
    post_attack_bc.csv                    - community summary after removing all 10 BC nodes
    post_attack_dc.csv                    - same for DC
    community_assignments_bc_full.pkl     - {node_id: community_index} at k=10 for BC
    community_assignments_dc_full.pkl     - same for DC
"""

import json
import pickle
import pathlib
import pandas as pd
import networkx as nx
import networkx.algorithms.community as nx_comm

ROOT = pathlib.Path(__file__).parent.parent
OUT  = ROOT / "output"


# ══════════════════════════════════════════════════════════════════════════════
# PART A — Tipping Point Scan
# ══════════════════════════════════════════════════════════════════════════════

def find_tipping_point(G_lcc: nx.Graph,
                       removal_order: list[int],
                       strategy: str,
                       seed: int = 42) -> dict:
    """
    Incrementally remove nodes until the network first fractures.

    Only tracks connected-component count at each step — no Louvain in the
    loop, so this is fast. When the tipping point is found, runs Louvain once
    on that graph state to capture the community structure at the moment of
    first isolation.

    Parameters
    ----------
    G_lcc          : clean LCC graph (copied internally, not mutated)
    removal_order  : list of node IDs in ranked removal order
    strategy       : 'BC' or 'DC' (for console output only)
    seed           : Louvain seed for reproducibility

    Returns
    -------
    dict with keys:
        tipping_k               - int k at first fracture, or None
        causal_node_id          - node whose removal caused the split
        component_sizes         - list of component sizes at tipping k (sorted desc)
        community_assignment    - {node_id: community_index} at tipping k, or None
        scan_log                - list of (k, node_id, num_components) for all steps
    """
    G = G_lcc.copy()
    tipping_k = None
    causal_node = None
    tipping_assignment = None
    tipping_sizes = []
    scan_log = []

    print(f"\n  [{strategy}] Scanning for tipping point...")
    for k, node in enumerate(removal_order, start=1):
        G.remove_node(node)
        num_comp = nx.number_connected_components(G)
        scan_log.append((k, node, num_comp))
        print(f"    k={k:2d}  removed node {node:6d}  →  {num_comp} component(s)")

        if tipping_k is None and num_comp > 1:
            tipping_k = k
            causal_node = node
            # Sizes of all components at the tipping moment
            tipping_sizes = sorted(
                [len(c) for c in nx.connected_components(G)], reverse=True
            )
            # Louvain snapshot at tipping point
            communities = nx_comm.louvain_communities(G, seed=seed)
            tipping_assignment = {}
            for comm_idx, comm in enumerate(communities):
                for nid in comm:
                    tipping_assignment[nid] = comm_idx
            print(f"\n  *** [{strategy}] TIPPING POINT at k={k} ***")
            print(f"      Node {causal_node} caused the first fracture.")
            print(f"      Component sizes: {tipping_sizes}")
            print(f"      Louvain communities at tipping: "
                  f"{len(communities)} communities, "
                  f"Q={nx_comm.modularity(G, communities):.4f}\n")

    if tipping_k is None:
        print(f"  [{strategy}] No fracture in k=1..10 — network remained connected.")

    return {
        "tipping_k":            tipping_k,
        "causal_node_id":       causal_node,
        "component_sizes":      tipping_sizes,
        "community_assignment": tipping_assignment,
        "scan_log":             scan_log,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PART B — Full-Attack Snapshot
# ══════════════════════════════════════════════════════════════════════════════

def full_attack_snapshot(G_lcc: nx.Graph,
                         removal_order: list[int],
                         strategy: str,
                         seed: int = 42) -> dict:
    """
    Remove all 10 top-centrality nodes at once and analyse the result.

    This answers the core question: what does the community geography look like
    after the worst-case targeted attack within k=10?

    Parameters
    ----------
    G_lcc          : clean LCC graph (copied internally)
    removal_order  : list of 10 node IDs to remove
    strategy       : 'BC' or 'DC'
    seed           : Louvain seed

    Returns
    -------
    dict with keys:
        num_nodes_removed       - always 10 (or fewer if some not in LCC)
        actually_removed        - list of node IDs successfully removed
        num_components          - connected components after attack
        component_sizes         - sorted list of component sizes
        num_communities         - Louvain community count
        num_nontrivial_communities - communities with ≥ 2 nodes
        modularity              - Louvain Q score
        community_sizes         - sorted list of community sizes
        community_assignment    - {node_id: community_index}
    """
    G = G_lcc.copy()
    actually_removed = [n for n in removal_order if n in G]
    G.remove_nodes_from(actually_removed)

    components = list(nx.connected_components(G))
    component_sizes = sorted([len(c) for c in components], reverse=True)

    communities = nx_comm.louvain_communities(G, seed=seed)
    community_sizes = sorted([len(c) for c in communities], reverse=True)
    num_nontrivial = sum(1 for c in communities if len(c) >= 2)
    modularity = nx_comm.modularity(G, communities)

    assignment = {}
    for comm_idx, comm in enumerate(communities):
        for nid in comm:
            assignment[nid] = comm_idx

    print(f"\n  [{strategy}] Full-attack snapshot (k=10):")
    print(f"    Removed nodes: {actually_removed}")
    print(f"    Connected components: {len(components)}")
    print(f"    Largest component:    {component_sizes[0]:,} nodes "
          f"({component_sizes[0]/G_lcc.number_of_nodes():.2%} of LCC)")
    if len(component_sizes) > 1:
        print(f"    Severed components:   {component_sizes[1:]} nodes")
    print(f"    Louvain communities:  {len(communities)} total, "
          f"{num_nontrivial} non-trivial")
    print(f"    Modularity (Q):       {modularity:.4f}")

    return {
        "num_nodes_removed":          len(actually_removed),
        "actually_removed":           actually_removed,
        "num_components":             len(components),
        "component_sizes":            component_sizes,
        "num_communities":            len(communities),
        "num_nontrivial_communities": num_nontrivial,
        "modularity":                 modularity,
        "community_sizes":            community_sizes,
        "community_assignment":       assignment,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("Script 02 — Tipping Point Scan + Full-Attack Community Analysis")
    print("=" * 65)

    # Load inputs
    print("\n[1/4] Loading LCC graph and removal orders...")
    with open(OUT / "lcc_graph.pkl", "rb") as f:
        G_lcc = pickle.load(f)
    bc_order = pd.read_csv(OUT / "removal_order_bc.csv")["node_id"].astype(int).tolist()
    dc_order = pd.read_csv(OUT / "removal_order_dc.csv")["node_id"].astype(int).tolist()
    print(f"      LCC: {G_lcc.number_of_nodes():,} nodes, {G_lcc.number_of_edges():,} edges")
    print(f"      BC removal order: {bc_order}")
    print(f"      DC removal order: {dc_order}")

    # ── PART A: Tipping Point Scans ───────────────────────────────────────────
    print("\n" + "─" * 65)
    print("[2/4] PART A — Tipping Point Scan")
    print("─" * 65)

    bc_tip = find_tipping_point(G_lcc, bc_order, strategy="BC", seed=42)
    dc_tip = find_tipping_point(G_lcc, dc_order, strategy="DC", seed=42)

    # Save tipping point summary
    tipping_rows = []
    for strategy, result in [("BC", bc_tip), ("DC", dc_tip)]:
        tipping_rows.append({
            "strategy":                 strategy,
            "tipping_k":               result["tipping_k"],
            "causal_node_id":          result["causal_node_id"],
            "component_sizes_json":    json.dumps(result["component_sizes"]),
        })
    tipping_df = pd.DataFrame(tipping_rows)
    tipping_df.to_csv(OUT / "tipping_points.csv", index=False)

    # Save Louvain assignments at tipping point
    for strategy, result in [("bc", bc_tip), ("dc", dc_tip)]:
        fname = OUT / f"community_assignments_{strategy}_tipping.pkl"
        with open(fname, "wb") as f:
            pickle.dump(result["community_assignment"], f, protocol=4)

    # ── PART B: Full-Attack Snapshots ─────────────────────────────────────────
    print("\n" + "─" * 65)
    print("[3/4] PART B — Full-Attack Snapshot (remove all 10 nodes)")
    print("─" * 65)

    bc_full = full_attack_snapshot(G_lcc, bc_order, strategy="BC", seed=42)
    dc_full = full_attack_snapshot(G_lcc, dc_order, strategy="DC", seed=42)

    # Save post-attack summaries as CSV
    for strategy, result in [("bc", bc_full), ("dc", dc_full)]:
        row = {k: v for k, v in result.items()
               if k not in ("community_assignment",)}
        row["component_sizes_json"] = json.dumps(row.pop("component_sizes"))
        row["community_sizes_json"] = json.dumps(row.pop("community_sizes"))
        row["actually_removed"]     = json.dumps(row["actually_removed"])
        pd.DataFrame([row]).to_csv(OUT / f"post_attack_{strategy}.csv", index=False)

    # Save community assignments at k=10
    for strategy, result in [("bc", bc_full), ("dc", dc_full)]:
        fname = OUT / f"community_assignments_{strategy}_full.pkl"
        with open(fname, "wb") as f:
            pickle.dump(result["community_assignment"], f, protocol=4)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("[4/4] All outputs saved:")
    for f in sorted(OUT.glob("*.csv")) + sorted(OUT.glob("*.pkl")):
        print(f"      {f.name}")

    print("\n" + "=" * 65)
    print("Done.")
    print("=" * 65)
