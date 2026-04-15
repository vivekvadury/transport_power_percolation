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
import random
import pathlib
import numpy as np
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
# PART C — Louvain Stability Check
# ══════════════════════════════════════════════════════════════════════════════

def louvain_stability_check(G_attacked: nx.Graph,
                            strategy: str,
                            seeds: range = range(10)) -> dict:
    """
    Run Louvain 10 times with different seeds on a post-attack graph.

    Quantifies how sensitive the community partition is to the algorithm's
    stochastic element. Low std relative to mean indicates a robust partition
    that does not depend heavily on the random seed.

    Parameters
    ----------
    G_attacked : post-attack graph (already has nodes removed)
    strategy   : 'BC' or 'DC' (for console output only)
    seeds      : iterable of integer seeds to try

    Returns
    -------
    dict with keys:
        strategy, mean_communities, std_communities,
        min_communities, max_communities,
        mean_modularity, std_modularity
    """
    counts       = []
    modularities = []

    print(f"\n  [{strategy}] Running Louvain stability check (seeds 0-9)...")
    for seed in seeds:
        communities = nx_comm.louvain_communities(G_attacked, seed=seed)
        counts.append(len(communities))
        modularities.append(nx_comm.modularity(G_attacked, communities))

    result = {
        "strategy":         strategy,
        "mean_communities": float(np.mean(counts)),
        "std_communities":  float(np.std(counts)),
        "min_communities":  int(min(counts)),
        "max_communities":  int(max(counts)),
        "mean_modularity":  float(np.mean(modularities)),
        "std_modularity":   float(np.std(modularities)),
    }

    print(f"    Communities: {result['mean_communities']:.1f} +/- {result['std_communities']:.2f}  "
          f"(range {result['min_communities']}–{result['max_communities']})")
    print(f"    Modularity:  {result['mean_modularity']:.4f} +/- {result['std_modularity']:.4f}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# PART D — Sampled Average Path Length (BC attack only)
# ══════════════════════════════════════════════════════════════════════════════

def sampled_avg_path_length(G: nx.Graph,
                            sample_size: int = 500,
                            seed: int = 42) -> float:
    """
    Estimate average shortest-path length by BFS from a random sample of nodes.

    Only valid on a connected graph — pass the LCC or a single component.
    Uses sampling because full APSP on 21k nodes is prohibitively slow.

    Parameters
    ----------
    G           : connected graph to measure
    sample_size : number of source nodes to sample (more = more accurate)
    seed        : random seed for source selection

    Returns
    -------
    Estimated average shortest-path length (float)
    """
    rng     = random.Random(seed)
    nodes   = list(G.nodes())
    sources = rng.sample(nodes, min(sample_size, len(nodes)))

    total = 0
    count = 0
    for source in sources:
        lengths = nx.single_source_shortest_path_length(G, source)
        for target, length in lengths.items():
            if target != source:
                total += length
                count += 1

    return total / count if count > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("Script 02 — Tipping Point Scan + Full-Attack Community Analysis")
    print("=" * 65)

    # Load inputs
    print("\n[1/6] Loading LCC graph and removal orders...")
    with open(OUT / "lcc_graph.pkl", "rb") as f:
        G_lcc = pickle.load(f)
    bc_order = pd.read_csv(OUT / "removal_order_bc.csv")["node_id"].astype(int).tolist()
    dc_order = pd.read_csv(OUT / "removal_order_dc.csv")["node_id"].astype(int).tolist()
    print(f"      LCC: {G_lcc.number_of_nodes():,} nodes, {G_lcc.number_of_edges():,} edges")
    print(f"      BC removal order: {bc_order}")
    print(f"      DC removal order: {dc_order}")

    # ── PART A: Tipping Point Scans ───────────────────────────────────────────
    print("\n" + "─" * 65)
    print("[2/6] PART A — Tipping Point Scan")
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
    print("[3/6] PART B — Full-Attack Snapshot (remove all 10 nodes)")
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

    # ── PART C: Louvain Stability Check ──────────────────────────────────────
    print("\n" + "─" * 65)
    print("[4/6] PART C — Louvain Stability Check (10 seeds each strategy)")
    print("─" * 65)

    # Re-build post-attack graphs for stability check
    G_bc_attacked = G_lcc.copy()
    G_bc_attacked.remove_nodes_from([n for n in bc_order if n in G_bc_attacked])

    G_dc_attacked = G_lcc.copy()
    G_dc_attacked.remove_nodes_from([n for n in dc_order if n in G_dc_attacked])

    bc_stability = louvain_stability_check(G_bc_attacked, strategy="BC")
    dc_stability = louvain_stability_check(G_dc_attacked, strategy="DC")

    stability_df = pd.DataFrame([bc_stability, dc_stability])
    stability_df.to_csv(OUT / "louvain_stability.csv", index=False)
    print(f"\n  Saved: louvain_stability.csv")

    # ── PART D: Path Length Analysis (BC only — graph stays connected) ────────
    print("\n" + "─" * 65)
    print("[5/6] PART D — Average Path Length  (BC attack; 500-node sample)")
    print("─" * 65)
    print("  Note: this may take 2-4 minutes depending on hardware.")

    print("\n  Computing baseline average path length (G_lcc)...")
    baseline_apl = sampled_avg_path_length(G_lcc, sample_size=500, seed=42)
    print(f"    Baseline APL:        {baseline_apl:.4f}")

    print("  Computing post-BC-attack APL (10 nodes removed)...")
    bc_apl     = sampled_avg_path_length(G_bc_attacked, sample_size=500, seed=42)
    pct_change = (bc_apl - baseline_apl) / baseline_apl * 100
    print(f"    Post-BC APL:         {bc_apl:.4f}")
    print(f"    Change:              {pct_change:+.2f}%")

    if pct_change > 0:
        print(f"\n  FINDING: BC attack increased average travel distance by "
              f"{pct_change:.1f}% without severing the network.")
    else:
        print(f"\n  FINDING: BC attack had minimal effect on average path length.")

    # Skip DC: graph is disconnected at k=10 — APL on full graph is undefined.
    # The DC finding (fracture at k=3) already tells the efficiency story.
    print("\n  [DC skipped — graph is disconnected at k=10; "
          "APL is undefined across components]")

    apl_df = pd.DataFrame([{
        "graph":              "baseline",
        "strategy":           "—",
        "avg_path_length":    baseline_apl,
        "pct_change_vs_base": 0.0,
        "sample_size":        500,
    }, {
        "graph":              "post_attack",
        "strategy":           "BC",
        "avg_path_length":    bc_apl,
        "pct_change_vs_base": pct_change,
        "sample_size":        500,
    }])
    apl_df.to_csv(OUT / "path_length_analysis.csv", index=False)
    print("  Saved: path_length_analysis.csv")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("[6/6] All outputs saved:")
    for f in sorted(OUT.glob("*.csv")) + sorted(OUT.glob("*.pkl")):
        print(f"      {f.name}")

    print("\n" + "=" * 65)
    print("Done.")
    print("=" * 65)
