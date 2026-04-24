"""
script_02_percolation.py
========================
Runs four analyses for BOTH networks (transportation and power):

  PART A — Tipping Point Scan
    Incrementally remove nodes (k=1..10) per strategy. Find the first k
    where the network fractures. Run Louvain once at that exact moment.

  PART B — Full-Attack Snapshot
    Remove all 10 top-centrality nodes at once. Run Louvain on the result.

  PART C — Louvain Stability Check
    Re-run Louvain with seeds 0–9 on the post-attack graph to quantify
    sensitivity to the stochastic element.

  PART D — Average Path Length (BC only)
    Estimate average shortest-path length before and after BC k=10 attack
    using a 500-node random sample. BC is the only valid case because the
    graph stays connected; DC fractures the graph, making APL undefined.

Run AFTER script_01_build_graph.py:
    python scripts/script_02_percolation.py

Outputs (written to output/transport/ and output/power/):
    tipping_points.csv
    community_assignments_{bc/dc}_tipping.pkl
    post_attack_{bc/dc}.csv
    community_assignments_{bc/dc}_full.pkl
    louvain_stability.csv
    path_length_analysis.csv
    baseline_community_summary.csv
    community_assignments_baseline.csv/.pkl
    post_attack_linked_bc.csv
    community_assignments_linked_bc.csv/.pkl
    output/multilayer/linked_bc_community_comparison.csv
"""

import json
import pickle
import random
import pathlib
import sys
import numpy as np
import pandas as pd
import networkx as nx
import networkx.algorithms.community as nx_comm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = pathlib.Path(__file__).parent.parent
DATA = ROOT / "data"
OUT  = ROOT / "output"
OUT_MULTILAYER = OUT / "multilayer"

NETWORKS = [
    {"name": "transport", "out_dir": OUT / "transport"},
    {"name": "power",     "out_dir": OUT / "power"},
]


def assignment_from_communities(communities: list[set[int]]) -> dict:
    """Convert Louvain community sets to a node_id -> community_id mapping."""
    return {nid: idx for idx, comm in enumerate(communities) for nid in comm}


def community_assignment_df(G: nx.Graph, assignment: dict) -> pd.DataFrame:
    """Build a tabular community assignment export with node coordinates."""
    rows = []
    for node_id, community_id in assignment.items():
        attr = G.nodes[node_id]
        rows.append({
            "node_id": node_id,
            "community_id": community_id,
            "longitude": attr.get("longitude"),
            "latitude": attr.get("latitude"),
        })
    return pd.DataFrame(rows).sort_values(["community_id", "node_id"]).reset_index(drop=True)


def save_community_assignment(G: nx.Graph,
                              assignment: dict,
                              out_dir: pathlib.Path,
                              stem: str):
    """Save community assignments in both pickle and CSV formats."""
    with open(out_dir / f"{stem}.pkl", "wb") as f:
        pickle.dump(assignment, f, protocol=4)
    community_assignment_df(G, assignment).to_csv(out_dir / f"{stem}.csv", index=False)


def community_summary_row(network: str,
                          scenario: str,
                          G: nx.Graph,
                          communities: list[set[int]],
                          modularity: float,
                          removed_nodes=None) -> dict:
    """Create one summary row for baseline or post-removal Louvain results."""
    component_sizes = sorted([len(c) for c in nx.connected_components(G)], reverse=True)
    community_sizes = sorted([len(c) for c in communities], reverse=True)
    removed_nodes = removed_nodes or []
    return {
        "network": network,
        "scenario": scenario,
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "num_nodes_removed": len(removed_nodes),
        "removed_node_ids_json": json.dumps(removed_nodes),
        "num_components": nx.number_connected_components(G),
        "component_sizes_json": json.dumps(component_sizes),
        "num_communities": len(communities),
        "num_nontrivial_communities": sum(1 for c in communities if len(c) >= 2),
        "community_sizes_json": json.dumps(community_sizes),
        "modularity": modularity,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PART A — Tipping Point Scan
# ══════════════════════════════════════════════════════════════════════════════

def find_tipping_point(G_lcc: nx.Graph,
                       removal_order: list[int],
                       strategy: str,
                       seed: int = 42) -> dict:
    """
    Incrementally remove nodes until the network first fractures.

    Only checks connected-component count per step (no Louvain in the loop).
    When the tipping point is found, runs Louvain once at that exact state.
    """
    G          = G_lcc.copy()
    tipping_k  = None
    causal_node = None
    tipping_assignment = None
    tipping_sizes = []
    scan_log  = []

    print(f"\n  [{strategy}] Scanning for tipping point...")
    for k, node in enumerate(removal_order, start=1):
        if node not in G:
            print(f"    k={k:2d}  node {node:6d} not in LCC — skipping")
            continue
        G.remove_node(node)
        num_comp = nx.number_connected_components(G)
        scan_log.append((k, node, num_comp))
        print(f"    k={k:2d}  removed node {node:6d}  →  {num_comp} component(s)")

        if tipping_k is None and num_comp > 1:
            tipping_k   = k
            causal_node = node
            tipping_sizes = sorted(
                [len(c) for c in nx.connected_components(G)], reverse=True
            )
            communities = nx_comm.louvain_communities(G, seed=seed)
            tipping_assignment = assignment_from_communities(communities)
            print(f"\n  *** [{strategy}] TIPPING POINT at k={k} ***")
            print(f"      Node {causal_node} caused the first fracture.")
            print(f"      Component sizes: {tipping_sizes}")
            print(f"      Louvain communities at tipping: "
                  f"{len(communities)}, "
                  f"Q={nx_comm.modularity(G, communities):.4f}\n")

    if tipping_k is None:
        print(f"  [{strategy}] No fracture in k=1..10 — network stayed connected.")

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
    Remove the requested centrality nodes at once and analyse the result.
    """
    G                = G_lcc.copy()
    actually_removed = [n for n in removal_order if n in G]
    G.remove_nodes_from(actually_removed)

    components      = list(nx.connected_components(G))
    component_sizes = sorted([len(c) for c in components], reverse=True)

    communities     = nx_comm.louvain_communities(G, seed=seed)
    community_sizes = sorted([len(c) for c in communities], reverse=True)
    num_nontrivial  = sum(1 for c in communities if len(c) >= 2)
    modularity      = nx_comm.modularity(G, communities)

    assignment = assignment_from_communities(communities)

    print(f"\n  [{strategy}] Full-attack snapshot (k={len(removal_order)}):")
    print(f"    Removed:              {actually_removed}")
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
    Run Louvain 10 times with different seeds. Reports mean ± std of
    community count and modularity to quantify stochastic sensitivity.
    """
    counts       = []
    modularities = []

    print(f"\n  [{strategy}] Louvain stability check (seeds 0–9)...")
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
    print(f"    Communities: {result['mean_communities']:.1f} +/- "
          f"{result['std_communities']:.2f}  "
          f"(range {result['min_communities']}–{result['max_communities']})")
    print(f"    Modularity:  {result['mean_modularity']:.4f} +/- "
          f"{result['std_modularity']:.4f}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PART D — Sampled Average Path Length
# ══════════════════════════════════════════════════════════════════════════════

def sampled_avg_path_length(G: nx.Graph,
                            sample_size: int = 500,
                            seed: int = 42) -> float:
    """
    Estimate average shortest-path length via BFS from a random sample
    of source nodes. Only valid on a connected graph.
    """
    rng     = random.Random(seed)
    nodes   = list(G.nodes())
    sources = rng.sample(nodes, min(sample_size, len(nodes)))

    total = 0
    count = 0
    for source in sources:
        for target, length in nx.single_source_shortest_path_length(G, source).items():
            if target != source:
                total += length
                count += 1
    return total / count if count > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Per-network runner
# ══════════════════════════════════════════════════════════════════════════════

def process_network(cfg: dict):
    name    = cfg["name"]
    out_dir = cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 65}")
    print(f"  Network: {name.upper()}")
    print(f"{'=' * 65}")

    # Load
    print("\n[1/6] Loading LCC graph and removal orders...")
    with open(out_dir / "lcc_graph.pkl", "rb") as f:
        G_lcc = pickle.load(f)
    bc_order = pd.read_csv(out_dir / "removal_order_bc.csv")["node_id"].astype(int).tolist()
    dc_order = pd.read_csv(out_dir / "removal_order_dc.csv")["node_id"].astype(int).tolist()
    print(f"      LCC: {G_lcc.number_of_nodes():,} nodes, {G_lcc.number_of_edges():,} edges")
    print(f"      BC removal order: {bc_order}")
    print(f"      DC removal order: {dc_order}")

    # Baseline modularity (pre-attack)
    print("\n[1b/6] Computing baseline modularity (pre-attack G_lcc)...")
    baseline_communities = nx_comm.louvain_communities(G_lcc, seed=42)
    baseline_q = nx_comm.modularity(G_lcc, baseline_communities)
    baseline_assignment = assignment_from_communities(baseline_communities)
    baseline_summary = community_summary_row(
        name, "baseline", G_lcc, baseline_communities, baseline_q
    )
    print(f"      Baseline Q = {baseline_q:.4f}  "
          f"({len(baseline_communities)} communities on unattacked network)")
    pd.DataFrame([{
        "network": name,
        "baseline_modularity": baseline_q,
        "baseline_num_communities": len(baseline_communities),
        "baseline_num_nontrivial_communities": baseline_summary["num_nontrivial_communities"],
        "baseline_community_sizes_json": baseline_summary["community_sizes_json"],
    }]).to_csv(out_dir / "baseline_modularity.csv", index=False)
    pd.DataFrame([baseline_summary]).to_csv(out_dir / "baseline_community_summary.csv", index=False)
    save_community_assignment(
        G_lcc, baseline_assignment, out_dir, "community_assignments_baseline"
    )
    print("      Saved baseline community assignments (.csv and .pkl).")

    # Part A
    print("\n" + "─" * 65)
    print("[2/6] PART A — Tipping Point Scan")
    print("─" * 65)
    bc_tip = find_tipping_point(G_lcc, bc_order, strategy="BC", seed=42)
    dc_tip = find_tipping_point(G_lcc, dc_order, strategy="DC", seed=42)

    tipping_rows = []
    for strategy, result in [("BC", bc_tip), ("DC", dc_tip)]:
        tipping_rows.append({
            "strategy":              strategy,
            "tipping_k":             result["tipping_k"],
            "causal_node_id":        result["causal_node_id"],
            "component_sizes_json":  json.dumps(result["component_sizes"]),
        })
    pd.DataFrame(tipping_rows).to_csv(out_dir / "tipping_points.csv", index=False)

    for strat, result in [("bc", bc_tip), ("dc", dc_tip)]:
        fname = out_dir / f"community_assignments_{strat}_tipping.pkl"
        with open(fname, "wb") as f:
            pickle.dump(result["community_assignment"], f, protocol=4)

    # Part B
    print("\n" + "─" * 65)
    print("[3/6] PART B — Full-Attack Snapshot (remove all 10 nodes)")
    print("─" * 65)
    bc_full = full_attack_snapshot(G_lcc, bc_order, strategy="BC", seed=42)
    dc_full = full_attack_snapshot(G_lcc, dc_order, strategy="DC", seed=42)

    for strat, result in [("bc", bc_full), ("dc", dc_full)]:
        row = {k: v for k, v in result.items() if k != "community_assignment"}
        row["component_sizes_json"] = json.dumps(row.pop("component_sizes"))
        row["community_sizes_json"] = json.dumps(row.pop("community_sizes"))
        row["actually_removed"]     = json.dumps(row["actually_removed"])
        pd.DataFrame([row]).to_csv(out_dir / f"post_attack_{strat}.csv", index=False)

    for strat, result in [("bc", bc_full), ("dc", dc_full)]:
        fname = out_dir / f"community_assignments_{strat}_full.pkl"
        with open(fname, "wb") as f:
            pickle.dump(result["community_assignment"], f, protocol=4)

    # Linked BC attack: nodes that are high centrality in both systems
    print("\n" + "-" * 65)
    print("[3b/6] PART B2 - Linked BC Node Attack")
    print("-" * 65)
    linked_order_path = DATA / "multilayer" / f"removal_order_linked_bc_{name}.csv"
    linked_order_df = pd.read_csv(linked_order_path)
    linked_order_df["node_id"] = linked_order_df["node_id"].astype(int)
    linked_order = linked_order_df["node_id"].tolist()
    linked_order_df.to_csv(out_dir / "removal_order_linked_bc.csv", index=False)
    print(f"  Linked BC removal order from {linked_order_path.relative_to(ROOT)}: {linked_order}")

    linked_full = full_attack_snapshot(
        G_lcc, linked_order, strategy="LINKED_BC", seed=42
    )
    linked_row = {k: v for k, v in linked_full.items() if k != "community_assignment"}
    linked_row["component_sizes_json"] = json.dumps(linked_row.pop("component_sizes"))
    linked_row["community_sizes_json"] = json.dumps(linked_row.pop("community_sizes"))
    linked_row["actually_removed"] = json.dumps(linked_row["actually_removed"])
    pd.DataFrame([linked_row]).to_csv(out_dir / "post_attack_linked_bc.csv", index=False)
    save_community_assignment(
        G_lcc, linked_full["community_assignment"], out_dir,
        "community_assignments_linked_bc"
    )
    print("  Saved linked BC post-attack results and community assignments.")

    # Modularity comparison: baseline vs post-attack
    print("\n  Modularity comparison (baseline vs post-attack):")
    print(f"    {'':30s}  {'Q':>8}  {'delta Q':>9}")
    print(f"    {'Baseline (no attack)':30s}  {baseline_q:8.4f}")
    for label, result in [
        ("BC attack (k=10)", bc_full),
        ("DC attack (k=10)", dc_full),
        (f"Linked BC attack (k={len(linked_order)})", linked_full),
    ]:
        delta = result["modularity"] - baseline_q
        print(f"    {label:30s}  {result['modularity']:8.4f}  {delta:+9.4f}")

    # Part C — Stability
    print("\n" + "─" * 65)
    print("[4/6] PART C — Louvain Stability Check")
    print("─" * 65)
    G_bc = G_lcc.copy()
    G_bc.remove_nodes_from([n for n in bc_order if n in G_bc])
    G_dc = G_lcc.copy()
    G_dc.remove_nodes_from([n for n in dc_order if n in G_dc])

    bc_stab = louvain_stability_check(G_bc, strategy="BC")
    dc_stab = louvain_stability_check(G_dc, strategy="DC")
    pd.DataFrame([bc_stab, dc_stab]).to_csv(out_dir / "louvain_stability.csv", index=False)
    print(f"  Saved: louvain_stability.csv")

    # Part D — Path length (BC only; DC disconnected)
    print("\n" + "─" * 65)
    print("[5/6] PART D — Average Path Length  (BC attack; 500-node sample)")
    print("─" * 65)
    print("  Note: may take 2–4 minutes.")

    print("\n  Computing baseline APL (G_lcc)...")
    baseline_apl = sampled_avg_path_length(G_lcc, sample_size=500, seed=42)
    print(f"    Baseline APL:  {baseline_apl:.4f}")

    print("  Computing post-BC APL...")
    bc_apl     = sampled_avg_path_length(G_bc, sample_size=500, seed=42)
    pct_change = (bc_apl - baseline_apl) / baseline_apl * 100
    print(f"    Post-BC APL:   {bc_apl:.4f}  ({pct_change:+.2f}%)")

    if pct_change > 0:
        print(f"\n  FINDING: BC attack increased average travel distance by "
              f"{pct_change:.1f}% without severing the network.")
    else:
        print(f"\n  FINDING: BC attack had minimal effect on average path length.")

    print("\n  [DC skipped — graph disconnected at k=10; APL undefined]")

    pd.DataFrame([
        {"graph": "baseline",    "strategy": "—",  "avg_path_length": baseline_apl,
         "pct_change_vs_base": 0.0,        "sample_size": 500},
        {"graph": "post_attack", "strategy": "BC", "avg_path_length": bc_apl,
         "pct_change_vs_base": pct_change, "sample_size": 500},
    ]).to_csv(out_dir / "path_length_analysis.csv", index=False)
    print("  Saved: path_length_analysis.csv")

    # Summary
    print("\n" + "─" * 65)
    print(f"[6/6] Outputs saved to output/{name}/:")
    for f in sorted(out_dir.glob("*.csv")) + sorted(out_dir.glob("*.pkl")):
        print(f"      {f.name}")

    return {
        "baseline_q":            baseline_q,
        "baseline_communities":  len(baseline_communities),
        "baseline_summary":      baseline_summary,
        "linked_full":           linked_full,
        "linked_order":          linked_order,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("Script 02 — Tipping Point + Full-Attack Analysis (Transport + Power)")
    print("=" * 65)

    per_network_results = {}
    for cfg in NETWORKS:
        per_network_results[cfg["name"]] = process_network(cfg)

    # Combined modularity comparison across both networks
    print("\n" + "─" * 65)
    print("Building modularity_comparison.csv (all networks)...")
    rows = []
    linked_rows = []
    for name, res in per_network_results.items():
        bc_q = float(pd.read_csv(OUT / name / "post_attack_bc.csv").iloc[0]["modularity"])
        dc_q = float(pd.read_csv(OUT / name / "post_attack_dc.csv").iloc[0]["modularity"])
        linked_q = float(pd.read_csv(OUT / name / "post_attack_linked_bc.csv").iloc[0]["modularity"])
        bq   = res["baseline_q"]
        rows.append({
            "network":              name,
            "baseline_q":           round(bq, 4),
            "baseline_communities": res["baseline_communities"],
            "bc_post_attack_q":     round(bc_q, 4),
            "dc_post_attack_q":     round(dc_q, 4),
            "linked_bc_post_attack_q": round(linked_q, 4),
            "bc_delta_q":           round(bc_q - bq, 4),
            "dc_delta_q":           round(dc_q - bq, 4),
            "linked_bc_delta_q":     round(linked_q - bq, 4),
        })
        linked = res["linked_full"]
        linked_rows.append({
            "network": name,
            "baseline_num_communities": res["baseline_communities"],
            "linked_bc_num_communities": linked["num_communities"],
            "delta_num_communities": linked["num_communities"] - res["baseline_communities"],
            "baseline_modularity": round(bq, 6),
            "linked_bc_modularity": round(linked["modularity"], 6),
            "delta_modularity": round(linked["modularity"] - bq, 6),
            "num_nodes_removed": linked["num_nodes_removed"],
            "removed_node_ids_json": json.dumps(linked["actually_removed"]),
            "num_components": linked["num_components"],
            "component_sizes_json": json.dumps(linked["component_sizes"]),
            "num_nontrivial_communities": linked["num_nontrivial_communities"],
            "community_sizes_json": json.dumps(linked["community_sizes"]),
        })
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(OUT / "modularity_comparison.csv", index=False)
    print("  Saved: output/modularity_comparison.csv")
    print()
    print(comparison_df.to_string(index=False))

    OUT_MULTILAYER.mkdir(parents=True, exist_ok=True)
    linked_summary_df = pd.DataFrame(linked_rows)
    linked_summary_df.to_csv(
        OUT_MULTILAYER / "linked_bc_community_comparison.csv", index=False
    )
    print("\n  Saved: output/multilayer/linked_bc_community_comparison.csv")
    print()
    print(linked_summary_df[[
        "network",
        "baseline_num_communities",
        "linked_bc_num_communities",
        "delta_num_communities",
        "baseline_modularity",
        "linked_bc_modularity",
        "delta_modularity",
        "removed_node_ids_json",
    ]].to_string(index=False))

    print("\n" + "=" * 65)
    print("Done.")
    print("=" * 65)
