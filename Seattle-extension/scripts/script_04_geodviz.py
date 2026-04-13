"""
script_04_geodviz.py
====================
Geographic scatter-plot maps of the Seattle street network colored by Louvain
community membership. Produces one map per strategy × snapshot:

  map_bc_tipping.png   — BC attack, network state at the tipping point k
  map_bc_full.png      — BC attack, network state after all 10 nodes removed
  map_dc_tipping.png   — DC attack, network state at the tipping point k
  map_dc_full.png      — DC attack, network state after all 10 nodes removed

Coordinates are WA State Plane North (EPSG:2926, US survey feet).
Plotted directly on X/Y axes — no reprojection needed.

Run AFTER script_02_percolation.py:
    python scripts/script_04_geodviz.py

Outputs (written to output/figures/):
    map_bc_tipping.png
    map_bc_full.png
    map_dc_tipping.png
    map_dc_full.png
"""

import matplotlib
matplotlib.use("Agg")

import pickle
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

ROOT    = pathlib.Path(__file__).parent.parent
OUT     = ROOT / "output"
FIGURES = OUT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

DPI      = 150
FIG_SIZE = (10, 12)


def build_node_df(G_lcc: nx.Graph,
                  assignment: dict) -> pd.DataFrame:
    """
    Build a DataFrame of nodes that survived the attack, with coordinates
    and their assigned Louvain community index.

    Parameters
    ----------
    G_lcc      : original LCC (coordinate source for all nodes)
    assignment : {node_id: community_index} — keys are survived nodes only

    Returns
    -------
    pd.DataFrame  columns: node_id, longitude, latitude, community_id
    """
    rows = []
    for node_id, comm_idx in assignment.items():
        attr = G_lcc.nodes[node_id]
        rows.append({
            "node_id":      node_id,
            "longitude":    attr["longitude"],
            "latitude":     attr["latitude"],
            "community_id": comm_idx,
        })
    return pd.DataFrame(rows)


def plot_community_map(G_lcc: nx.Graph,
                       assignment: dict,
                       removal_order_df: pd.DataFrame,
                       k: int,
                       strategy: str,
                       snapshot_label: str,
                       save_path: pathlib.Path):
    """
    Render a geographic scatter map colored by Louvain community.

    Survived nodes are scattered in community colors. The k removed nodes
    are shown as red X markers with rank and node-ID labels.

    Parameters
    ----------
    G_lcc            : original LCC (coordinate lookup)
    assignment       : {node_id: community_index} for survived nodes
    removal_order_df : full 10-row removal order (node_id, longitude, latitude)
    k                : how many nodes were removed in this snapshot
    strategy         : 'BC' or 'DC'
    snapshot_label   : 'Tipping Point' or 'Full Attack (k=10)'
    save_path        : output PNG path
    """
    survived_df = build_node_df(G_lcc, assignment)
    removed_df  = removal_order_df.iloc[:k].copy().reset_index(drop=True)

    num_communities = survived_df["community_id"].nunique()

    # Build colormap — tab20 handles up to 20 categories cleanly
    if num_communities <= 20:
        cmap = plt.cm.get_cmap("tab20", num_communities)
    else:
        cmap = plt.cm.get_cmap("hsv", num_communities)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # ── Survived nodes colored by community ───────────────────────────────────
    for comm_id in sorted(survived_df["community_id"].unique()):
        subset = survived_df[survived_df["community_id"] == comm_id]
        ax.scatter(subset["longitude"], subset["latitude"],
                   c=[cmap(comm_id % num_communities)],
                   s=1.5, alpha=0.55, linewidths=0, rasterized=True)

    # ── Removed nodes as red X markers ────────────────────────────────────────
    ax.scatter(removed_df["longitude"], removed_df["latitude"],
               c="red", marker="x", s=140, linewidths=2.5,
               zorder=5, label=f"Removed nodes  (n={k})")

    for i, row in removed_df.iterrows():
        ax.annotate(
            f" #{i+1}  (node {int(row['node_id'])})",
            xy=(row["longitude"], row["latitude"]),
            fontsize=7.5, color="darkred", fontweight="bold",
            xytext=(6, 3), textcoords="offset points", zorder=6,
        )

    # ── Colorbar for community index ──────────────────────────────────────────
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=0, vmax=max(num_communities - 1, 1))
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Louvain community index", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # ── Formatting ────────────────────────────────────────────────────────────
    removed_patch = mpatches.Patch(color="red", label=f"Removed nodes  (n={k})")
    ax.legend(handles=[removed_patch], loc="lower right", fontsize=10)

    ax.set_aspect("equal")
    ax.set_xlabel("Easting  (ft, EPSG:2926 — WA State Plane North)", fontsize=10)
    ax.set_ylabel("Northing  (ft, EPSG:2926 — WA State Plane North)", fontsize=10)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title(
        f"Seattle Street Network — {strategy} Attack  |  {snapshot_label}\n"
        f"{num_communities} Louvain communities  ·  "
        f"{len(survived_df):,} surviving nodes  ·  "
        f"{k} node(s) removed",
        fontsize=12,
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}  "
          f"({num_communities} communities, k={k})")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Script 04 — Geographic Community Maps")
    print("=" * 60)

    print("\n[1/3] Loading inputs...")
    with open(OUT / "lcc_graph.pkl", "rb") as f:
        G_lcc = pickle.load(f)

    bc_order_df = pd.read_csv(OUT / "removal_order_bc.csv")
    dc_order_df = pd.read_csv(OUT / "removal_order_dc.csv")

    tips_df = pd.read_csv(OUT / "tipping_points.csv")

    def get_tip(strategy: str):
        row = tips_df[tips_df["strategy"] == strategy].iloc[0]
        return None if pd.isna(row["tipping_k"]) else int(row["tipping_k"])

    bc_tip = get_tip("BC")
    dc_tip = get_tip("DC")
    print(f"  BC tipping k = {bc_tip}")
    print(f"  DC tipping k = {dc_tip}")
    print(f"  LCC graph: {G_lcc.number_of_nodes():,} nodes")

    print("\n[2/3] Generating maps...")

    configs = [
        # (strategy_label, order_df, tipping_k, tipping_pkl, full_pkl)
        ("BC", bc_order_df, bc_tip,
         OUT / "community_assignments_bc_tipping.pkl",
         OUT / "community_assignments_bc_full.pkl"),
        ("DC", dc_order_df, dc_tip,
         OUT / "community_assignments_dc_tipping.pkl",
         OUT / "community_assignments_dc_full.pkl"),
    ]

    for strategy, order_df, tip_k, tipping_pkl, full_pkl in configs:

        # Map at tipping point
        if tip_k is not None:
            with open(tipping_pkl, "rb") as f:
                tip_assignment = pickle.load(f)
            plot_community_map(
                G_lcc          = G_lcc,
                assignment     = tip_assignment,
                removal_order_df = order_df,
                k              = tip_k,
                strategy       = strategy,
                snapshot_label = f"Tipping Point  (k={tip_k})",
                save_path      = FIGURES / f"map_{strategy.lower()}_tipping.png",
            )
        else:
            print(f"  [{strategy}] No tipping point — skipping tipping map.")

        # Map at k=10 (full attack)
        with open(full_pkl, "rb") as f:
            full_assignment = pickle.load(f)
        plot_community_map(
            G_lcc          = G_lcc,
            assignment     = full_assignment,
            removal_order_df = order_df,
            k              = len(order_df),
            strategy       = strategy,
            snapshot_label = f"Full Attack  (k={len(order_df)})",
            save_path      = FIGURES / f"map_{strategy.lower()}_full.png",
        )

    print("\n[3/3] Maps generated:")
    for f in sorted(FIGURES.glob("map_*.png")):
        print(f"  {f.name}")

    print("\n" + "=" * 60)
    print("Done. All maps saved to output/figures/")
    print("=" * 60)
