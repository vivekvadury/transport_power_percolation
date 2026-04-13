"""
script_04_geodviz.py
====================
Geographic scatter-plot maps of the Seattle street network, colored by
Louvain community membership, at two snapshots per strategy:
  - At the tipping point k (first new component appears)
  - At k=10 (final state)

Coordinates are WA State Plane North (EPSG:2926, US survey feet).
Plotted directly as X/Y — no reprojection needed for relative geometry.

Run AFTER script_02_percolation.py:
    python scripts/script_04_geodviz.py

Outputs (written to output/figures/):
    map_bc_k_tipping.png
    map_bc_k10.png
    map_dc_k_tipping.png
    map_dc_k10.png
"""

import matplotlib
matplotlib.use("Agg")

import pickle
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = pathlib.Path(__file__).parent.parent
OUT     = ROOT / "output"
FIGURES = OUT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

DPI      = 150
FIG_SIZE = (10, 12)


def build_node_df(G_lcc: nx.Graph,
                  assignment: dict[int, int]) -> pd.DataFrame:
    """
    Build a DataFrame of survived nodes with coordinates and community ID.

    Parameters
    ----------
    G_lcc      : original LCC graph (for coordinate lookup on all nodes)
    assignment : {node_id: community_index} for nodes remaining after attack

    Returns
    -------
    pd.DataFrame  columns: node_id, longitude, latitude, community_id
    """
    rows = []
    for node_id, comm_idx in assignment.items():
        attr = G_lcc.nodes[node_id]
        rows.append({
            "node_id":     node_id,
            "longitude":   attr["longitude"],
            "latitude":    attr["latitude"],
            "community_id": comm_idx,
        })
    return pd.DataFrame(rows)


def plot_community_map(G_lcc: nx.Graph,
                       assignment: dict[int, int],
                       removal_order_df: pd.DataFrame,
                       k: int,
                       strategy: str,
                       save_path: pathlib.Path):
    """
    Produce a geographic scatter map colored by community.

    Survived nodes are scattered in community colors; removed nodes shown
    as red X markers with node-ID annotations.

    Parameters
    ----------
    G_lcc            : original LCC (coordinate source)
    assignment       : {node_id: community_index} at this k
    removal_order_df : DataFrame with columns node_id, longitude, latitude
                       (full 10-node removal order; first k rows are removed)
    k                : number of nodes removed
    strategy         : 'BC' or 'DC'
    save_path        : output PNG path
    """
    survived_df = build_node_df(G_lcc, assignment)
    num_communities = survived_df["community_id"].nunique()

    removed_df = removal_order_df.iloc[:k].copy()

    # ── Colormap for communities ───────────────────────────────────────────────
    if num_communities <= 20:
        cmap = plt.cm.get_cmap("tab20", num_communities)
    else:
        cmap = plt.cm.get_cmap("hsv", num_communities)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # ── Scatter survived nodes ─────────────────────────────────────────────────
    for comm_id in sorted(survived_df["community_id"].unique()):
        subset = survived_df[survived_df["community_id"] == comm_id]
        ax.scatter(subset["longitude"], subset["latitude"],
                   c=[cmap(comm_id % num_communities)],
                   s=1.5, alpha=0.55, linewidths=0, rasterized=True)

    # ── Overlay removed nodes ──────────────────────────────────────────────────
    ax.scatter(removed_df["longitude"], removed_df["latitude"],
               c="red", marker="x", s=120, linewidths=2.5,
               zorder=5, label=f"Removed nodes (k={k})")

    # Annotate each removed node with its node_id and rank
    for rank_idx, row in removed_df.iterrows():
        rank = rank_idx + 1  # 1-based rank within removal_order_df
        ax.annotate(
            f" #{rank} (id {int(row['node_id'])})",
            xy=(row["longitude"], row["latitude"]),
            fontsize=7, color="darkred", fontweight="bold",
            xytext=(6, 3), textcoords="offset points",
            zorder=6,
        )

    # ── Legend for removed nodes ───────────────────────────────────────────────
    removed_patch = mpatches.Patch(color="red", label=f"Removed nodes (k={k})")
    ax.legend(handles=[removed_patch], loc="lower right", fontsize=10)

    # ── Labels and formatting ──────────────────────────────────────────────────
    ax.set_aspect("equal")
    ax.set_xlabel("Easting  (ft, EPSG:2926 — WA State Plane North)", fontsize=11)
    ax.set_ylabel("Northing  (ft, EPSG:2926 — WA State Plane North)", fontsize=11)
    ax.set_title(
        f"Seattle Street Network — {strategy} Attack,  k = {k}\n"
        f"{num_communities} Louvain communities  |  "
        f"{len(survived_df):,} surviving nodes  |  "
        f"{k} node(s) removed",
        fontsize=13,
    )
    ax.tick_params(axis="both", labelsize=8)

    # Colorbar legend for communities
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=0, vmax=num_communities - 1)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Community index", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved: {save_path.name}  "
          f"({num_communities} communities, k={k})")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Script 04 — Geographic Community Maps")
    print("=" * 60)

    # Load inputs
    print("\n[1/3] Loading graph, community assignments, and tipping points...")
    with open(OUT / "lcc_graph.pkl", "rb") as f:
        G_lcc = pickle.load(f)
    with open(OUT / "community_assignments_bc.pkl", "rb") as f:
        bc_assignments = pickle.load(f)
    with open(OUT / "community_assignments_dc.pkl", "rb") as f:
        dc_assignments = pickle.load(f)

    bc_order_df = pd.read_csv(OUT / "removal_order_bc.csv")
    dc_order_df = pd.read_csv(OUT / "removal_order_dc.csv")

    tips = pd.read_csv(OUT / "tipping_points.csv")

    def get_tip(strategy: str) -> int | None:
        row = tips[tips["strategy"] == strategy]
        val = row["tipping_k"].values[0]
        return None if pd.isna(val) else int(val)

    bc_tip = get_tip("bc")
    dc_tip = get_tip("dc")
    print(f"      BC tipping point: k={bc_tip}")
    print(f"      DC tipping point: k={dc_tip}")

    print(f"\n      LCC graph: {G_lcc.number_of_nodes()} nodes")

    # Generate maps
    print("\n[2/3] Generating geographic maps...")

    configs = [
        # (strategy_label, assignments_dict, order_df, tipping_k, k_list)
        ("BC", bc_assignments, bc_order_df, bc_tip),
        ("DC", dc_assignments, dc_order_df, dc_tip),
    ]

    for strategy, assignments, order_df, tip in configs:
        k_snapshots = []
        if tip is not None and tip != 10:
            k_snapshots.append((tip, "k_tipping"))
        elif tip is not None:
            k_snapshots.append((tip, "k_tipping"))   # tip == 10: same as k10
        else:
            print(f"      {strategy}: no tipping point — skipping tipping map")
        k_snapshots.append((10, "k10"))

        for k_val, suffix in k_snapshots:
            fname = f"map_{strategy.lower()}_{suffix}.png"
            plot_community_map(
                G_lcc=G_lcc,
                assignment=assignments[k_val],
                removal_order_df=order_df,
                k=k_val,
                strategy=strategy,
                save_path=FIGURES / fname,
            )

    print("\n[3/3] Summary of generated maps:")
    for f in sorted(FIGURES.glob("map_*.png")):
        print(f"      {f.name}")

    print("\n" + "=" * 60)
    print("Done. All maps saved to output/figures/")
    print("=" * 60)
