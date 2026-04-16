"""
script_04_geodviz.py
====================
Geographic scatter-plot maps colored by Louvain community membership.

Produces separate maps per network AND side-by-side combined maps:

  Per-network (output/figures/transport/ and output/figures/power/):
    map_bc_tipping.png   — BC attack at tipping point k
    map_bc_full.png      — BC attack after all 10 nodes removed
    map_dc_tipping.png   — DC attack at tipping point k
    map_dc_full.png      — DC attack after all 10 nodes removed

  Combined side-by-side (output/figures/combined/):
    map_bc_tipping.png   — transport (left) + power (right)
    map_bc_full.png
    map_dc_tipping.png
    map_dc_full.png

Coordinate systems:
  Transport: EPSG:2926 — WA State Plane North (US survey feet)
  Power:     WGS84 — Longitude / Latitude (decimal degrees)

Run AFTER script_02_percolation.py:
    python scripts/script_04_geodviz.py
"""

import matplotlib
matplotlib.use("Agg")

import pickle
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

ROOT     = pathlib.Path(__file__).parent.parent
OUT      = ROOT / "output"
FIG_T    = OUT / "figures" / "transport"
FIG_P    = OUT / "figures" / "power"
FIG_COMB = OUT / "figures" / "combined"
for d in (FIG_T, FIG_P, FIG_COMB):
    d.mkdir(parents=True, exist_ok=True)

DPI      = 150
FIG_SIZE = (10, 12)

# Per-network metadata
NETWORKS = {
    "transport": {
        "out_dir":      OUT / "transport",
        "fig_dir":      FIG_T,
        "xlabel":       "Easting  (ft, EPSG:2926 — WA State Plane North)",
        "ylabel":       "Northing  (ft, EPSG:2926 — WA State Plane North)",
    },
    "power": {
        "out_dir":      OUT / "power",
        "fig_dir":      FIG_P,
        "xlabel":       "Longitude  (decimal degrees, WGS84)",
        "ylabel":       "Latitude   (decimal degrees, WGS84)",
    },
}


# ── Core map function ──────────────────────────────────────────────────────────

def build_node_df(G_lcc: nx.Graph, assignment: dict) -> pd.DataFrame:
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
                       network: str,
                       save_path: pathlib.Path,
                       ax=None,
                       fig=None):
    """
    Render one geographic scatter map colored by Louvain community.

    If ax is provided, draw into that axis (used for combined side-by-side maps).
    Otherwise create a standalone figure and save it.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=FIG_SIZE)

    survived_df = build_node_df(G_lcc, assignment)
    removed_df  = removal_order_df.iloc[:k].copy().reset_index(drop=True)
    num_communities = survived_df["community_id"].nunique()

    if num_communities <= 20:
        cmap = plt.colormaps["tab20"].resampled(num_communities)
    else:
        cmap = plt.colormaps["hsv"].resampled(num_communities)

    # Survived nodes
    for comm_id in sorted(survived_df["community_id"].unique()):
        subset = survived_df[survived_df["community_id"] == comm_id]
        ax.scatter(subset["longitude"], subset["latitude"],
                   c=[cmap(comm_id % num_communities)],
                   s=1.5, alpha=0.55, linewidths=0, rasterized=True)

    # Removed nodes
    ax.scatter(removed_df["longitude"], removed_df["latitude"],
               c="red", marker="x", s=140, linewidths=2.5,
               zorder=5, label=f"Removed nodes  (n={k})")
    for i, row in removed_df.iterrows():
        ax.annotate(
            f" #{i+1}  (node {int(row['node_id'])})",
            xy=(row["longitude"], row["latitude"]),
            fontsize=6.5, color="darkred", fontweight="bold",
            xytext=(6, 3), textcoords="offset points", zorder=6,
        )

    cfg = NETWORKS[network]
    ax.set_aspect("equal")
    ax.set_xlabel(cfg["xlabel"], fontsize=9)
    ax.set_ylabel(cfg["ylabel"], fontsize=9)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_title(
        f"{network.title()} — {strategy} Attack  |  {snapshot_label}\n"
        f"{num_communities} Louvain communities  ·  "
        f"{len(survived_df):,} surviving nodes  ·  "
        f"{k} node(s) removed",
        fontsize=10,
    )

    removed_patch = mpatches.Patch(color="red", label=f"Removed nodes  (n={k})")
    ax.legend(handles=[removed_patch], loc="lower right", fontsize=9)

    if standalone:
        # Add colorbar only in standalone figures (shared layout breaks with twin axes)
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=0, vmax=max(num_communities - 1, 1))
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label("Louvain community index", fontsize=9)
        cbar.ax.tick_params(labelsize=7)

        fig.tight_layout()
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path.relative_to(ROOT)}  "
              f"({num_communities} communities, k={k})")

    return num_communities


def plot_combined_map(G_t: nx.Graph, t_assignment: dict, t_order: pd.DataFrame,
                      G_p: nx.Graph, p_assignment: dict, p_order: pd.DataFrame,
                      k_t: int, k_p: int,
                      strategy: str,
                      snapshot_label: str,
                      save_path: pathlib.Path):
    """
    Side-by-side map: transport (left) + power (right).
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 12))

    nc_t = plot_community_map(
        G_t, t_assignment, t_order, k_t,
        strategy, snapshot_label, "transport",
        save_path=None, ax=axes[0], fig=fig
    )
    nc_p = plot_community_map(
        G_p, p_assignment, p_order, k_p,
        strategy, snapshot_label, "power",
        save_path=None, ax=axes[1], fig=fig
    )

    fig.suptitle(
        f"Seattle — {strategy} Attack  |  {snapshot_label}\n"
        f"Transport: {nc_t} communities    Power: {nc_p} communities",
        fontsize=13, y=1.01
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.relative_to(ROOT)}  "
          f"(T={nc_t} communities, P={nc_p} communities, k_t={k_t}, k_p={k_p})")


# ── Load all inputs for one network ───────────────────────────────────────────

def load_network_data(name: str) -> dict:
    out_dir = NETWORKS[name]["out_dir"]
    with open(out_dir / "lcc_graph.pkl", "rb") as f:
        G_lcc = pickle.load(f)

    bc_order = pd.read_csv(out_dir / "removal_order_bc.csv")
    dc_order = pd.read_csv(out_dir / "removal_order_dc.csv")
    tips_df  = pd.read_csv(out_dir / "tipping_points.csv")

    def get_tip(strategy):
        row = tips_df[tips_df["strategy"] == strategy].iloc[0]
        return None if pd.isna(row["tipping_k"]) else int(row["tipping_k"])

    def load_pkl(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    return {
        "G_lcc":      G_lcc,
        "bc_order":   bc_order,
        "dc_order":   dc_order,
        "bc_tip":     get_tip("BC"),
        "dc_tip":     get_tip("DC"),
        "bc_tip_asgn": load_pkl(out_dir / "community_assignments_bc_tipping.pkl"),
        "dc_tip_asgn": load_pkl(out_dir / "community_assignments_dc_tipping.pkl"),
        "bc_full_asgn": load_pkl(out_dir / "community_assignments_bc_full.pkl"),
        "dc_full_asgn": load_pkl(out_dir / "community_assignments_dc_full.pkl"),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("Script 04 — Geographic Community Maps (Transport + Power)")
    print("=" * 65)

    print("\n[1/3] Loading inputs...")
    t = load_network_data("transport")
    p = load_network_data("power")
    print(f"  Transport LCC: {t['G_lcc'].number_of_nodes():,} nodes  "
          f"| BC tip k={t['bc_tip']}  DC tip k={t['dc_tip']}")
    print(f"  Power LCC:     {p['G_lcc'].number_of_nodes():,} nodes  "
          f"| BC tip k={p['bc_tip']}  DC tip k={p['dc_tip']}")

    # ── Per-network maps ──────────────────────────────────────────────────────
    print("\n[2/3] Generating per-network maps...")
    for name, d in [("transport", t), ("power", p)]:
        fig_dir = NETWORKS[name]["fig_dir"]
        print(f"\n  -- {name.upper()} --")

        for strategy, tip_k, tip_asgn, full_asgn, order_df in [
            ("BC", d["bc_tip"], d["bc_tip_asgn"], d["bc_full_asgn"], d["bc_order"]),
            ("DC", d["dc_tip"], d["dc_tip_asgn"], d["dc_full_asgn"], d["dc_order"]),
        ]:
            if tip_k is not None:
                plot_community_map(
                    d["G_lcc"], tip_asgn, order_df, tip_k,
                    strategy, f"Tipping Point  (k={tip_k})", name,
                    fig_dir / f"map_{strategy.lower()}_tipping.png"
                )
            else:
                print(f"  [{strategy}] No tipping point — skipping tipping map.")

            full_k = len(order_df)
            plot_community_map(
                d["G_lcc"], full_asgn, order_df, full_k,
                strategy, f"Full Attack  (k={full_k})", name,
                fig_dir / f"map_{strategy.lower()}_full.png"
            )

    # ── Combined side-by-side maps ────────────────────────────────────────────
    print("\n[3/3] Generating combined side-by-side maps...")

    for strategy, t_tip, p_tip, t_tip_asgn, p_tip_asgn, t_full_asgn, p_full_asgn in [
        ("BC",
         t["bc_tip"], p["bc_tip"],
         t["bc_tip_asgn"], p["bc_tip_asgn"],
         t["bc_full_asgn"], p["bc_full_asgn"]),
        ("DC",
         t["dc_tip"], p["dc_tip"],
         t["dc_tip_asgn"], p["dc_tip_asgn"],
         t["dc_full_asgn"], p["dc_full_asgn"]),
    ]:
        # Tipping point combined map — only when BOTH networks have a tipping point
        if t_tip is not None and p_tip is not None:
            plot_combined_map(
                t["G_lcc"], t_tip_asgn, t["bc_order" if strategy == "BC" else "dc_order"],
                p["G_lcc"], p_tip_asgn, p["bc_order" if strategy == "BC" else "dc_order"],
                t_tip, p_tip,
                strategy, f"Tipping Point",
                FIG_COMB / f"map_{strategy.lower()}_tipping.png"
            )
        elif t_tip is None and p_tip is None:
            print(f"  [{strategy}] Neither network has a tipping point — skipping combined tipping map.")
        else:
            missing = "transport" if t_tip is None else "power"
            print(f"  [{strategy}] {missing} has no tipping point — skipping combined tipping map "
                  f"(see per-network maps for the network that did fracture).")

        # Full attack combined map
        t_order = t["bc_order" if strategy == "BC" else "dc_order"]
        p_order = p["bc_order" if strategy == "BC" else "dc_order"]
        plot_combined_map(
            t["G_lcc"], t_full_asgn, t_order,
            p["G_lcc"], p_full_asgn, p_order,
            len(t_order), len(p_order),
            strategy, f"Full Attack  (k=10)",
            FIG_COMB / f"map_{strategy.lower()}_full.png"
        )

    print("\n" + "=" * 65)
    print("Done. Maps saved to:")
    print("  output/figures/transport/")
    print("  output/figures/power/")
    print("  output/figures/combined/")
    print("=" * 65)
