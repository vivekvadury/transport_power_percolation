"""
script_03_comparison.py
=======================
Load results from script_02 for BOTH networks and produce:

  Per-network plots (output/figures/transport/ and output/figures/power/):
    tipping_point_comparison.png
    community_size_distribution.png
    post_attack_summary.png

  Combined plots (output/figures/combined/):
    tipping_point_comparison.png  — all 4 strategies (BC-T, DC-T, BC-P, DC-P)
    community_size_distribution.png — 2-row subplot (transport / power)
    post_attack_summary.png       — 4-bar groups per metric

No graph computation — pure pandas + matplotlib.

Run AFTER script_02_percolation.py:
    python scripts/script_03_comparison.py
"""

import matplotlib
matplotlib.use("Agg")

import json
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT     = pathlib.Path(__file__).parent.parent
OUT      = ROOT / "output"
FIG_T    = OUT / "figures" / "transport"
FIG_P    = OUT / "figures" / "power"
FIG_COMB = OUT / "figures" / "combined"
for d in (FIG_T, FIG_P, FIG_COMB):
    d.mkdir(parents=True, exist_ok=True)

BC_COLOR   = "#1f77b4"   # blue   — BC Transport
DC_COLOR   = "#ff7f0e"   # orange — DC Transport
BC_P_COLOR = "#2ca02c"   # green  — BC Power
DC_P_COLOR = "#d62728"   # red    — DC Power
DPI = 150


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_network(out_dir: pathlib.Path) -> dict:
    return {
        "tips":   pd.read_csv(out_dir / "tipping_points.csv"),
        "bc_row": pd.read_csv(out_dir / "post_attack_bc.csv").iloc[0],
        "dc_row": pd.read_csv(out_dir / "post_attack_dc.csv").iloc[0],
    }


def tip_value_and_label(tips_df: pd.DataFrame, strategy: str, network: str):
    row = tips_df[tips_df["strategy"] == strategy].iloc[0]
    k   = row["tipping_k"]
    if pd.isna(k):
        return 10, f"{strategy} ({network}) — no fracture in k=1..10"
    k    = int(k)
    node = int(row["causal_node_id"])
    sizes = json.loads(row["component_sizes_json"])
    return k, f"{strategy} ({network}) — fractures at k={k}  (node {node})\n  Sizes: {sizes}"


# ══════════════════════════════════════════════════════════════════════════════
# Per-network plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_tipping_point(tips_df: pd.DataFrame,
                       save_path: pathlib.Path,
                       network: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    values, labels = [], []
    for strat in ["BC", "DC"]:
        v, lbl = tip_value_and_label(tips_df, strat, network)
        values.append(v)
        labels.append(lbl)

    bars = ax.barh(["BC", "DC"], values,
                   color=[BC_COLOR, DC_COLOR], height=0.4, alpha=0.85)
    for bar, label in zip(bars, labels):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=9)

    ax.set_xlim(0, 15)
    ax.set_xlabel("k  (nodes removed to first fracture)", fontsize=12)
    ax.set_title(f"Tipping Point — {network.title()} Network\n"
                 f"BC vs DC Targeted Attack", fontsize=13)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.relative_to(ROOT)}")


def plot_community_sizes(bc_row: pd.Series,
                         dc_row: pd.Series,
                         save_path: pathlib.Path,
                         network: str):
    bc_sizes = json.loads(bc_row["community_sizes_json"])
    dc_sizes = json.loads(dc_row["community_sizes_json"])
    max_rank = max(len(bc_sizes), len(dc_sizes))
    bc_sizes += [0] * (max_rank - len(bc_sizes))
    dc_sizes += [0] * (max_rank - len(dc_sizes))

    top_n = min(20, max_rank)
    ranks = np.arange(1, top_n + 1)
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(ranks - width / 2, bc_sizes[:top_n], width,
           label="BC attack", color=BC_COLOR, alpha=0.85)
    ax.bar(ranks + width / 2, dc_sizes[:top_n], width,
           label="DC attack", color=DC_COLOR, alpha=0.85)
    ax.set_xlabel("Community rank  (1 = largest)", fontsize=12)
    ax.set_ylabel("Number of nodes", fontsize=12)
    ax.set_title(f"Post-Attack Community Size Distribution  (k=10)\n"
                 f"{network.title()} Network — top 20 communities", fontsize=13)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.relative_to(ROOT)}")


def plot_post_attack_summary(bc_row: pd.Series,
                             dc_row: pd.Series,
                             save_path: pathlib.Path,
                             network: str):
    metrics = ["Connected\nComponents", "Non-trivial\nCommunities"]
    bc_vals = [int(bc_row["num_components"]),
               int(bc_row["num_nontrivial_communities"])]
    dc_vals = [int(dc_row["num_components"]),
               int(dc_row["num_nontrivial_communities"])]

    x, width = np.arange(len(metrics)), 0.3
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(x - width / 2, bc_vals, width,
            label="BC attack", color=BC_COLOR, alpha=0.85)
    ax1.bar(x + width / 2, dc_vals, width,
            label="DC attack", color=DC_COLOR, alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax2  = ax1.twinx()
    bc_q = float(bc_row["modularity"])
    dc_q = float(dc_row["modularity"])
    ax2.bar(2.0 - width / 2, bc_q, width, color=BC_COLOR, alpha=0.55, hatch="//")
    ax2.bar(2.0 + width / 2, dc_q, width, color=DC_COLOR, alpha=0.55, hatch="//")
    ax2.set_ylabel("Modularity (Q)", fontsize=12, color="grey")
    ax2.tick_params(axis="y", labelcolor="grey")
    ax2.set_ylim(0, 1)

    ax1.set_xlim(-0.6, 2.8)
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(
        metrics + [f"Modularity Q\n(BC={bc_q:.3f}, DC={dc_q:.3f})"],
        fontsize=11)
    ax1.set_title(f"Post-Attack Network State  (k=10)\n"
                  f"{network.title()} Network — BC vs DC", fontsize=13)
    ax1.legend(fontsize=11, loc="upper left")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.relative_to(ROOT)}")


# ══════════════════════════════════════════════════════════════════════════════
# Combined plots (transport + power on one figure)
# ══════════════════════════════════════════════════════════════════════════════

def plot_tipping_combined(t_tips: pd.DataFrame,
                          p_tips: pd.DataFrame,
                          save_path: pathlib.Path):
    """
    4-bar horizontal chart: BC-Transport, DC-Transport, BC-Power, DC-Power.
    """
    entries = [
        ("BC", "Transport", t_tips, BC_COLOR),
        ("DC", "Transport", t_tips, DC_COLOR),
        ("BC", "Power",     p_tips, BC_P_COLOR),
        ("DC", "Power",     p_tips, DC_P_COLOR),
    ]
    ylabels, values, colors, labels = [], [], [], []
    for strat, net, tips_df, color in entries:
        v, lbl = tip_value_and_label(tips_df, strat, net)
        ylabels.append(f"{strat}  ({net})")
        values.append(v)
        colors.append(color)
        labels.append(lbl)

    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.barh(ylabels, values, color=colors, height=0.45, alpha=0.85)
    for bar, label in zip(bars, labels):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=8.5)

    ax.set_xlim(0, 18)
    ax.set_xlabel("k  (nodes removed to first fracture)", fontsize=12)
    ax.set_title("Tipping Point Comparison — Transport vs Power Network\n"
                 "BC and DC Targeted Attack", fontsize=13)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.relative_to(ROOT)}")


def plot_community_sizes_combined(t_bc: pd.Series, t_dc: pd.Series,
                                  p_bc: pd.Series, p_dc: pd.Series,
                                  save_path: pathlib.Path):
    """
    2-row subplot: top = transport (BC vs DC), bottom = power (BC vs DC).
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)

    for ax, bc_row, dc_row, network in [
        (axes[0], t_bc, t_dc, "Transport"),
        (axes[1], p_bc, p_dc, "Power"),
    ]:
        bc_sizes = json.loads(bc_row["community_sizes_json"])
        dc_sizes = json.loads(dc_row["community_sizes_json"])
        max_rank = max(len(bc_sizes), len(dc_sizes))
        bc_sizes += [0] * (max_rank - len(bc_sizes))
        dc_sizes += [0] * (max_rank - len(dc_sizes))
        top_n = min(20, max_rank)
        ranks = np.arange(1, top_n + 1)
        width = 0.35
        ax.bar(ranks - width / 2, bc_sizes[:top_n], width,
               label="BC attack", color=BC_COLOR, alpha=0.85)
        ax.bar(ranks + width / 2, dc_sizes[:top_n], width,
               label="DC attack", color=DC_COLOR, alpha=0.85)
        ax.set_title(f"{network} Network", fontsize=12)
        ax.set_xlabel("Community rank  (1 = largest)", fontsize=10)
        ax.set_ylabel("Number of nodes", fontsize=10)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle("Post-Attack Community Size Distribution  (k=10)\n"
                 "Transport vs Power — top 20 communities shown", fontsize=13)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.relative_to(ROOT)}")


def plot_post_attack_combined(t_bc: pd.Series, t_dc: pd.Series,
                              p_bc: pd.Series, p_dc: pd.Series,
                              save_path: pathlib.Path):
    """
    Grouped bars: for each metric, 4 bars (BC-T, DC-T, BC-P, DC-P).
    Modularity on secondary axis. Solid bars = transport, hatched = power.
    """
    metric_labels = ["Connected\nComponents", "Non-trivial\nCommunities"]
    groups = [
        ("BC  Transport", [int(t_bc["num_components"]), int(t_bc["num_nontrivial_communities"])],
         BC_COLOR,   ""),
        ("DC  Transport", [int(t_dc["num_components"]), int(t_dc["num_nontrivial_communities"])],
         DC_COLOR,   ""),
        ("BC  Power",     [int(p_bc["num_components"]), int(p_bc["num_nontrivial_communities"])],
         BC_P_COLOR, "//"),
        ("DC  Power",     [int(p_dc["num_components"]), int(p_dc["num_nontrivial_communities"])],
         DC_P_COLOR, "//"),
    ]

    n_groups = len(groups)
    width    = 0.18
    x        = np.arange(len(metric_labels))
    offsets  = np.linspace(-(n_groups - 1) / 2,
                            (n_groups - 1) / 2, n_groups) * width

    fig, ax1 = plt.subplots(figsize=(11, 6))
    for (label, vals, color, hatch), offset in zip(groups, offsets):
        ax1.bar(x + offset, vals, width, label=label,
                color=color, alpha=0.85, hatch=hatch)

    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Modularity on secondary axis at x=2
    ax2     = ax1.twinx()
    q_vals  = [float(t_bc["modularity"]), float(t_dc["modularity"]),
               float(p_bc["modularity"]), float(p_dc["modularity"])]
    q_x     = 2.0 + offsets
    q_colors = [BC_COLOR, DC_COLOR, BC_P_COLOR, DC_P_COLOR]
    q_hatch  = ["", "", "//", "//"]
    for qx, qv, qc, qh in zip(q_x, q_vals, q_colors, q_hatch):
        ax2.bar(qx, qv, width, color=qc, alpha=0.55, hatch=qh)
    ax2.set_ylabel("Modularity (Q)", fontsize=12, color="grey")
    ax2.tick_params(axis="y", labelcolor="grey")
    ax2.set_ylim(0, 1)

    q_label = (f"Modularity Q\n"
               f"BC-T={q_vals[0]:.3f}  DC-T={q_vals[1]:.3f}\n"
               f"BC-P={q_vals[2]:.3f}  DC-P={q_vals[3]:.3f}")
    ax1.set_xlim(-0.55, 2.85)
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(metric_labels + [q_label], fontsize=10)
    ax1.set_title("Post-Attack Network State  (k=10)\n"
                  "Transport vs Power — BC and DC Attack", fontsize=13)
    ax1.legend(fontsize=9, loc="upper left", ncol=2)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.relative_to(ROOT)}")


# ── Console summary ────────────────────────────────────────────────────────────

def print_summary(network: str, tips_df: pd.DataFrame,
                  bc_row: pd.Series, dc_row: pd.Series):
    def fmt_tip(strat):
        row = tips_df[tips_df["strategy"] == strat].iloc[0]
        return f"k={int(row['tipping_k'])}" if not pd.isna(row["tipping_k"]) else "None"

    bc_sizes = json.loads(bc_row["community_sizes_json"])
    dc_sizes = json.loads(dc_row["community_sizes_json"])
    rows = [
        ("Tipping point",               fmt_tip("BC"),                          fmt_tip("DC")),
        ("Connected components",        int(bc_row["num_components"]),          int(dc_row["num_components"])),
        ("Louvain communities (total)", int(bc_row["num_communities"]),         int(dc_row["num_communities"])),
        ("  Non-trivial (size >= 2)",   int(bc_row["num_nontrivial_communities"]),
                                        int(dc_row["num_nontrivial_communities"])),
        ("Modularity Q",               f"{float(bc_row['modularity']):.4f}",   f"{float(dc_row['modularity']):.4f}"),
        ("Largest community (nodes)",   bc_sizes[0],                            dc_sizes[0]),
    ]
    print(f"\n  {network.upper()} — Post-Attack Comparison (k=10)")
    print(f"  {'Metric':<35} {'BC':>10}  {'DC':>10}")
    print("  " + "-" * 57)
    for label, bv, dv in rows:
        print(f"  {label:<35} {str(bv):>10}  {str(dv):>10}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("Script 03 — Comparison Plots & Summary (Transport + Power)")
    print("=" * 65)

    print("\n[1/3] Loading results from script_02...")
    t = load_network(OUT / "transport")
    p = load_network(OUT / "power")

    print("\n[2/3] Generating per-network plots...")
    for data, fig_dir, network in [
        (t, FIG_T, "transport"),
        (p, FIG_P, "power"),
    ]:
        print(f"\n  -- {network.upper()} --")
        plot_tipping_point(data["tips"],
                           fig_dir / "tipping_point_comparison.png", network)
        plot_community_sizes(data["bc_row"], data["dc_row"],
                             fig_dir / "community_size_distribution.png", network)
        plot_post_attack_summary(data["bc_row"], data["dc_row"],
                                 fig_dir / "post_attack_summary.png", network)

    print("\n[3/3] Generating combined plots...")
    plot_tipping_combined(t["tips"], p["tips"],
                          FIG_COMB / "tipping_point_comparison.png")
    plot_community_sizes_combined(t["bc_row"], t["dc_row"],
                                  p["bc_row"], p["dc_row"],
                                  FIG_COMB / "community_size_distribution.png")
    plot_post_attack_combined(t["bc_row"], t["dc_row"],
                              p["bc_row"], p["dc_row"],
                              FIG_COMB / "post_attack_summary.png")

    print("\n" + "=" * 65)
    print("Summary tables:")
    print_summary("transport", t["tips"], t["bc_row"], t["dc_row"])
    print_summary("power",     p["tips"], p["bc_row"], p["dc_row"])
    print("\n" + "=" * 65)
    print("Done.")
    print("=" * 65)
