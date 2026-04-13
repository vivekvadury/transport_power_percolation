"""
script_03_comparison.py
=======================
Load the results from script_02 and produce three focused plots:

  Plot 1 — Tipping Point Comparison
    Side-by-side: at what k does BC vs DC first fracture the network?
    Marks which specific node caused each split.

  Plot 2 — Post-Attack Community Size Distribution
    After removing all 10 nodes: how many communities formed and how large
    are they? One grouped bar chart comparing BC vs DC.

  Plot 3 — Post-Attack Summary Bar Chart
    Side-by-side bars for: number of components, number of communities,
    and modularity (Q) — BC vs DC at k=10.

No graph computation here — pure pandas + matplotlib.

Run AFTER script_02_percolation.py:
    python scripts/script_03_comparison.py

Outputs (written to output/figures/ and output/):
    figures/tipping_point_comparison.png
    figures/community_size_distribution.png
    figures/post_attack_summary.png
    comparison_summary.csv
"""

import matplotlib
matplotlib.use("Agg")

import json
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT    = pathlib.Path(__file__).parent.parent
OUT     = ROOT / "output"
FIGURES = OUT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

BC_COLOR = "#1f77b4"   # blue
DC_COLOR = "#ff7f0e"   # orange
DPI      = 150


# ── Plot 1: Tipping Point Comparison ──────────────────────────────────────────

def plot_tipping_point(tips_df: pd.DataFrame, save_path: pathlib.Path):
    """
    Horizontal bar chart showing tipping k for BC and DC.
    If no tipping point, bar reaches to 10 with a 'No fracture' label.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    strategies = ["BC", "DC"]
    colors     = [BC_COLOR, DC_COLOR]
    labels     = []
    values     = []

    for strat in strategies:
        row = tips_df[tips_df["strategy"] == strat].iloc[0]
        k   = row["tipping_k"]
        if pd.isna(k):
            values.append(10)
            labels.append(f"{strat} — no fracture in k=1..10")
        else:
            k = int(k)
            node = int(row["causal_node_id"])
            sizes = json.loads(row["component_sizes_json"])
            values.append(k)
            labels.append(
                f"{strat} — fractures at k={k}  (node {node} removed)\n"
                f"  Component sizes: {sizes}"
            )

    bars = ax.barh(strategies, values, color=colors, height=0.4, alpha=0.85)

    # Annotate bars with the label text
    for bar, label in zip(bars, labels):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=10)

    ax.set_xlim(0, 13)
    ax.set_xlabel("k  (number of nodes removed to first fracture)", fontsize=12)
    ax.set_title(
        "Tipping Point: First Network Fracture\n"
        "BC vs DC Targeted Attack on Seattle Street Network",
        fontsize=13
    )
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# ── Plot 2: Community Size Distribution ───────────────────────────────────────

def plot_community_sizes(bc_row: pd.Series,
                         dc_row: pd.Series,
                         save_path: pathlib.Path):
    """
    Grouped bar chart: for each community rank (1st largest, 2nd largest, ...),
    show the size under BC and DC attack.

    This directly answers: how fragmented is each attack's result, and are the
    resulting community sizes similar or very different between strategies?
    """
    bc_sizes = json.loads(bc_row["community_sizes_json"])
    dc_sizes = json.loads(dc_row["community_sizes_json"])

    # Align to the same number of ranks (pad shorter list with 0)
    max_rank = max(len(bc_sizes), len(dc_sizes))
    bc_sizes += [0] * (max_rank - len(bc_sizes))
    dc_sizes += [0] * (max_rank - len(dc_sizes))

    # Only plot top 20 ranks for readability (tail communities are tiny)
    top_n = min(20, max_rank)
    ranks  = np.arange(1, top_n + 1)
    bc_top = bc_sizes[:top_n]
    dc_top = dc_sizes[:top_n]

    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(ranks - width / 2, bc_top, width, label="BC attack", color=BC_COLOR, alpha=0.85)
    ax.bar(ranks + width / 2, dc_top, width, label="DC attack", color=DC_COLOR, alpha=0.85)

    ax.set_xlabel("Community rank  (1 = largest community)", fontsize=12)
    ax.set_ylabel("Number of nodes", fontsize=12)
    ax.set_title(
        "Post-Attack Community Size Distribution  (k=10)\n"
        "BC vs DC Targeted Attack — top 20 communities shown",
        fontsize=13
    )
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# ── Plot 3: Post-Attack Summary Bars ──────────────────────────────────────────

def plot_post_attack_summary(bc_row: pd.Series,
                             dc_row: pd.Series,
                             save_path: pathlib.Path):
    """
    Three grouped bar pairs comparing BC vs DC on the key post-attack metrics:
      - Connected components
      - Non-trivial Louvain communities
      - Modularity Q (on a secondary axis for scale)
    """
    metrics = ["Connected\nComponents", "Non-trivial\nCommunities"]
    bc_vals = [
        int(bc_row["num_components"]),
        int(bc_row["num_nontrivial_communities"]),
    ]
    dc_vals = [
        int(dc_row["num_components"]),
        int(dc_row["num_nontrivial_communities"]),
    ]

    x     = np.arange(len(metrics))
    width = 0.3

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(x - width / 2, bc_vals, width, label="BC attack", color=BC_COLOR, alpha=0.85)
    ax1.bar(x + width / 2, dc_vals, width, label="DC attack", color=DC_COLOR, alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Modularity on a secondary axis
    ax2 = ax1.twinx()
    bc_q = float(bc_row["modularity"])
    dc_q = float(dc_row["modularity"])
    q_x  = [2.0 - width / 2, 2.0 + width / 2]  # position after the count bars
    ax2.bar(q_x[0], bc_q, width, color=BC_COLOR, alpha=0.55, hatch="//")
    ax2.bar(q_x[1], dc_q, width, color=DC_COLOR, alpha=0.55, hatch="//")
    ax2.set_ylabel("Modularity (Q)", fontsize=12, color="grey")
    ax2.tick_params(axis="y", labelcolor="grey")
    ax2.set_ylim(0, 1)

    ax1.set_xlim(-0.6, 2.8)
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(metrics + [f"Modularity Q\n(BC={bc_q:.3f}, DC={dc_q:.3f})"],
                        fontsize=11)

    ax1.set_title(
        "Post-Attack Network State  (k=10)\n"
        "BC vs DC Targeted Attack",
        fontsize=13
    )
    ax1.legend(fontsize=11, loc="upper left")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# ── Console summary table ──────────────────────────────────────────────────────

def print_summary(tips_df: pd.DataFrame,
                  bc_row: pd.Series,
                  dc_row: pd.Series):
    bc_tip = tips_df[tips_df["strategy"] == "BC"].iloc[0]
    dc_tip = tips_df[tips_df["strategy"] == "DC"].iloc[0]

    def fmt_tip(row):
        return f"k={int(row['tipping_k'])}" if not pd.isna(row["tipping_k"]) else "None"

    bc_sizes = json.loads(bc_row["community_sizes_json"])
    dc_sizes = json.loads(dc_row["community_sizes_json"])

    print()
    print("=" * 60)
    print("  Post-Attack Comparison Summary (k=10)")
    print("=" * 60)
    rows = [
        ("Tipping point",                fmt_tip(bc_tip),                    fmt_tip(dc_tip)),
        ("Connected components",         int(bc_row["num_components"]),       int(dc_row["num_components"])),
        ("Louvain communities (total)",  int(bc_row["num_communities"]),      int(dc_row["num_communities"])),
        ("  Non-trivial (size ≥ 2)",     int(bc_row["num_nontrivial_communities"]),
                                         int(dc_row["num_nontrivial_communities"])),
        ("Modularity Q",                 f"{float(bc_row['modularity']):.4f}", f"{float(dc_row['modularity']):.4f}"),
        ("Largest community (nodes)",    bc_sizes[0],                         dc_sizes[0]),
        ("2nd largest community",        bc_sizes[1] if len(bc_sizes) > 1 else "—",
                                         dc_sizes[1] if len(dc_sizes) > 1 else "—"),
    ]
    print(f"  {'Metric':<35} {'BC':>10}  {'DC':>10}")
    print("  " + "-" * 57)
    for label, bv, dv in rows:
        print(f"  {label:<35} {str(bv):>10}  {str(dv):>10}")
    print("=" * 60)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Script 03 — Comparison Plots & Summary")
    print("=" * 60)

    print("\n[1/3] Loading results from script_02...")
    tips_df = pd.read_csv(OUT / "tipping_points.csv")
    bc_row  = pd.read_csv(OUT / "post_attack_bc.csv").iloc[0]
    dc_row  = pd.read_csv(OUT / "post_attack_dc.csv").iloc[0]

    print("\n[2/3] Generating plots...")
    plot_tipping_point(tips_df, FIGURES / "tipping_point_comparison.png")
    plot_community_sizes(bc_row, dc_row, FIGURES / "community_size_distribution.png")
    plot_post_attack_summary(bc_row, dc_row, FIGURES / "post_attack_summary.png")

    print("\n[3/3] Saving comparison_summary.csv and printing table...")
    summary = pd.DataFrame([{
        "strategy":                   s,
        "tipping_k":                  tips_df[tips_df["strategy"] == s]["tipping_k"].values[0],
        "num_components":             int(r["num_components"]),
        "num_communities":            int(r["num_communities"]),
        "num_nontrivial_communities": int(r["num_nontrivial_communities"]),
        "modularity":                 float(r["modularity"]),
        "largest_community_size":     json.loads(r["community_sizes_json"])[0],
    } for s, r in [("BC", bc_row), ("DC", dc_row)]])
    summary.to_csv(OUT / "comparison_summary.csv", index=False)

    print_summary(tips_df, bc_row, dc_row)

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)
