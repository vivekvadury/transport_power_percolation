"""
script_03_comparison.py
=======================
Load percolation results and produce comparison plots + summary statistics.
No graph computation here — pure pandas + matplotlib.

Run AFTER script_02_percolation.py:
    python scripts/script_03_comparison.py

Outputs (written to output/figures/ and output/):
    figures/lcc_fraction_comparison.png
    figures/components_comparison.png
    figures/modularity_comparison.png
    figures/communities_comparison.png
    comparison_summary.csv
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — must be set before pyplot import

import json
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = pathlib.Path(__file__).parent.parent
OUT     = ROOT / "output"
FIGURES = OUT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Plot style ─────────────────────────────────────────────────────────────────
BC_COLOR  = "#1f77b4"   # blue
DC_COLOR  = "#ff7f0e"   # orange
BC_MARKER = "o"
DC_MARKER = "^"
LINEWIDTH = 2
MARKERSIZE = 8
FIG_SIZE  = (10, 6)
DPI       = 150


def add_tipping_lines(ax, bc_tip, dc_tip):
    """Draw vertical dashed lines at each strategy's tipping point."""
    if bc_tip is not None:
        ax.axvline(bc_tip, color=BC_COLOR, linestyle="--", linewidth=1.2,
                   alpha=0.7, label=f"BC tipping k={bc_tip}")
    if dc_tip is not None:
        ax.axvline(dc_tip, color=DC_COLOR, linestyle="--", linewidth=1.2,
                   alpha=0.7, label=f"DC tipping k={dc_tip}")


def plot_lcc_fraction(bc_df, dc_df, bc_tip, dc_tip, save_path):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(bc_df["k"], bc_df["lcc_fraction"], color=BC_COLOR,
            marker=BC_MARKER, markersize=MARKERSIZE, linewidth=LINEWIDTH,
            label="BC (Betweenness Centrality)")
    ax.plot(dc_df["k"], dc_df["lcc_fraction"], color=DC_COLOR,
            marker=DC_MARKER, markersize=MARKERSIZE, linewidth=LINEWIDTH,
            label="DC (Degree Centrality)")
    add_tipping_lines(ax, bc_tip, dc_tip)

    ax.set_xlabel("k  (number of nodes removed)", fontsize=13)
    ax.set_ylabel("LCC fraction  (LCC size / original N)", fontsize=13)
    ax.set_title("LCC Fraction vs. Nodes Removed\nBC vs. DC Targeted Attack",
                 fontsize=14)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=3))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    print(f"      Saved: {save_path.name}")


def plot_num_components(bc_df, dc_df, bc_tip, dc_tip, save_path):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(bc_df["k"], bc_df["num_components"], color=BC_COLOR,
            marker=BC_MARKER, markersize=MARKERSIZE, linewidth=LINEWIDTH,
            label="BC (Betweenness Centrality)")
    ax.plot(dc_df["k"], dc_df["num_components"], color=DC_COLOR,
            marker=DC_MARKER, markersize=MARKERSIZE, linewidth=LINEWIDTH,
            label="DC (Degree Centrality)")
    add_tipping_lines(ax, bc_tip, dc_tip)

    ax.set_xlabel("k  (number of nodes removed)", fontsize=13)
    ax.set_ylabel("Number of connected components", fontsize=13)
    ax.set_title("Connected Components vs. Nodes Removed\nBC vs. DC Targeted Attack",
                 fontsize=14)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    print(f"      Saved: {save_path.name}")


def plot_modularity(bc_df, dc_df, bc_tip, dc_tip, save_path):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(bc_df["k"], bc_df["modularity"], color=BC_COLOR,
            marker=BC_MARKER, markersize=MARKERSIZE, linewidth=LINEWIDTH,
            label="BC (Betweenness Centrality)")
    ax.plot(dc_df["k"], dc_df["modularity"], color=DC_COLOR,
            marker=DC_MARKER, markersize=MARKERSIZE, linewidth=LINEWIDTH,
            label="DC (Degree Centrality)")
    add_tipping_lines(ax, bc_tip, dc_tip)

    ax.set_xlabel("k  (number of nodes removed)", fontsize=13)
    ax.set_ylabel("Louvain Modularity (Q)", fontsize=13)
    ax.set_title("Louvain Modularity vs. Nodes Removed\n"
                 "BC vs. DC Targeted Attack  (higher Q = stronger community structure)",
                 fontsize=14)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    print(f"      Saved: {save_path.name}")


def plot_num_communities(bc_df, dc_df, bc_tip, dc_tip, save_path):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(bc_df["k"], bc_df["num_nontrivial_communities"], color=BC_COLOR,
            marker=BC_MARKER, markersize=MARKERSIZE, linewidth=LINEWIDTH,
            label="BC (Betweenness Centrality)")
    ax.plot(dc_df["k"], dc_df["num_nontrivial_communities"], color=DC_COLOR,
            marker=DC_MARKER, markersize=MARKERSIZE, linewidth=LINEWIDTH,
            label="DC (Degree Centrality)")
    add_tipping_lines(ax, bc_tip, dc_tip)

    ax.set_xlabel("k  (number of nodes removed)", fontsize=13)
    ax.set_ylabel("Non-trivial communities  (size ≥ 2)", fontsize=13)
    ax.set_title("Louvain Community Count vs. Nodes Removed\n"
                 "BC vs. DC Targeted Attack  (singleton communities excluded)",
                 fontsize=14)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    print(f"      Saved: {save_path.name}")


def build_summary(bc_df, dc_df, bc_tip, dc_tip) -> pd.DataFrame:
    rows = []
    for strategy, df, tip in [("bc", bc_df, bc_tip), ("dc", dc_df, dc_tip)]:
        last = df.iloc[-1]
        first = df.iloc[0]
        rows.append({
            "strategy":               strategy.upper(),
            "tipping_k":              tip if tip is not None else "None",
            "lcc_fraction_at_k10":    round(last["lcc_fraction"], 6),
            "max_num_components":     int(df["num_components"].max()),
            "modularity_at_k1":       round(first["modularity"], 6),
            "modularity_at_k10":      round(last["modularity"], 6),
            "nontrivial_communities_at_k10": int(last["num_nontrivial_communities"]),
        })
    return pd.DataFrame(rows)


def print_summary_table(summary_df: pd.DataFrame):
    bc = summary_df[summary_df["strategy"] == "BC"].iloc[0]
    dc = summary_df[summary_df["strategy"] == "DC"].iloc[0]
    print()
    print("=" * 55)
    print("  BC vs DC Comparison Summary")
    print("=" * 55)
    rows = [
        ("Tipping point k",              bc["tipping_k"],                dc["tipping_k"]),
        ("LCC fraction at k=10",         f"{bc['lcc_fraction_at_k10']:.6f}",
                                         f"{dc['lcc_fraction_at_k10']:.6f}"),
        ("Max connected components",     bc["max_num_components"],       dc["max_num_components"]),
        ("Modularity at k=1",            f"{bc['modularity_at_k1']:.4f}", f"{dc['modularity_at_k1']:.4f}"),
        ("Modularity at k=10",           f"{bc['modularity_at_k10']:.4f}", f"{dc['modularity_at_k10']:.4f}"),
        ("Non-trivial communities k=10", bc["nontrivial_communities_at_k10"],
                                         dc["nontrivial_communities_at_k10"]),
    ]
    print(f"  {'Metric':<32} {'BC':>9}  {'DC':>9}")
    print("  " + "-" * 51)
    for label, bc_val, dc_val in rows:
        print(f"  {label:<32} {str(bc_val):>9}  {str(dc_val):>9}")
    print("=" * 55)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Script 03 — Comparison Plots & Summary")
    print("=" * 60)

    # Load percolation results
    print("\n[1/3] Loading percolation results...")
    bc_df = pd.read_csv(OUT / "percolation_bc.csv")
    dc_df = pd.read_csv(OUT / "percolation_dc.csv")
    tips  = pd.read_csv(OUT / "tipping_points.csv")

    def get_tip(strategy: str):
        row = tips[tips["strategy"] == strategy]
        val = row["tipping_k"].values[0]
        return None if pd.isna(val) else int(val)

    bc_tip = get_tip("bc")
    dc_tip = get_tip("dc")
    print(f"      BC tipping point: k={bc_tip}")
    print(f"      DC tipping point: k={dc_tip}")

    # Generate plots
    print("\n[2/3] Generating comparison plots...")
    plot_lcc_fraction(bc_df, dc_df, bc_tip, dc_tip,
                      FIGURES / "lcc_fraction_comparison.png")
    plot_num_components(bc_df, dc_df, bc_tip, dc_tip,
                        FIGURES / "components_comparison.png")
    plot_modularity(bc_df, dc_df, bc_tip, dc_tip,
                    FIGURES / "modularity_comparison.png")
    plot_num_communities(bc_df, dc_df, bc_tip, dc_tip,
                         FIGURES / "communities_comparison.png")

    # Summary table
    print("\n[3/3] Building summary...")
    summary_df = build_summary(bc_df, dc_df, bc_tip, dc_tip)
    summary_df.to_csv(OUT / "comparison_summary.csv", index=False)
    print("      comparison_summary.csv  ✓")
    print_summary_table(summary_df)

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)
