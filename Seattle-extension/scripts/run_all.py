"""
run_all.py
==========
Runs all four analysis scripts in order.

    python scripts/run_all.py

Equivalent to running:
    python scripts/script_01_build_graph.py
    python scripts/script_02_percolation.py
    python scripts/script_03_comparison.py
    python scripts/script_04_geodviz.py
"""

import subprocess
import sys
import pathlib
import time

SCRIPTS_DIR = pathlib.Path(__file__).parent

SCRIPTS = [
    "script_01_build_graph.py",
    "script_02_percolation.py",
    "script_03_comparison.py",
    "script_04_geodviz.py",
]


def run_script(script_name: str) -> bool:
    path = SCRIPTS_DIR / script_name
    print(f"\n{'#' * 65}")
    print(f"# Running: {script_name}")
    print(f"{'#' * 65}\n")
    start = time.time()
    result = subprocess.run([sys.executable, str(path)], cwd=str(path.parent.parent))
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n[ERROR] {script_name} failed (exit code {result.returncode}).")
        print("Stopping — fix the error above before re-running.")
        return False
    print(f"\n[OK] {script_name} finished in {elapsed:.1f}s")
    return True


if __name__ == "__main__":
    overall_start = time.time()
    print("=" * 65)
    print("  Seattle Extension — Full Analysis Pipeline")
    print("=" * 65)

    for script in SCRIPTS:
        if not run_script(script):
            sys.exit(1)

    total = time.time() - overall_start
    print(f"\n{'=' * 65}")
    print(f"  All scripts completed in {total:.1f}s")
    print(f"  Results are in:  output/transport/  and  output/power/")
    print(f"  Figures are in:  output/figures/transport/")
    print(f"                   output/figures/power/")
    print(f"                   output/figures/combined/")
    print(f"{'=' * 65}")
