"""
plot_trajectory_Et.py

Plots E vs t (energy evolution during SA) to visualize the stagnation / plateauing
of the energy, from a trajectory_*_aggregated.csv produced by trajectory_SA4NPP.jl.

One panel per N, one curve per b (color), with shaded +/- stderr bands. X-axis is
linear MCS-step by default (use --logx for log-spaced step axis, which is often
clearer since the underlying checkpoint grid is itself log-spaced).

Usage:
    python plot_trajectory_Et.py                       # latest experiment, best_energy
    python plot_trajectory_Et.py trajectory_20260101_120000
    python plot_trajectory_Et.py --energy current       # plot current_energy instead
    python plot_trajectory_Et.py --logx                 # log-spaced x-axis
    python plot_trajectory_Et.py --show
"""

import sys
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FOLDER_PATH = Path(__file__).parent / "Experiments"
TIMESTAMP_RE = re.compile(r"trajectory_(\d{8}_\d{6})_aggregated\.csv$")


def find_latest_aggregated_csv(folder=None):
    """Find the most recent trajectory_*_aggregated.csv by embedded timestamp."""
    folder = Path(folder) if folder else FOLDER_PATH
    candidates = []
    for f in folder.glob("trajectory_*_aggregated.csv"):
        m = TIMESTAMP_RE.search(f.name)
        if m:
            candidates.append((m.group(1), f))
    if not candidates:
        raise FileNotFoundError(f"No trajectory_*_aggregated.csv files found in {folder}")
    candidates.sort(key=lambda p: p[0])
    return candidates[-1][1]


def find_aggregated_csv(folder=None, identifier=None):
    """Locate a single aggregated CSV: by identifier, or the latest if None."""
    folder = Path(folder) if folder else FOLDER_PATH
    if identifier is None:
        return find_latest_aggregated_csv(folder)
    identifier = str(identifier).strip()
    if identifier.endswith("_aggregated.csv") or identifier.endswith(".csv"):
        candidate = folder / identifier
    else:
        candidate = folder / f"{identifier}_aggregated.csv"
    if not candidate.exists():
        raise FileNotFoundError(f"No aggregated CSV matching '{identifier}' found in {folder}")
    return candidate


def load_aggregated_data(csv_file):
    """Load a trajectory aggregated CSV into a DataFrame, sorted for plotting."""
    required = ["N", "b", "kappa_actual", "regime", "cooling_strategy", "step",
                "mean_best_energy", "mean_current_energy"]
    df = pd.read_csv(csv_file)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_file}: {missing}")
    return df.sort_values(["N", "b", "step"]).reset_index(drop=True)


def plot_Et(df, csv_file, energy_col, stderr_col, logx, show, save, output_path):
    """
    Build one figure with one panel per N, overlaying one E(t) curve per b.

    Arguments:
        df: aggregated trajectory DataFrame (single regime/cooling_strategy expected;
            if several are present, each (regime, cooling_strategy) pair gets its own
            figure)
        energy_col, stderr_col: which energy/stderr columns to plot
    """
    match = TIMESTAMP_RE.search(Path(csv_file).name)
    timestamp = match.group(1) if match else Path(csv_file).stem

    combos = df[["regime", "cooling_strategy"]].drop_duplicates().values.tolist()
    figs = []

    for regime, cooling_strategy in combos:
        sub_all = df[(df["regime"] == regime) & (df["cooling_strategy"] == cooling_strategy)]
        N_values = sorted(sub_all["N"].unique())
        n_panels = len(N_values)
        ncols = min(2, n_panels)
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
        axes_flat = axes.flatten()

        for ax, N in zip(axes_flat, N_values):
            sub_N = sub_all[sub_all["N"] == N]
            b_values = sorted(sub_N["b"].unique())
            cmap = plt.colormaps.get_cmap("plasma")
            colors = {b: cmap(i / max(1, len(b_values) - 1)) for i, b in enumerate(b_values)}

            for b in b_values:
                curve = sub_N[sub_N["b"] == b].sort_values("step")
                kappa = curve["kappa_actual"].iloc[0]
                c = colors[b]
                ax.plot(curve["step"], curve[energy_col], "o-", color=c, markersize=3,
                        label=f"b={b} (kappa={kappa:.3f})")
                if stderr_col in curve:
                    ax.fill_between(curve["step"],
                                     curve[energy_col] - curve[stderr_col],
                                     curve[energy_col] + curve[stderr_col],
                                     color=c, alpha=0.15)

            if logx:
                ax.set_xscale("log")
            ax.set_title(f"N={N}", fontsize=12, fontweight="bold")
            ax.set_xlabel("MCS step (t)", fontsize=10)
            ax.set_ylabel(energy_col, fontsize=10)
            ax.grid(True, alpha=0.3, which="both" if logx else "major")
            ax.legend(fontsize=7, loc="best")

        # Hide unused panels
        for ax in axes_flat[n_panels:]:
            ax.set_visible(False)

        fig.suptitle(f"NPP SA energy trajectories -- {energy_col} -- regime={regime}, "
                     f"cooling={cooling_strategy} -- experiment {timestamp}",
                     fontsize=14, fontweight="bold", y=1.0)
        plt.tight_layout()

        if save:
            FOLDER_PATH.mkdir(exist_ok=True)
            energy = "best_energy" if energy_col == "mean_best_energy" else "current_energy"
            out = output_path or FOLDER_PATH / f"trajectory_{timestamp}_{regime}_{cooling_strategy}_Et_{energy}.png"
            plt.savefig(out, dpi=300, bbox_inches="tight")
            print(f"✓ Plot saved: {out}")

        figs.append(fig)

    if show:
        plt.show()

    return figs


def main():
    parser = argparse.ArgumentParser(description="Plot E vs t stagnation curves from SA trajectory data.")
    parser.add_argument("identifier", nargs="?", default=None,
                         help="timestamp/basename/filename of a specific experiment (default: latest)")
    parser.add_argument("--energy", choices=["best", "current"], default="best",
                         help="plot mean_best_energy (default) or mean_current_energy")
    parser.add_argument("--logx", action="store_true", help="use a log-spaced x-axis")
    parser.add_argument("--show", "-s", action="store_true", help="display interactively")
    parser.add_argument("--no-save", action="store_true", help="don't save a PNG")
    args = parser.parse_args()

    if args.show:
        import matplotlib
        matplotlib.use("TkAgg")
    else:
        import matplotlib
        matplotlib.use("Agg")

    csv_file = find_aggregated_csv(identifier=args.identifier)
    print(f"{'Using latest experiment' if args.identifier is None else 'Loading'}: {csv_file.name}")

    df = load_aggregated_data(csv_file)
    print(f"\nLoaded {len(df)} (config, step) rows:")
    print(f"  N values: {sorted(df['N'].unique())}")
    print(f"  (regime, cooling_strategy) combos: {df[['regime','cooling_strategy']].drop_duplicates().values.tolist()}")

    energy_col = "mean_best_energy" if args.energy == "best" else "mean_current_energy"
    stderr_col = "stderr_best_energy" if args.energy == "best" else "stderr_current_energy"

    plot_Et(df, csv_file, energy_col, stderr_col, args.logx, args.show, not args.no_save, None)

    if not args.show:
        print("(Use --show or -s flag to display interactively)")
    print("\n✓ Plot complete!")


if __name__ == "__main__":
    main()
