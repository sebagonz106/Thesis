"""
plot_trajectory_loglog_kappac.py

Log-log (log E vs log t) view of the SA energy trajectories, used to visually AND
numerically estimate kappa_c(N): the b (equivalently kappa = b/N) at which the
energy trajectory switches from "plateaus near 0 / solved" (easy, kappa < kappa_c)
to "stays stuck at an O(sqrt(N))-ish plateau" (hard, kappa > kappa_c). For a given N,
the b-curve closest to kappa_c is the one that looks most like the dividing straight
line between the two families on the log-log plot.

kappa_c estimation (a SIMPLE plateau-height interpolation across curves,
analogous to the P_sat=0.5 crossing already used in plot_phase_diagram_SA.py, but
here on the PLATEAU ENERGY of each trajectory rather than on P_sat):

    1. For each (N, b) curve, take the plateau energy = mean_best_energy at the LAST
       checkpoint (i.e. the value the run has settled into, since trajectories are
       flat-filled after an early stop -- see trajectory_SA4NPP.jl).
    2. Sort these plateau energies by kappa_actual = b/N.
    3. Linearly interpolate (in kappa) for the kappa at which the plateau energy
       crosses a fixed `--threshold` (default: SUCCESS_THRESHOLD = 1.1, matching the
       Julia side's definition of a "perfect" partition) between the two bracketing
       points -- same bisection-free bracketing approach as find_crossing() in
       plot_phase_diagram_SA.py, just on plateau energy instead of P_sat.

This requires >= 2 b points per N that bracket the threshold; if the sweep doesn't
bracket it for a given N, no crossing is reported for that N (a warning is printed
and plotting continues without it).

Usage:
    python plot_trajectory_loglog_kappac.py                       # latest experiment
    python plot_trajectory_loglog_kappac.py trajectory_20260101_120000
    python plot_trajectory_loglog_kappac.py --energy current
    python plot_trajectory_loglog_kappac.py --threshold 2.0
    python plot_trajectory_loglog_kappac.py --show
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
SUCCESS_THRESHOLD = 1.1


def find_latest_aggregated_csv(folder=None):
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
    required = ["N", "b", "kappa_actual", "regime", "cooling_strategy", "step",
                "mean_best_energy", "mean_current_energy"]
    df = pd.read_csv(csv_file)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_file}: {missing}")
    return df.sort_values(["N", "b", "step"]).reset_index(drop=True)


def plateau_energies(df_N, energy_col):
    """
    For each b within df_N (a single N), return the plateau (final-step) energy,
    sorted by kappa_actual. Returns (kappas, b_values, plateau_E) as parallel arrays.
    """
    rows = []
    for b, curve in df_N.groupby("b"):
        curve = curve.sort_values("step")
        last = curve.iloc[-1]
        rows.append((last["kappa_actual"], b, last[energy_col]))
    rows.sort(key=lambda r: r[0])
    kappas = np.array([r[0] for r in rows])
    b_values = [r[1] for r in rows]
    plateau_E = np.array([r[2] for r in rows])
    return kappas, b_values, plateau_E


def estimate_kappa_c(kappas, plateau_E, threshold):
    """
    Simple linear-interpolation crossing of plateau_E(kappa) = threshold, scanning
    consecutive (kappa) pairs for a sign change (same bracketing logic as
    find_crossing() in plot_phase_diagram_SA.py). Returns (kappa_c, bracket_idx) or
    (None, None) if no bracketing crossing is found.
    """
    for i in range(len(kappas) - 1):
        y0, y1 = plateau_E[i] - threshold, plateau_E[i + 1] - threshold
        if y0 * y1 <= 0 and y0 != y1:
            frac = -y0 / (y1 - y0)
            kappa_c = kappas[i] + frac * (kappas[i + 1] - kappas[i])
            return kappa_c, i
    return None, None


def plot_loglog_kappac(df, csv_file, energy_col, threshold, show, save, output_path):
    match = TIMESTAMP_RE.search(Path(csv_file).name)
    timestamp = match.group(1) if match else Path(csv_file).stem

    combos = df[["regime", "cooling_strategy"]].drop_duplicates().values.tolist()
    figs = []
    kappa_c_results = {}  # (regime, cooling_strategy, N) -> kappa_c or None

    for regime, cooling_strategy in combos:
        sub_all = df[(df["regime"] == regime) & (df["cooling_strategy"] == cooling_strategy)]
        N_values = sorted(sub_all["N"].unique())
        n_panels = len(N_values)
        ncols = min(2, n_panels)
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.5 * nrows), squeeze=False)
        axes_flat = axes.flatten()

        for ax, N in zip(axes_flat, N_values):
            sub_N = sub_all[sub_all["N"] == N]
            b_values_sorted = sorted(sub_N["b"].unique())
            cmap = plt.colormaps.get_cmap("plasma")
            colors = {b: cmap(i / max(1, len(b_values_sorted) - 1)) for i, b in enumerate(b_values_sorted)}

            kappas, b_list, plateau_E = plateau_energies(sub_N, energy_col)
            kappa_c, bracket_idx = estimate_kappa_c(kappas, plateau_E, threshold)
            kappa_c_results[(regime, cooling_strategy, N)] = kappa_c

            # Curve closest to kappa_c (if found) drawn bold/black as the
            # approximate dividing line; others plotted normally in color.
            closest_b = None
            if kappa_c is not None:
                closest_idx = int(np.argmin(np.abs(kappas - kappa_c)))
                closest_b = b_list[closest_idx]

            for b in b_values_sorted:
                curve = sub_N[sub_N["b"] == b].sort_values("step")
                kappa = curve["kappa_actual"].iloc[0]
                # Only positive, finite values are valid on a log-log plot.
                mask = (curve["step"] > 0) & (curve[energy_col] > 0)
                x = curve["step"][mask]
                y = curve[energy_col][mask]
                is_critical = (b == closest_b)
                ax.plot(x, y, "o-" if not is_critical else "s-",
                        color="black" if is_critical else colors[b],
                        linewidth=2.5 if is_critical else 1.3,
                        markersize=5 if is_critical else 3,
                        zorder=5 if is_critical else 3,
                        label=f"b={b} (kappa={kappa:.3f})" + (" [~kappa_c]" if is_critical else ""))

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"N={N}", fontsize=12, fontweight="bold")
            ax.set_xlabel("log t  (MCS step)", fontsize=10)
            ax.set_ylabel(f"log {energy_col}", fontsize=10)
            ax.grid(True, alpha=0.3, which="both")
            ax.legend(fontsize=7, loc="best")

            if kappa_c is not None:
                note = f"kappa_c(N={N}) ~ {kappa_c:.4f}\n(threshold={threshold:g} crossing,\nbracketed between b={b_list[bracket_idx]} and b={b_list[bracket_idx+1]})"
            else:
                note = f"No plateau-energy crossing of threshold={threshold:g}\nfound in this b range for N={N}"
            ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=8,
                    verticalalignment="bottom", horizontalalignment="left",
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

        for ax in axes_flat[n_panels:]:
            ax.set_visible(False)

        fig.suptitle(f"NPP SA log-log trajectories & kappa_c estimate -- {energy_col} -- "
                     f"regime={regime}, cooling={cooling_strategy} -- experiment {timestamp}",
                     fontsize=14, fontweight="bold", y=1.0)
        plt.tight_layout()

        if save:
            FOLDER_PATH.mkdir(exist_ok=True)
            energy = "best_energy" if energy_col == "mean_best_energy" else "current_energy"
            out = output_path or FOLDER_PATH / f"trajectory_{timestamp}_{regime}_{cooling_strategy}_loglog_{energy}_kappac.png"
            plt.savefig(out, dpi=300, bbox_inches="tight")
            print(f"✓ Plot saved: {out}")

        figs.append(fig)

    if show:
        plt.show()

    return figs, kappa_c_results


def main():
    parser = argparse.ArgumentParser(
        description="Log-log E vs t plot with numeric kappa_c estimation per N.")
    parser.add_argument("identifier", nargs="?", default=None,
                         help="timestamp/basename/filename of a specific experiment (default: latest)")
    parser.add_argument("--energy", choices=["best", "current"], default="best",
                         help="plot mean_best_energy (default) or mean_current_energy")
    parser.add_argument("--threshold", type=float, default=SUCCESS_THRESHOLD,
                         help=f"plateau-energy crossing threshold for kappa_c (default: {SUCCESS_THRESHOLD})")
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

    print(f"\nEstimating kappa_c via plateau-energy threshold crossing (threshold={args.threshold:g})...")
    _, kappa_c_results = plot_loglog_kappac(df, csv_file, energy_col, args.threshold,
                                             args.show, not args.no_save, None)

    print("\nkappa_c estimates:")
    for (regime, cooling_strategy, N), kc in sorted(kappa_c_results.items()):
        kc_str = f"{kc:.4f}" if kc is not None else "N/A (no bracketing crossing)"
        print(f"  regime={regime:<10} cooling={cooling_strategy:<12} N={N:<5} kappa_c ~ {kc_str}")

    if not args.show:
        print("\n(Use --show or -s flag to display interactively)")
    print("\n✓ Plot complete!")


if __name__ == "__main__":
    main()
