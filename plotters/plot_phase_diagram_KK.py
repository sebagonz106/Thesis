"""
plot_phase_diagram_KK.py

Phase-diagram visualization for the NPP Karmarkar-Karp family of runs.

Companion script to Exhaustive_NPP.jl. Reads a SINGLE *_aggregated.csv produced
by one run of the Julia script -- either an exact CKK run (regime="exhaustive")
or a pure KK heuristic baseline run (regime="kk"), one row per (N, kappa) -- and
plots:

    1. P_sat  (fraction of perfect partitions) vs kappa, one curve per N
    2. U_f    (mean best_energy reached) vs kappa, one curve per N

This mirrors plot_phase_diagram_SA.py's behavior exactly (same two panels, same
per-N coloring, same P_sat=0.5 crossing markers), but is adapted to the CKK/KK
schema, which has NO cooling_strategy/A_prefactor axis (those are SA-specific):
here grouping is by `regime` alone, so a single aggregated CSV normally produces
one row of panels, but a file that happens to mix "exhaustive" and "kk" rows (or
several of either) still gets one row per regime present, exactly like the SA
plotter does per (regime, cooling_strategy) pair. Exact CKK runs additionally
carry `mean_nodes_visited` / `n_limit_hit` columns (absent for pure KK, where a
single pass is always exactly N-1 steps and the node cap never applies); when
present, these are summarized in the P_sat panel's info box so you can see at a
glance whether CKK runs are prematurely capped in the hard phase.

By default this script finds and plots ONLY the most recent experiment in the
Experiments folder (identified by the timestamp embedded in its filename), NOT
a combination of every run ever saved there. Pass an explicit timestamp/filename
to plot a specific past run instead.

Usage:
    python plot_phase_diagram_KK.py                        # latest experiment only
    python plot_phase_diagram_KK.py npp_20260729_120000    # a specific run
    python plot_phase_diagram_KK.py --show                 # interactive display
    python plot_phase_diagram_KK.py --all                  # regenerate a plot for every experiment in FOLDER_PATH/
    python plot_phase_diagram_KK.py --all --show           # same, but pausing on each interactive window until closed
"""

import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FOLDER_PATH = Path(__file__).parent / "Experiments"

TIMESTAMP_RE = re.compile(r"npp_(\d{8}_\d{6})_aggregated\.csv$")

# Preferred row order when several regimes are present in the same file: exact
# CKK ("exhaustive") first as the ground-truth baseline, then the fast "kk"
# heuristic, then anything else (alphabetical) in case this schema is reused
# for a future algorithm.
REGIME_RANK = {"exhaustive": 0, "kk": 1}

MAX_N_PER_FIG = 20

def find_latest_aggregated_csv(folder=None):
    """
    Find the most recent *_aggregated.csv file in the folder, using the
    TIMESTAMP embedded in the filename (npp_YYYYMMDD_HHMMSS_aggregated.csv)
    rather than filesystem mtime, so results are stable even if files get
    copied/moved and their mtimes change.

    Arguments:
        folder: Path to search (default: FOLDER_PATH)

    Returns:
        Path: file path of the most recent npp_*_aggregated.csv

    Raises:
        FileNotFoundError: if no aggregated CSV files are found
    """
    if folder is None:
        folder = FOLDER_PATH
    folder = Path(folder)

    candidates = []
    for f in folder.glob("npp_*_aggregated.csv"):
        match = TIMESTAMP_RE.search(f.name)
        if match:
            candidates.append((match.group(1), f))

    if not candidates:
        raise FileNotFoundError(f"No npp_*_aggregated.csv files found in {folder}")

    # Timestamps are zero-padded "YYYYMMDD_HHMMSS" strings, so lexicographic
    # sort is equivalent to chronological sort.
    candidates.sort(key=lambda pair: pair[0])
    return candidates[-1][1]


def find_all_aggregated_csvs(folder=None):
    """
    Find EVERY npp_*_aggregated.csv file in the folder, using the TIMESTAMP
    embedded in each filename (not filesystem mtime, for the same stability
    reason as find_latest_aggregated_csv), sorted oldest-to-newest.

    Used by the --all / -a batch mode to regenerate a figure for every
    experiment on disk, not just the most recent one.

    Arguments:
        folder: Path to search (default: FOLDER_PATH)

    Returns:
        list[Path]: all matching files, sorted chronologically (oldest first)

    Raises:
        FileNotFoundError: if no aggregated CSV files are found
    """
    if folder is None:
        folder = FOLDER_PATH
    folder = Path(folder)

    candidates = []
    for f in folder.glob("npp_*_aggregated.csv"):
        match = TIMESTAMP_RE.search(f.name)
        if match:
            candidates.append((match.group(1), f))

    if not candidates:
        raise FileNotFoundError(f"No npp_*_aggregated.csv files found in {folder}")

    candidates.sort(key=lambda pair: pair[0])
    return [f for _, f in candidates]


def find_aggregated_csv(folder=None, identifier=None):
    """
    Locate a single aggregated CSV to load: either the one matching `identifier`
    (a timestamp, base name, or filename) or, if identifier is None, the most
    recent experiment in the folder (see find_latest_aggregated_csv).

    Arguments:
        folder: Path to search (default: FOLDER_PATH)
        identifier: optional timestamp/basename/filename to match a specific run

    Returns:
        Path

    Raises:
        FileNotFoundError: if no matching file is found
    """
    if folder is None:
        folder = FOLDER_PATH
    folder = Path(folder)

    if identifier is None:
        return find_latest_aggregated_csv(folder)

    identifier = str(identifier).strip()
    if identifier.endswith("_aggregated.csv"):
        candidate = folder / identifier
    elif identifier.endswith(".csv"):
        candidate = folder / identifier
    else:
        candidate = folder / f"{identifier}_aggregated.csv"

    if not candidate.exists():
        raise FileNotFoundError(f"No aggregated CSV matching '{identifier}' found in {folder}")
    return candidate


def load_aggregated_data(csv_file):
    """
    Load a single aggregated CSV (produced by Exhaustive_NPP.jl) into a
    DataFrame.

    Arguments:
        csv_file: Path to a *_aggregated.csv file

    Returns:
        pd.DataFrame with (at least) columns:
        N, regime, kappa_actual, b, count, P_sat, mean_Uf, std_Uf, stderr_Uf,
        mean_final_energy, std_final_energy, mean_nodes_visited, n_limit_hit

    Note:
        Unlike SA4NPP_phase_diagram.jl's output, there is no cooling_strategy
        axis here -- CKK and pure KK are each a single deterministic/exact
        procedure per instance, not a family of annealing schedules. Grouping
        in this script is therefore by `regime` alone.
    """
    required_columns = ["N", "regime", "kappa_actual", "P_sat", "mean_Uf", "stderr_Uf"]
    df = pd.read_csv(csv_file)
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_file}: {missing}")
    return df.sort_values(["regime", "N", "kappa_actual"]).reset_index(drop=True)


def find_crossing(df_subset_by_N, target=0.5, value_col="P_sat", x_col="kappa_actual"):
    """
    Estimate the kappa at which each N-curve crosses `target` (default: P_sat = 0.5),
    via linear interpolation between the two bracketing points.

    Arguments:
        df_subset_by_N: dict {N: pd.DataFrame sorted by x_col} for a single regime
        target: value to find the crossing of (default 0.5 for P_sat)
        value_col: column to interpolate on
        x_col: column to interpolate against

    Returns:
        dict {N: kappa_crossing_estimate or None if no bracketing found}
    """
    crossings = {}
    for N, df in df_subset_by_N.items():
        x = df[x_col].values
        y = df[value_col].values
        crossing = None
        for i in range(len(x) - 1):
            if (y[i] - target) * (y[i + 1] - target) <= 0 and y[i] != y[i + 1]:
                frac = (target - y[i]) / (y[i + 1] - y[i])
                crossing = x[i] + frac * (x[i + 1] - x[i])
                break
        crossings[N] = crossing
    return crossings


def _algo_info_suffix(df_group):
    """
    Build a short suffix describing CKK-specific diagnostics (mean nodes
    visited, node-cap hits) for the group label, when those columns are
    present and meaningful for this regime. Returns "" for pure KK runs
    (nodes_visited there is trivially N-1, not worth annotating) or when the
    columns are missing.

    Arguments:
        df_group: pd.DataFrame filtered to a single regime

    Returns:
        str: suffix to append to the panel title, e.g. " (node cap hit in 3/2000 runs)"
    """
    if "regime" not in df_group.columns or df_group["regime"].iloc[0] != "exhaustive":
        return ""
    if "n_limit_hit" not in df_group.columns or "count" not in df_group.columns:
        return ""
    total_hit = int(df_group["n_limit_hit"].sum())
    total_runs = int(df_group["count"].sum())
    if total_runs == 0:
        return ""
    return f"  [node cap hit in {total_hit}/{total_runs} runs]"


def plot_group_panel(ax_psat, ax_uf, df_group, group_label, colors):
    """
    Plot P_sat and U_f vs kappa for all N values within a single `regime`
    group, onto the given pair of axes. Same visual language as
    plot_phase_diagram_SA.py's plot_group_panel (P_sat=0.5 crossing markers,
    per-N coloring, log-scale U_f), minus the cooling_strategy/A_prefactor
    axis that script annotates (CKK/KK have neither).

    Arguments:
        ax_psat, ax_uf: matplotlib Axes for the P_sat and U_f panels
        df_group: pd.DataFrame filtered to a single `regime`
        group_label: str, used in titles (e.g. "exhaustive (exact CKK)")
        colors: dict {N: color} for consistent coloring across panels
    """
    by_N = {}

    for N in colors.keys():
        sub = df_group[df_group["N"] == N].sort_values("kappa_actual")
        by_N[N] = sub
        c = colors[N]

        ax_psat.plot(sub["kappa_actual"], sub["P_sat"], "o-", color=c, label=f"N={N}", markersize=4)

        stderr = sub["stderr_Uf"] if "stderr_Uf" in sub else np.zeros(len(sub))
        ax_uf.errorbar(sub["kappa_actual"], (sub["mean_Uf"]/N), yerr=(stderr/N),
                        fmt="o-", color=c, label=f"N={N}", markersize=4, capsize=2)

    # P_sat = 0.5 crossing markers (only meaningful with >=2 N curves)
    crossings = find_crossing(by_N, target=0.5, value_col="P_sat")
    for N, kc in crossings.items():
        if kc is not None:
            ax_psat.axvline(kc, color=colors[N], linestyle=":", alpha=0.4, linewidth=1)

    # valid_crossings = [v for v in crossings.values() if v is not None]

    # if len(valid_crossings) >= 2:
    #     spread = max(valid_crossings) - min(valid_crossings)
    #     note = f"kappa_c estimates span {spread:.3f} across N\n(drift => threshold not yet converged)"
    # elif len(valid_crossings) == 1:
    #     note = f"kappa_c ~ {valid_crossings[0]:.3f} (only one N has a bracketed crossing)"
    # else:
    #     note = "No P_sat=0.5 crossing found in this kappa range"

    # ax_psat.text(0.02, 0.02, note, transform=ax_psat.transAxes, fontsize=8,
    #              verticalalignment="bottom", horizontalalignment="left",
    #              bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))

    info_suffix = _algo_info_suffix(df_group)
    ax_psat.set_title(f"P_sat vs kappa -- {group_label}{info_suffix}", fontsize=12, fontweight="bold")
    ax_psat.set_xlabel("kappa = b / N", fontsize=11)
    ax_psat.set_ylabel("P_sat", fontsize=11)
    ax_psat.set_ylim(-0.05, 1.05)
    ax_psat.grid(True, alpha=0.3)
    ax_psat.axhline(0.5, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
    ax_psat.legend(fontsize=8, loc="best")

    ax_uf.set_title(f"U_f vs kappa -- {group_label}", fontsize=12, fontweight="bold")
    ax_uf.set_xlabel("kappa = b / N", fontsize=11)
    ax_uf.set_ylabel("U_f  (mean best_energy normalized by N)", fontsize=11)
    ax_uf.set_yscale("log")
    ax_uf.grid(True, alpha=0.3, which="both")
    ax_uf.legend(fontsize=8, loc="best")


def _regime_display_name(regime):
    """Human-readable label for a regime tag, for panel titles / figure suptitle."""
    return {
        "exhaustive": "exhaustive (exact CKK)",
        "kk": "kk (pure KK heuristic)",
    }.get(regime, regime)


def create_phase_diagram_figure(df, csv_file, show=True, save=True, output_path=None):
    """
    Build the figure for a single experiment: one row per `regime` present in
    `df` (normally just one, since one Julia run covers a single algorithm --
    but a file combining "exhaustive" and "kk" rows is supported too), two
    columns (P_sat, U_f), one color per N.

    Arguments:
        df: aggregated DataFrame for a single experiment (see load_aggregated_data)
        csv_file: Path to the source CSV (used to derive the output filename/timestamp)
        show: display interactively if True
        save: export to PNG if True
        output_path: explicit output PNG path (default: derived from csv_file's timestamp)

    Returns:
        (fig, axes)
    """
    regimes = sorted(df["regime"].unique(), key=lambda r: (REGIME_RANK.get(r, 99), r))

    all_N = sorted(df["N"].unique())
    if len(all_N) > MAX_N_PER_FIG:
        temp = [all_N[0]]
        for i, N in enumerate(all_N):
            if i / len(all_N) > len(temp) / MAX_N_PER_FIG:
                temp.append(N)
        all_N = temp
        
    cmap = plt.colormaps.get_cmap("viridis")
    colors = {N: cmap(i / max(1, len(all_N) - 1)) for i, N in enumerate(all_N)}

    n_rows = len(regimes)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5.5 * n_rows), squeeze=False)

    for row, regime in enumerate(regimes):
        df_group = df[df["regime"] == regime]
        plot_group_panel(axes[row][0], axes[row][1], df_group, _regime_display_name(regime), colors)

    match = TIMESTAMP_RE.search(Path(csv_file).name)
    timestamp = match.group(1) if match else Path(csv_file).stem

    fig.suptitle(f"NPP Karmarkar-Karp Phase Diagram -- experiment {timestamp}",
                 fontsize=15, fontweight="bold", y=1.0)
    plt.tight_layout()

    if save:
        if output_path is None:
            FOLDER_PATH.mkdir(exist_ok=True)
            output_path = FOLDER_PATH / f"npp_{timestamp}_phase_diagram_KK.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Plot saved: {output_path}")

    if show:
        plt.show()

    return fig, axes


def process_one_experiment(csv_file, show_plot):
    """
    Load and plot a single experiment's aggregated CSV, printing the same
    summary/status lines used by the single-experiment code path. Shared by
    both the default (latest-only) and --all (batch) modes so their console
    output and figure-generation logic never drift apart.

    Arguments:
        csv_file: Path to a *_aggregated.csv file
        show_plot: bool - display interactively if True (blocks until closed)

    Returns:
        (fig, axes)
    """
    df = load_aggregated_data(csv_file)
    print(f"\nLoaded {len(df)} (N, kappa) points from this experiment:")
    print(f"  regime(s): {sorted(df['regime'].unique())}")
    print(f"  N values: {sorted(df['N'].unique())}")
    print(f"  kappa range: [{df['kappa_actual'].min():.3f}, {df['kappa_actual'].max():.3f}]")

    print("\nCreating phase-diagram figure...")
    return create_phase_diagram_figure(df, csv_file, show=show_plot, save=True)


def main():
    show_plot = "--show" in sys.argv or "-s" in sys.argv
    batch_mode = "--all" in sys.argv or "-a" in sys.argv

    if show_plot:
        import matplotlib
        matplotlib.use("TkAgg")
    else:
        import matplotlib
        matplotlib.use("Agg")

    if batch_mode:
        csv_files = find_all_aggregated_csvs()
        print(f"Found {len(csv_files)} experiment(s) in {FOLDER_PATH}:")
        for f in csv_files:
            print(f"  - {f.name}")

        for i, csv_file in enumerate(csv_files, start=1):
            print(f"\n{'='*80}\n[{i}/{len(csv_files)}] {csv_file.name}\n{'='*80}")
            fig, _ = process_one_experiment(csv_file, show_plot)
            plt.close(fig)

        print(f"\nBatch complete: {len(csv_files)} plot(s) generated.")
        return

    csv_args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    identifier = csv_args[0] if csv_args else None

    csv_file = find_aggregated_csv(identifier=identifier)
    if identifier is None:
        print(f"Using latest experiment: {csv_file.name}")
    else:
        print(f"Loading: {csv_file.name}")

    process_one_experiment(csv_file, show_plot)

    if not show_plot:
        print("(Use --show or -s flag to display interactively)")
    print("\n✓ Plot complete!")


if __name__ == "__main__":
    main()
