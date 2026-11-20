#!/usr/bin/env python3
"""
plot_search_cost_vs_kappa.py

Search-cost exploration: plot the average computational cost of solving NPP
instances against kappa = b/N, to see whether the cost peaks at the phase
transition (kappa_c) and how it behaves on either side of it.

Works with EITHER algorithm family, auto-detected from the columns present in
the aggregated CSV:
    - Exhaustive_NPP.jl (CKK exact / pure KK heuristic, regime "exhaustive"/"kk"):
          plots mean_nodes_visited vs kappa  (mean CKK search-tree nodes visited
          per instance -- for pure "kk" this is trivially N-1 everywhere and the
          plot will show a flat line, which is itself the expected/correct result
          since KK does a single deterministic pass with no search at all)
    - SA4NPP_phase_diagram.jl (regime "linear"/"quadratic"/"cubic"):
          plots mean_MCS_used vs kappa  (mean Monte Carlo sweeps actually
          performed per instance, which can stop early on a "fixed" budget run
          once a perfect partition is found, or run to the n_MCS_limit cap under
          "geometric"/"logarithmic" cooling)

One curve per N (the "N" vs "b" pairs available in the aggregated CSV for that
N), restricted to whichever N values are passed via --N (default: every N
present in the data). For each N, the estimated critical kappa_c (P_sat
crossing `--pcrit`, via the same simple bracketing linear interpolation used by
plot_k_vs_N.py / plot_phase_diagram_SA.py) is drawn as a dashed vertical line in
that N's color, so it's easy to see by eye whether the cost curve peaks there.

Usage:
    python plot_search_cost_vs_kappa.py                          # latest experiment in Experiments/
    python plot_search_cost_vs_kappa.py npp_20260729_120000      # a specific run
    python plot_search_cost_vs_kappa.py path/to/data_aggregated.csv
    python plot_search_cost_vs_kappa.py --N 20 25 30             # only these N curves
    python plot_search_cost_vs_kappa.py --pcrit 0.8              # different P_sat threshold for kappa_c
    python plot_search_cost_vs_kappa.py --show                   # interactive display
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

FOLDER_PATH = Path(__file__).parent / "Experiments"
TIMESTAMP_RE = re.compile(r"npp_(.+)_aggregated\.csv$")

DEFAULT_PCRIT = 0.5

# Only these are required for every regime; the cost column itself
# (mean_nodes_visited / mean_MCS_used) is detected separately (see
# detect_cost_column), since which one exists depends on which script produced
# the CSV.
REQUIRED_COLUMNS = ["N", "kappa_actual", "P_sat", "regime"]


# ==============================================================================
# 1. FILE DISCOVERY (same convention as the other plotters in this project)
# ==============================================================================

def find_latest_aggregated_csv(folder=None):
    """
    Find the most recent npp_*_aggregated.csv file in the folder, using the
    TIMESTAMP embedded in the filename (not filesystem mtime).

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

    candidates.sort(key=lambda pair: pair[0])
    return candidates[-1][1]


def resolve_csv_path(identifier):
    """
    Resolve a user-supplied identifier to an aggregated CSV path: an explicit
    file path, a bare timestamp/basename (looked up in FOLDER_PATH), or, if
    `identifier` is None, the latest experiment in FOLDER_PATH.

    Arguments:
        identifier: str, Path, or None

    Returns:
        Path
    """
    if identifier is None:
        path = find_latest_aggregated_csv()
        print(f"Using latest experiment: {path.name}")
        return path

    candidate = Path(identifier)
    if candidate.exists():
        return candidate

    name = str(identifier).strip()
    if not name.endswith(".csv"):
        name = f"{name}_aggregated.csv"
    candidate = FOLDER_PATH / name
    if not candidate.exists():
        raise FileNotFoundError(f"No aggregated CSV matching '{identifier}' found "
                                 f"(tried '{identifier}' and '{candidate}').")
    return candidate


# ==============================================================================
# 2. kappa_c ESTIMATION (same bracketing linear interpolation as plot_k_vs_N.py
#    and plot_phase_diagram_SA.py's find_crossing, so every grapher in this
#    project agrees on how a crossing is computed)
# ==============================================================================

def estimate_kappa_critical(kappa_vals, psat_vals, pcrit=DEFAULT_PCRIT):
    """
    Estimate the kappa at which P_sat crosses `pcrit`, via linear
    interpolation between the two BRACKETING points only (no extrapolation).

    Arguments:
        kappa_vals: array-like of kappa values, any order
        psat_vals: array-like of P_sat values, same order as kappa_vals
        pcrit: the P_sat threshold to find the crossing of

    Returns:
        float or None: interpolated kappa at the crossing, or None if pcrit
        is not bracketed by the data
    """
    order = np.argsort(kappa_vals)
    kappa_vals = np.asarray(kappa_vals)[order]
    psat_vals = np.asarray(psat_vals)[order]

    for i in range(len(kappa_vals) - 1):
        y0, y1 = psat_vals[i], psat_vals[i + 1]
        if (y0 - pcrit) * (y1 - pcrit) <= 0 and y0 != y1:
            frac = (pcrit - y0) / (y1 - y0)
            return kappa_vals[i] + frac * (kappa_vals[i + 1] - kappa_vals[i])

    return None


# ==============================================================================
# 3. ALGORITHM / COST-COLUMN DETECTION
# ==============================================================================

def detect_cost_column(df):
    """
    Decide which column holds the per-(N, kappa) search-cost metric to plot,
    based on which columns are actually present -- this is how the script
    tells a CKK/KK aggregated CSV (Exhaustive_NPP.jl) apart from an SA one
    (SA4NPP_phase_diagram.jl) without needing a separate --algorithm flag.

    Arguments:
        df: aggregated DataFrame

    Returns:
        (cost_col: str, cost_label: str, title_tag: str)

    Raises:
        ValueError: if neither known cost column is present
    """
    if "mean_nodes_visited" in df.columns:
        regimes = sorted(df["regime"].unique()) if "regime" in df.columns else []
        is_pure_kk = regimes == ["kk"]
        tag = "pure KK heuristic" if is_pure_kk else "CKK / KK (Exhaustive_NPP.jl)"
        return "mean_nodes_visited", "Mean CKK search-tree nodes visited", tag
    if "mean_MCS_used" in df.columns:
        return "mean_MCS_used", "Mean Monte Carlo sweeps (MCS) performed", "Simulated Annealing (SA4NPP_phase_diagram.jl)"
    raise ValueError(
        "Neither 'mean_nodes_visited' (CKK/KK, from Exhaustive_NPP.jl) nor "
        "'mean_MCS_used' (SA, from SA4NPP_phase_diagram.jl) is present in this "
        "aggregated CSV -- don't know which search-cost metric to plot. If this "
        "is an older SA aggregated CSV saved before mean_MCS_used was added to "
        "its output, make sure the matching *_data.csv sits alongside it in the "
        "same folder (used below as a fallback to recompute it)."
    )


def ensure_cost_column(df, csv_path):
    """
    If the aggregated CSV doesn't carry a known cost column (e.g. an SA
    aggregated CSV produced before SA4NPP_phase_diagram.jl saved
    mean_MCS_used/std_MCS_used to disk), fall back to recomputing it from the
    sibling *_data.csv (same timestamp, same folder, one row per individual
    run -- always has n_MCS_used) rather than failing outright.

    Arguments:
        df: aggregated DataFrame, as loaded
        csv_path: Path to the aggregated CSV (used to locate the sibling data CSV)

    Returns:
        pd.DataFrame: df, with mean_MCS_used/std_MCS_used merged in if recovered
        (unchanged if already present, or if no fallback was possible --
        detect_cost_column will raise with a clear message in that case)
    """
    if "mean_nodes_visited" in df.columns or "mean_MCS_used" in df.columns:
        return df

    data_path = Path(str(csv_path).replace("_aggregated.csv", "_data.csv"))
    if not data_path.exists():
        return df

    data_df = pd.read_csv(data_path)
    if "n_MCS_used" not in data_df.columns:
        return df

    group_cols = [c for c in ["N", "regime", "cooling_strategy", "kappa_actual"] if c in data_df.columns]
    if "N" not in group_cols or "kappa_actual" not in group_cols:
        return df

    print(f"NOTE: {csv_path.name} has no mean_MCS_used column (pre-fix aggregated "
          f"schema) -- recomputing it from {data_path.name} instead.")
    data_df = data_df.copy()
    data_df["kappa_actual"] = data_df["kappa_actual"].round(6)
    stats = (data_df.groupby(group_cols)["n_MCS_used"]
             .agg(mean_MCS_used="mean", std_MCS_used="std")
             .reset_index())

    merge_cols = [c for c in group_cols]
    df = df.merge(stats, on=merge_cols, how="left")
    return df


# ==============================================================================
# 4. PLOTTING
# ==============================================================================

def _safe_xlim(ax, values, margin_frac=0.05, min_margin=1e-3):
    """Pad the x-axis around `values`, guarding against a zero-width range."""
    lo, hi = float(np.min(values)), float(np.max(values))
    margin = (hi - lo) * margin_frac
    if margin <= 0:
        margin = max(abs(lo) * margin_frac, min_margin)
    ax.set_xlim(lo - margin, hi + margin)


def plot_search_cost(df, cost_col, cost_label, title_tag, N_filter, pcrit,
                      csv_file, show=True, save=True, output_path=None):
    """
    Build the (nodes visited / MCS performed) vs kappa figure: one curve per
    N, one dashed vertical line per N marking its estimated kappa_c.

    Arguments:
        df: aggregated DataFrame (already has `cost_col`)
        cost_col: column to plot on the y-axis ("mean_nodes_visited" or "mean_MCS_used")
        cost_label: y-axis label / legend wording for that column
        title_tag: short string naming the algorithm family, for the figure title
        N_filter: list[int] or None -- restrict to these N values (None: all N present)
        pcrit: P_sat threshold defining kappa_c
        csv_file: Path to the source CSV (used for the output filename/timestamp)
        show: display interactively if True
        save: export to PNG if True
        output_path: explicit output PNG path (default: derived from csv_file)

    Returns:
        (fig, ax)
    """
    available_N = sorted(df["N"].unique())
    if N_filter:
        missing = [n for n in N_filter if n not in available_N]
        if missing:
            print(f"NOTE: N={missing} not found in this data (available: {available_N}) -- skipped.")
        N_values = [n for n in N_filter if n in available_N]
        if not N_values:
            raise ValueError(f"None of the requested --N values {N_filter} are present "
                              f"in this data (available: {available_N}).")
    else:
        N_values = available_N

    cmap = plt.colormaps.get_cmap("viridis")
    colors = {N: cmap(i / max(1, len(N_values) - 1)) for i, N in enumerate(N_values)}

    fig, ax = plt.subplots(figsize=(11, 7))

    no_crossing_N = []
    all_kappa = []
    for N in N_values:
        sub = df[df["N"] == N].sort_values("kappa_actual")
        if sub[cost_col].isna().all():
            print(f"NOTE: N={N} has no usable {cost_col} values -- skipped.")
            continue
        c = colors[N]
        all_kappa.append(sub["kappa_actual"].values)

        ax.plot(sub["kappa_actual"], sub[cost_col], "o-", color=c,
                label=f"N={N}", markersize=5, linewidth=1.6)

        kappa_c = estimate_kappa_critical(sub["kappa_actual"].values, sub["P_sat"].values, pcrit)
        if kappa_c is not None:
            ax.axvline(kappa_c, color=c, linestyle="--", alpha=0.6, linewidth=1.3,
                       label=f"kappa_c(N={N}) ~ {kappa_c:.3f}")
        else:
            no_crossing_N.append(N)

    ax.set_yscale("log")
    ax.set_xlabel("kappa = b / N", fontsize=12)
    ax.set_ylabel(f"{cost_label}  (log scale)", fontsize=12)
    ax.set_title(
        f"NPP Search Cost vs kappa -- {title_tag}\n"
        "Does search cost peak at the phase transition and stay there?",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="best", ncol=2)

    if all_kappa:
        _safe_xlim(ax, np.concatenate(all_kappa))

    note_lines = [f"P_sat threshold for kappa_c: {pcrit:.2f}"]
    if no_crossing_N:
        note_lines.append(f"No crossing found for N={no_crossing_N} (not drawn)")
    ax.text(0.02, 0.02, "\n".join(note_lines), transform=ax.transAxes, fontsize=8,
            verticalalignment="bottom", horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.75))

    match = TIMESTAMP_RE.search(Path(csv_file).name)
    timestamp = match.group(1) if match else Path(csv_file).stem
    plt.tight_layout()

    if save:
        if output_path is None:
            FOLDER_PATH.mkdir(exist_ok=True)
            output_path = FOLDER_PATH / f"npp_{timestamp}_search_cost_vs_kappa.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Plot saved: {output_path}")

    if show:
        plt.show()

    return fig, ax


# ==============================================================================
# 5. CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot average search cost (CKK nodes visited, or SA MCS "
                     "performed) vs kappa, one curve per N, with each N's "
                     "estimated kappa_c marked as a dashed vertical line.",
    )
    parser.add_argument(
        "csv_file", nargs="?", default=None,
        help="Path, timestamp, or basename of an aggregated CSV. Omit to use "
             "the latest experiment in Experiments/.",
    )
    parser.add_argument("--N", type=int, nargs="+", default=None,
                        help="Only plot these N values (default: every N present in the data).")
    parser.add_argument("--pcrit", type=float, default=DEFAULT_PCRIT,
                        help=f"P_sat threshold defining the critical kappa (default: {DEFAULT_PCRIT})")
    parser.add_argument("--show", "-s", action="store_true",
                        help="Display the plot interactively (requires a GUI backend)")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save the plot to PNG")
    args = parser.parse_args()

    matplotlib.use("TkAgg" if args.show else "Agg")

    csv_path = resolve_csv_path(args.csv_file)
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"Error: missing required columns in {csv_path}: {missing}", file=sys.stderr)
        sys.exit(1)

    df = ensure_cost_column(df, csv_path)
    cost_col, cost_label, title_tag = detect_cost_column(df)

    print(f"  {len(df)} rows, regime(s): {sorted(df['regime'].unique())}, "
          f"N values: {sorted(df['N'].unique())}")
    print(f"  Plotting '{cost_col}' ({cost_label}) vs kappa, algorithm: {title_tag}")

    plot_search_cost(df, cost_col, cost_label, title_tag, args.N, args.pcrit,
                      csv_path, show=args.show, save=not args.no_save)

    if not args.show:
        print("(Use --show or -s flag to display interactively)")
    print("✓ Done.")


if __name__ == "__main__":
    main()
