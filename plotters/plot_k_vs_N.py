#!/usr/bin/env python3
"""
plot_k_vs_N.py

Reproduces Figure 4 from Mertens (2000) "The Easiest Hard Problem": the
critical bit-length kappa_c(N) vs 1/N, compared to the theoretical curve.

Given one or more aggregated CSVs (produced by SA4NPP_phase_diagram.jl or
aggregate_geometric_experiment.py -- same "npp_*_aggregated.csv" schema used
throughout this project: columns N, regime, kappa_actual, P_sat, ...), this
script:

- For each N, finds the critical kappa where P_sat crosses a given threshold
  (default 0.5) by linear interpolation between the two BRACKETING data
  points (no extrapolation beyond the data -- see estimate_kappa_critical).
- Plots these critical kappa values as colored squares against 1/N.
- Overlays the theoretical curve kappa_c(N) from Eq. (32) of Mertens (2000).
- With a SINGLE input file: one color, annotated with that experiment's
  regime/prefactor/sample-count info.
- With MULTIPLE input files: one color per regime actually present in the
  data (read from each file's own "regime" column, not assumed from file
  order), so you can overlay e.g. linear/quadratic/cubic/geometric on the
  same kappa_c(N) plot to compare how the estimated threshold depends on the
  time-scaling regime.

Usage:
    python plot_k_vs_N.py                          # latest experiment in Experiments/
    python plot_k_vs_N.py path/to/data.csv         # a specific aggregated CSV
    python plot_k_vs_N.py file1.csv file2.csv ...  # overlay multiple experiments
    python plot_k_vs_N.py --pcrit 0.8              # different P_sat threshold
    python plot_k_vs_N.py --show                   # interactive display
    python plot_k_vs_N.py --help                   # show all options
"""

import argparse
import re
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

FOLDER_PATH = Path(__file__).parent / "Experiments"
TIMESTAMP_RE = re.compile(r"npp_(.+)_aggregated\.csv$")

DEFAULT_PCRIT = 0.5

# Only these are truly required -- everything else (A_prefactor, n_MCS,
# cooling_strategy, cooling_factor, ...) varies by regime/cooling strategy
# and is treated as optional throughout this script, since not every
# combination populates every column
REQUIRED_COLUMNS = ["N", "kappa_actual", "P_sat", "regime"]


# ==============================================================================
# 1. FILE DISCOVERY (same convention as plot_phase_diagram_SA.py)
# ==============================================================================

def find_latest_aggregated_csv(folder=None):
    """
    Find the most recent npp_*_aggregated.csv file in the folder, using the
    TIMESTAMP embedded in the filename (not filesystem mtime), matching the
    convention used by plot_phase_diagram_SA.py / plot_large_size_limit_SA.py.

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


# ==============================================================================
# 2. THEORY CURVE
# ==============================================================================

def kappa_c_theory(N):
    """
    Theoretical kappa_c(N) = 1 - ln(pi*N/6) / (2*N*ln(2)), Eq. (32) of
    Mertens (2000), "The Easiest Hard Problem: Number Partitioning".

    Arguments:
        N: array-like, problem size(s)

    Returns:
        array-like: theoretical critical kappa values
    """
    N = np.asarray(N, dtype=float)
    return 1.0 - np.log(np.pi * N / 6.0) / (2.0 * N * np.log(2.0))


# ==============================================================================
# 3. CRITICAL KAPPA ESTIMATION
# ==============================================================================

def estimate_kappa_critical(kappa_vals, psat_vals, pcrit=DEFAULT_PCRIT):
    """
    Estimate the kappa at which P_sat crosses `pcrit`, via linear
    interpolation between the two BRACKETING points only.

    This deliberately does NOT extrapolate beyond the available kappa range:
    if every point is above pcrit, or every point is below pcrit, there is no
    real crossing in the data and this returns None rather than guessing one
    by extending the trend line past where it was actually measured (an
    earlier version of this function did that, and it produced misleading,
    unbounded "estimates" whenever a kappa grid didn't happen to straddle the
    threshold). This mirrors find_crossing() in plot_phase_diagram_SA.py, so
    every grapher in this project reports crossings the same way.

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


def compute_kappa_c_by_N(df, pcrit=DEFAULT_PCRIT):
    """
    For each N in `df`, estimate the critical kappa (see
    estimate_kappa_critical) and collect the results into a tidy DataFrame.
    Shared by both the single-file and multi-file plotting paths so they can
    never disagree on how a crossing is computed.

    Prints a short note listing any N values where no crossing was found, so
    a sparse or badly-chosen kappa grid doesn't fail silently.

    Arguments:
        df: DataFrame with (at least) columns "N", "kappa_actual", "P_sat"
        pcrit: the P_sat threshold to find the crossing of

    Returns:
        pd.DataFrame with columns ["N", "kappa_c"], sorted by N (only rows
        where a crossing was actually found)
    """
    rows = []
    missing_N = []
    for N, group in df.groupby("N"):
        kappa_c = estimate_kappa_critical(group["kappa_actual"].values, group["P_sat"].values, pcrit)
        if kappa_c is not None:
            rows.append({"N": N, "kappa_c": kappa_c})
        else:
            missing_N.append(N)

    if missing_N:
        print(f"NOTE: no P_sat={pcrit} crossing found for N={missing_N} "
              f"(all points on one side of the threshold) -- these N values "
              f"are excluded from the kappa_c(N) plot.")

    if not rows:
        raise ValueError(f"No P_sat={pcrit} crossing found for ANY N in this data. "
                          f"Check --pcrit and the kappa range that was actually sampled.")

    return pd.DataFrame(rows).sort_values("N").reset_index(drop=True)


# ==============================================================================
# 4. INFO BOX TEXT
# ==============================================================================

def describe_experiment(df):
    """
    Build a short, human-readable description of one experiment's
    configuration, for the info box on the single-file plot. Every field is
    read defensively (only if the column exists), since not all regimes
    populate all optional columns (e.g. "geometric" data has no A_prefactor).

    Arguments:
        df: DataFrame for a single experiment

    Returns:
        str: multi-line description text
    """
    regime = df["regime"].iloc[0] if "regime" in df.columns else "unknown"
    lines = [f"Regime: {regime}"]

    if "cooling_strategy" in df.columns:
        strategies = sorted(df["cooling_strategy"].dropna().unique())
        if strategies:
            lines.append(f"Cooling strategy: {', '.join(strategies)}")

    if "A_prefactor" in df.columns:
        A_values = sorted(df["A_prefactor"].unique())
        shown = ", ".join(f"{v:g}" for v in A_values[:3]) + (", ..." if len(A_values) > 3 else "")
        lines.append(f"Prefactor A: {shown}")
    if "n_MCS" in df.columns:
        n_mcs_values = df["n_MCS"]
        if n_mcs_values.nunique() == 1:
            lines.append(f"n_MCS: {n_mcs_values.iloc[0]:g}")
        else:
            # Varies across rows (e.g. per-N MCS_per_T budgets under geometric/
            # logarithmic cooling) -- show the range instead of one arbitrary
            # row's value.
            lines.append(f"n_MCS: {n_mcs_values.min():.3g} - {n_mcs_values.max():.3g} (varies)")
    # "cooling_factor" is the current field name (only meaningful for
    # cooling_strategy="geometric"; blank/NaN otherwise). Older unmigrated
    # files may instead carry a "cooling_rate" column under its legacy name.
    cf_col = "cooling_factor" if "cooling_factor" in df.columns else (
        "cooling_rate" if "cooling_rate" in df.columns else None)
    if cf_col is not None:
        cooling_factors = sorted(df[cf_col].dropna().astype(str).unique())
        if cooling_factors:
            shown = ", ".join(cooling_factors[:3]) + (", ..." if len(cooling_factors) > 3 else "")
            lines.append(f"Cooling factor: {shown}")
    if "count" in df.columns:
        lines.append(f"Samples per point: ~{int(round(df['count'].mean()))}")

    # Fields populated by Exhaustive_NPP.jl (CKK exact / pure KK heuristic runs),
    # absent from SA's aggregated CSVs -- shown only when present, so this box
    # degrades gracefully regardless of which script produced the data.
    if "mean_nodes_visited" in df.columns:
        nv = df["mean_nodes_visited"]
        lines.append(f"Mean nodes visited: {nv.min():.3g} - {nv.max():.3g}")
    if "n_limit_hit" in df.columns and "count" in df.columns:
        total_hit = int(df["n_limit_hit"].sum())
        total_runs = int(df["count"].sum())
        if total_runs > 0:
            lines.append(f"Node-cap hits: {total_hit}/{total_runs} runs")

    return "\n".join(lines)


# ==============================================================================
# 5. PLOTTING
# ==============================================================================

def _safe_xlim(ax, values, margin_frac=0.05, min_margin=1e-3):
    """
    Set x-axis limits around `values` with a fractional margin, guarding
    against the degenerate case of a single value (or all-identical values),
    where a zero margin would make matplotlib set identical low/high xlims
    and raise a warning (or render a blank, zero-width axis).

    Arguments:
        ax: matplotlib Axes
        values: array-like of x-values that must be visible
        margin_frac: fraction of the data's range to pad on each side
        min_margin: minimum absolute margin, used when the range is zero
    """
    lo, hi = float(np.min(values)), float(np.max(values))
    margin = (hi - lo) * margin_frac
    if margin <= 0:
        margin = max(abs(lo) * margin_frac, min_margin)
    ax.set_xlim(lo - margin, hi + margin)


def plot_theory_curve(ax, N_values_all):
    """
    Plot the theoretical kappa_c(N) curve (Eq. 32) across a range that
    comfortably covers every N value being shown, in terms of 1/N.

    Arguments:
        ax: matplotlib Axes
        N_values_all: array-like of every N value that will be plotted, across
                      all experiments (used to size the theory curve's range)

    Returns:
        None (draws directly on ax)
    """
    N_min = max(2, np.min(N_values_all) - 5)
    N_max = np.max(N_values_all) + 5
    N_theory = np.linspace(N_min, N_max, 500)
    kappa_theory = kappa_c_theory(N_theory)
    ax.plot(1.0 / N_theory, kappa_theory, color="blue", linewidth=2.5,
            label=r"$\kappa_c(N) = 1 - \frac{\ln(\pi N/6)}{2N\ln 2}$", zorder=3)
    return kappa_theory


def style_axes(ax, pcrit):
    """Shared axis labels/title/grid for both single- and multi-file plots."""
    ax.set_xlabel(r"$1/N$", fontsize=14, fontweight="bold")
    ax.set_ylabel(r"$\kappa$", fontsize=14, fontweight="bold")
    ax.set_title(
        "Phase Transition in Number Partitioning\n"
        r"Critical Bit-Length $\kappa_c$ vs. Problem Size $N$" + f"  (P_sat = {pcrit:.2f})",
        fontsize=14, fontweight="bold", pad=20,
    )
    ax.grid(True, which="major", alpha=0.3, linestyle="-", linewidth=0.7)
    ax.grid(True, which="minor", alpha=0.1, linestyle=":", linewidth=0.5)


def create_single_experiment_plot(df, csv_file, pcrit, show=True, save=True, output_path=None):
    """
    Plot kappa_c(N) vs 1/N for a single experiment, against the theory curve.

    Arguments:
        df: DataFrame for one experiment (see REQUIRED_COLUMNS)
        csv_file: Path to the source CSV (used for the output filename/timestamp)
        pcrit: P_sat threshold defining the crossing
        show: display interactively if True
        save: export to PNG if True
        output_path: explicit output PNG path (default: derived from csv_file)

    Returns:
        (fig, ax)
    """
    df_crit = compute_kappa_c_by_N(df, pcrit)

    fig, ax = plt.subplots(figsize=(10, 8))
    kappa_theory = plot_theory_curve(ax, df_crit["N"].values)

    inv_N = 1.0 / np.asanyarray(df_crit["N"].values)
    ax.scatter(inv_N, df_crit["kappa_c"], color="red", marker="s", s=100,
               edgecolors="darkred", linewidths=1.5,
               label=f"Estimated critical kappa (P_sat={pcrit:.2f})", zorder=4, alpha=0.85)

    style_axes(ax, pcrit)
    _safe_xlim(ax, inv_N)

    # Get kappa_c values as plain 1D arrays
    kappa_c_vals = np.asarray(df_crit["kappa_c"]).ravel()
    kappa_theory_vals = np.asarray(kappa_theory).ravel()

    # Compute margins without concatenation
    y_min = min(float(kappa_c_vals.min()), float(kappa_theory_vals.min()))
    y_max = max(float(kappa_c_vals.max()), float(kappa_theory_vals.max()))
    y_margin = max((y_max - y_min) * 0.05, 0.005)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    ax.legend(fontsize=10, loc="best", framealpha=0.95)

    ax.text(0.98, 0.02, describe_experiment(df) + f"\nData points: {len(df_crit)}",
            transform=ax.transAxes, fontsize=9, verticalalignment="bottom",
            horizontalalignment="right", family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

    match = TIMESTAMP_RE.search(Path(csv_file).name)
    timestamp = match.group(1) if match else Path(csv_file).stem
    plt.tight_layout()

    if save:
        if output_path is None:
            FOLDER_PATH.mkdir(exist_ok=True)
            output_path = FOLDER_PATH / f"npp_{timestamp}_k_vs_N_pcrit_{pcrit:g}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Plot saved: {output_path}")

    if show:
        plt.show()

    return fig, ax


def create_multi_experiment_plot(dfs, csv_files, pcrit, show=True, save=True, output_path=None):
    """
    Overlay kappa_c(N) vs 1/N for an ARBITRARY number of experiments, each
    colored and labeled by its own "regime" column (read from the data, not
    assumed from file order/position -- this replaces a previous version of
    this function that hardcoded exactly 3 datasets as linear/quadratic/cubic
    regardless of what they actually were).

    If the same regime string appears in more than one file, they share a
    color (and a single legend entry) rather than getting a confusing
    duplicate.

    Arguments:
        dfs: list of DataFrames, one per experiment
        csv_files: list of Path, same order as dfs (used only for the output
                   filename, since there's no single source timestamp to reuse
                   when combining several experiments)
        pcrit: P_sat threshold defining the crossing
        show: display interactively if True
        save: export to PNG if True
        output_path: explicit output PNG path (default: a fresh timestamp,
                     since no single source file's timestamp is authoritative
                     for a combined figure)

    Returns:
        (fig, ax)
    """
    per_experiment = []
    for df, csv_file in zip(dfs, csv_files): 
        regime = df["regime"].iloc[0] if "regime" in df.columns else "unknown"
        # SA runs carry cooling_strategy/A_prefactor; CKK ("exhaustive") and pure
        # KK runs from Exhaustive_NPP.jl carry neither, so both axes are read
        # defensively and simply omitted from the label/legend key when absent,
        # rather than printing a misleading "A=unknown" for non-SA regimes.
        if "cooling_strategy" in df.columns:
            cooling_strategy = df["cooling_strategy"].iloc[0]
            regime_key = f"{regime}-{cooling_strategy}"
        else:
            cooling_strategy = None
            regime_key = regime
        # Distinguish SA runs that warm-start from the KK partition from those
        # that start from a uniformly random state
        if load_use_kk_init(csv_file):
            regime_key = f"{regime_key}, KK-init"
        if "A_prefactor" in df.columns:
            prefactor = df["A_prefactor"].iloc[0]
            label = f"{regime_key} (A={prefactor})"
        else:
            label = regime_key
        df_crit = compute_kappa_c_by_N(df, pcrit)
        per_experiment.append((label, df_crit))

    unique_regimes = sorted({regime for regime, _ in per_experiment})
    cmap = plt.colormaps.get_cmap("tab10")
    color_by_regime = {r: cmap(i % 10) for i, r in enumerate(unique_regimes)}

    fig, ax = plt.subplots(figsize=(10, 8))

    all_N = np.concatenate([df_crit["N"].values for _, df_crit in per_experiment])
    kappa_theory = plot_theory_curve(ax, all_N)

    plotted_regimes = set()
    all_kappa_c = []
    all_inv_N = []
    for regime, df_crit in per_experiment:
        inv_N = 1.0 / df_crit["N"].values
        all_inv_N.append(inv_N)
        all_kappa_c.append(df_crit["kappa_c"].values)

        # Only label the first occurrence of a given regime, so overlaying
        # e.g. two "geometric" files doesn't produce two identical legend rows.
        label = f"{regime} (P_sat={pcrit:.2f})" if regime not in plotted_regimes else None
        plotted_regimes.add(regime)

        ax.scatter(inv_N, df_crit["kappa_c"], color=color_by_regime[regime],
                   marker="s", s=100, edgecolors="black", linewidths=1.0,
                   label=label, zorder=4, alpha=0.85)

    style_axes(ax, pcrit)
    _safe_xlim(ax, np.concatenate(all_inv_N))
    y_all = np.concatenate(all_kappa_c + [kappa_theory])
    y_margin = max((y_all.max() - y_all.min()) * 0.05, 0.005)
    ax.set_ylim(y_all.min() - y_margin, y_all.max() + y_margin)
    ax.legend(fontsize=10, loc="best", framealpha=0.95)

    ax.text(0.98, 0.02, f"Experiments overlaid: {len(dfs)}\nRegimes: {', '.join(unique_regimes)}",
            transform=ax.transAxes, fontsize=9, verticalalignment="bottom",
            horizontalalignment="right", family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

    plt.tight_layout()

    if save:
        if output_path is None:
            FOLDER_PATH.mkdir(exist_ok=True)
            now_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = FOLDER_PATH / f"npp_multi_{now_ts}_k_vs_N_pcrit_{pcrit:g}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Plot saved: {output_path}")

    if show:
        plt.show()

    return fig, ax


# ==============================================================================
# 6. DATA LOADING
# ==============================================================================

def load_and_validate(csv_path):
    """
    Load a single aggregated CSV and check it has the columns this script
    actually needs. A_prefactor/n_MCS/cooling_rate are treated as optional
    throughout this script (see describe_experiment), so they are
    deliberately NOT required here -- requiring them would reject valid data
    from regimes that don't have them (e.g. "geometric").

    Arguments:
        csv_path: Path to a *_aggregated.csv file

    Returns:
        pd.DataFrame

    Raises:
        ValueError: if any of REQUIRED_COLUMNS is missing
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    return df

def load_use_kk_init(agg_csv_path):
    """
    Read the `use_kk_init` flag from the params.json sitting next to an
    aggregated CSV (same timestamp prefix, same directory). Allows the 
    plotting layer to distinguish SA experiments whose only difference 
    is the initial-state policy.

    Missing-field handling mirrors merge_experiments.py: params.json files
    written before the KK-init feature existed simply lack the key, and those
    older experiments were all random-init, so a missing field is reported as
    False.

    A missing params.json entirely is a different case -- we genuinely cannot
    tell -- so that returns None, and the caller leaves the label unannotated
    rather than asserting a value.

    Arguments:
        agg_csv_path: Path to an npp_*_aggregated.csv file

    Returns:
        True  -> params.json exists and use_kk_init is truthy
        False -> params.json exists but lacks the field (legacy file)
        None  -> params.json missing or unreadable
    """
    agg_csv_path = Path(agg_csv_path)
    match = TIMESTAMP_RE.search(agg_csv_path.name)
    if not match:
        return None
    params_path = agg_csv_path.with_name(f"npp_{match.group(1)}_params.json")
    if not params_path.exists():
        return None
    try:
        with open(params_path, "r", encoding="utf-8") as fh:
            params = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return bool(params.get("use_kk_init", False))


# ==============================================================================
# 7. CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Figure 4 from Mertens (2000): kappa_c(N) vs 1/N.",
    )
    parser.add_argument(
        "csv_files", nargs="*",
        help="Path(s) to aggregated CSV file(s). Zero: use the latest experiment "
             "in Experiments/. One: single-experiment plot. Two or more: overlay "
             "them on one plot, colored by regime.",
    )
    parser.add_argument("--pcrit", type=float, default=DEFAULT_PCRIT,
                        help=f"P_sat threshold defining the critical kappa (default: {DEFAULT_PCRIT})")
    parser.add_argument("--show", "-s", action="store_true",
                        help="Display the plot interactively (requires a GUI backend)")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save the plot to PNG")
    args = parser.parse_args()

    matplotlib.use("TkAgg" if args.show else "Agg")

    if args.csv_files:
        csv_paths = [Path(p) for p in args.csv_files]
        missing = [p for p in csv_paths if not p.exists()]
        if missing:
            print(f"Error: file(s) not found: {missing}", file=sys.stderr)
            sys.exit(1)
    else:
        csv_paths = [find_latest_aggregated_csv()]
        print(f"Using latest experiment: {csv_paths[0].name}")

    dfs = []
    for csv_path in csv_paths:
        print(f"Loading data from: {csv_path}")
        df = load_and_validate(csv_path)
        print(f"  {len(df)} rows, regime(s): {sorted(df['regime'].unique())}")
        dfs.append(df)

    print(f"\nEstimating critical kappa with P_sat = {args.pcrit} ...")
    if len(dfs) == 1:
        create_single_experiment_plot(dfs[0], csv_paths[0], args.pcrit,
                                       show=args.show, save=not args.no_save)
    else:
        create_multi_experiment_plot(dfs, csv_paths, args.pcrit,
                                      show=args.show, save=not args.no_save)

    if not args.show:
        print("(Use --show flag to display interactively)")
    print("✓ Done.")


if __name__ == "__main__":
    main()
