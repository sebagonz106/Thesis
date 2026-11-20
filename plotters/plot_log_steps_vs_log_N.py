#!/usr/bin/env python3
"""
plot_log_steps_vs_log_N.py

Checks the exponential-growth-of-search-cost claim from Gent & Walsh (1996),
"Phase Transitions and Annealed Theories: Number Partitioning as a Case
Study" (Section 5): at the phase-transition value kappa_c, CKK's search cost
(nodes visited) should grow as approximately 2^(0.85 N).

Works on EITHER kind of aggregated CSV produced in this project:
    - Exhaustive_NPP.jl,   regime="exhaustive" (CKK)  -> column mean_nodes_visited
    - SA4NPP_phase_diagram.jl, regime="linear"/... (SA) -> column mean_MCS_used
and auto-detects which one it is being handed from the columns present (same
defensive-column-reading convention as plot_k_vs_N.py).

Method, for each N present in the data:
    1. Estimate kappa_c(N) the same way plot_k_vs_N.py does: linear
       interpolation of P_sat against kappa_actual, bracketing the P_sat=pcrit
       crossing (see estimate_kappa_critical).
    2. Estimate the search cost AT that kappa_c by a second, independent
       linear interpolation -- this time of the search-cost metric
       (mean_nodes_visited or mean_MCS_used) against kappa_actual, bracketing
       kappa_c between the kappa value immediately below and immediately
       above it (see interpolate_metric_at_kappa).
    3. Plot ln(metric) against N*ln(2). Since metric ~ 2^(c*N) <=> ln(metric)
       = c * N * ln(2), the slope of THIS plot is exactly the growth-rate
       exponent c -- no extra log-base bookkeeping needed.

For CKK ("exhaustive") data, an OLS straight-line fit is overlaid and its
slope is reported (compare to the paper's claimed ~0.85). For SA data, the
growth law is not known a priori, so we deliberately do NOT fit a line --
consecutive (N, metric) points are simply joined to let the shape of the
curve speak for itself.

Usage:
    python plot_log_steps_vs_log_N.py                          # latest experiment in Experiments/
    python plot_log_steps_vs_log_N.py path/to/data.csv         # a specific aggregated CSV
    python plot_log_steps_vs_log_N.py ckk.csv sa.csv           # overlay CKK and SA on one plot
    python plot_log_steps_vs_log_N.py --pcrit 0.8              # different P_sat threshold for kappa_c
    python plot_log_steps_vs_log_N.py --show                   # interactive display
    python plot_log_steps_vs_log_N.py --help                   # show all options
"""

import argparse
import re
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

FOLDER_PATH = Path(__file__).parent / "Experiments"
TIMESTAMP_RE = re.compile(r"npp_(.+)_aggregated\.csv$")

DEFAULT_PCRIT = 0.5
PAPER_CKK_SLOPE = 0.85   # Gent & Walsh (1996), Section 5: worst-case CKK cost ~ 2^(0.85 N)

# Only these are truly required -- the search-cost column itself
# (mean_nodes_visited OR mean_MCS_used) is checked separately in
# get_metric_column, since which one is present tells us CKK vs SA.
REQUIRED_COLUMNS = ["N", "kappa_actual", "P_sat", "regime"]


# ==============================================================================
# 1. FILE DISCOVERY (same convention as plot_k_vs_N.py)
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


# ==============================================================================
# 2. ALGORITHM / METRIC DETECTION
# ==============================================================================

def get_metric_column(df):
    """
    Decide which search-cost column this aggregated CSV carries, and whether
    that makes it a CKK (exact branch-and-bound) or SA (simulated annealing)
    experiment. Exhaustive_NPP.jl and SA4NPP_phase_diagram.jl never populate
    both columns in the same file, so presence of one is a reliable
    discriminator (mirrors the defensive column-reading used throughout
    plot_k_vs_N.py).

    Arguments:
        df: DataFrame for a single experiment

    Returns:
        (metric_col: str, is_ckk: bool)

    Raises:
        ValueError: if neither known search-cost column is present
    """
    if "mean_nodes_visited" in df.columns:
        return "mean_nodes_visited", True
    if "mean_MCS_used" in df.columns:
        return "mean_MCS_used", False
    raise ValueError(
        "Neither 'mean_nodes_visited' (CKK) nor 'mean_MCS_used' (SA) found in "
        "this aggregated CSV -- is it really from Exhaustive_NPP.jl or "
        "SA4NPP_phase_diagram.jl?"
    )


# ==============================================================================
# 3. CRITICAL KAPPA ESTIMATION (mirrors plot_k_vs_N.py exactly)
# ==============================================================================

def estimate_kappa_critical(kappa_vals, psat_vals, pcrit=DEFAULT_PCRIT):
    """
    Estimate the kappa at which P_sat crosses `pcrit`, via linear
    interpolation between the two BRACKETING points only. Does NOT
    extrapolate beyond the available kappa range: if every point is on one
    side of pcrit, returns None rather than guessing (see plot_k_vs_N.py's
    function of the same name, which this mirrors exactly so both scripts
    agree on where kappa_c sits).

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


def interpolate_metric_at_kappa(kappa_vals, metric_vals, kappa_target):
    """
    Estimate the search-cost metric at kappa_target by a SIMPLE linear
    interpolation between the kappa value immediately BELOW kappa_target and
    the one immediately ABOVE it -- i.e. the straight line through
    (k1, metric1) and (k2, metric2), evaluated at kappa_target, exactly as
    requested: no fitting, no extrapolation beyond the bracketing pair.

    If kappa_target falls outside [min(kappa_vals), max(kappa_vals)], or
    exactly on a sampled kappa, this degrades gracefully (returns None for
    out-of-range; returns the exact sampled value for an exact match).

    Arguments:
        kappa_vals: array-like of sampled kappa values for one N, any order
        metric_vals: array-like, same order/length as kappa_vals
        kappa_target: the kappa at which to estimate the metric (e.g. kappa_c)

    Returns:
        float or None: interpolated metric value, or None if kappa_target is
        not bracketed by the data
    """
    order = np.argsort(kappa_vals)
    kappa_vals = np.asarray(kappa_vals)[order]
    metric_vals = np.asarray(metric_vals)[order]

    # Exact match (within floating-point noise) -- no interpolation needed.
    exact = np.isclose(kappa_vals, kappa_target)
    if exact.any():
        return float(metric_vals[exact][0])

    if kappa_target < kappa_vals[0] or kappa_target > kappa_vals[-1]:
        return None

    i = np.searchsorted(kappa_vals, kappa_target) - 1
    i = max(0, min(i, len(kappa_vals) - 2))
    k1, k2 = kappa_vals[i], kappa_vals[i + 1]
    m1, m2 = metric_vals[i], metric_vals[i + 1]
    frac = (kappa_target - k1) / (k2 - k1)
    return float(m1 + frac * (m2 - m1))


def compute_metric_at_kappa_c(df, metric_col, pcrit=DEFAULT_PCRIT):
    """
    For each N in `df`: estimate kappa_c (P_sat crossing), then interpolate
    the search-cost metric at that kappa_c (bracketing kappa values, see
    interpolate_metric_at_kappa). Collects the results into a tidy DataFrame.

    Prints a short note listing any N values that had to be excluded (no
    P_sat crossing, or kappa_c fell outside the sampled kappa range for that
    N), so a sparse or badly-chosen kappa grid doesn't fail silently.

    Arguments:
        df: DataFrame with columns "N", "kappa_actual", "P_sat", metric_col
        metric_col: name of the search-cost column to interpolate
        pcrit: the P_sat threshold defining kappa_c

    Returns:
        pd.DataFrame with columns ["N", "kappa_c", "metric"], sorted by N
    """
    rows = []
    skipped = []
    for N, group in df.groupby("N"):
        kappa_c = estimate_kappa_critical(group["kappa_actual"].values, group["P_sat"].values, pcrit)
        if kappa_c is None:
            skipped.append((N, "no P_sat crossing"))
            continue
        metric = interpolate_metric_at_kappa(group["kappa_actual"].values, group[metric_col].values, kappa_c)
        if metric is None:
            skipped.append((N, "kappa_c outside sampled kappa range"))
            continue
        rows.append({"N": N, "kappa_c": kappa_c, "metric": metric})

    if skipped:
        print(f"NOTE: excluded N values from the {metric_col} plot: "
              + ", ".join(f"N={n} ({why})" for n, why in skipped))

    if not rows:
        raise ValueError(f"Could not estimate {metric_col} at kappa_c for ANY N in this data. "
                          f"Check --pcrit and the kappa range that was actually sampled.")

    return pd.DataFrame(rows).sort_values("N").reset_index(drop=True)


# ==============================================================================
# 4. INFO BOX TEXT
# ==============================================================================

def describe_experiment(df, metric_col, is_ckk):
    """
    Build a short, human-readable description of one experiment's
    configuration, for the info box. Every field is read defensively (only
    if the column exists), mirroring plot_k_vs_N.py's describe_experiment.

    Arguments:
        df: DataFrame for a single experiment
        metric_col: the search-cost column being plotted
        is_ckk: True for CKK ("exhaustive") data, False for SA

    Returns:
        str: multi-line description text
    """
    regime = df["regime"].iloc[0] if "regime" in df.columns else "unknown"
    algo = "CKK (exact branch & bound)" if is_ckk else "SA (simulated annealing)"
    lines = [f"Algorithm: {algo}", f"Regime: {regime}", f"Metric: {metric_col}"]

    if not is_ckk and "cooling_strategy" in df.columns:
        strategies = sorted(df["cooling_strategy"].dropna().unique())
        if strategies:
            lines.append(f"Cooling strategy: {', '.join(strategies)}")
    if not is_ckk and "A_prefactor" in df.columns:
        A_values = sorted(df["A_prefactor"].unique())
        shown = ", ".join(f"{v:g}" for v in A_values[:3]) + (", ..." if len(A_values) > 3 else "")
        lines.append(f"Prefactor A: {shown}")
    if is_ckk and "n_limit_hit" in df.columns and "count" in df.columns:
        total_hit = int(df["n_limit_hit"].sum())
        total_runs = int(df["count"].sum())
        if total_runs > 0:
            lines.append(f"Node-cap hits: {total_hit}/{total_runs} runs")
    if "count" in df.columns:
        lines.append(f"Samples per point: ~{int(round(df['count'].mean()))}")

    return "\n".join(lines)


# ==============================================================================
# 5. PLOTTING
# ==============================================================================

def _safe_xlim(ax, values, margin_frac=0.05, min_margin=1e-3):
    """
    Set x-axis limits around `values` with a fractional margin, guarding
    against the degenerate case of a single value (or all-identical values).

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


def fit_ckk_line(x, y):
    """
    OLS straight-line fit y = slope*x + intercept (via np.polyfit, degree 1),
    used only for CKK data, where the paper predicts metric ~ 2^(0.85 N) i.e.
    a straight line in this (N*ln2, ln(metric)) representation.

    Arguments:
        x: array-like, N*ln(2) values
        y: array-like, ln(metric) values, same length as x

    Returns:
        (slope: float, intercept: float)
    """
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def style_axes(ax, pcrit):
    """Shared axis labels/title/grid for the plot."""
    ax.set_xlabel(r"$N \cdot \ln 2$", fontsize=14, fontweight="bold")
    ax.set_ylabel(r"$\ln(\mathrm{search\ cost})$", fontsize=14, fontweight="bold")
    ax.set_title(
        "Search Cost at the Phase Transition vs. Problem Size\n"
        r"$\ln(\mathrm{nodes\_visited}\ /\ \mathrm{MCS\_used})$ at $\kappa_c$"
        + f"  (P_sat = {pcrit:.2f})",
        fontsize=14, fontweight="bold", pad=20,
    )
    ax.grid(True, which="major", alpha=0.3, linestyle="-", linewidth=0.7)
    ax.grid(True, which="minor", alpha=0.1, linestyle=":", linewidth=0.5)


def plot_one_experiment(ax, df, csv_file, pcrit, color, label_prefix=""):
    """
    Add one experiment's (N*ln2, ln(metric)) points to `ax`: a CKK dataset
    gets a scatter plus an OLS line (slope annotated in the legend), an SA
    dataset gets its consecutive points simply joined so the unknown-shape
    curve is visible without imposing a model on it.

    Arguments:
        ax: matplotlib Axes to draw on
        df: DataFrame for one experiment
        csv_file: Path to the source CSV (used only in error messages)
        pcrit: P_sat threshold defining kappa_c
        color: matplotlib color for this experiment's points/line
        label_prefix: text prepended to this experiment's legend label
                      (e.g. to distinguish several SA regimes on one plot)

    Returns:
        pd.DataFrame: the per-N (N, kappa_c, metric) table that was plotted,
        with an added "ln_metric" column, for use in axis-limit bookkeeping
    """
    metric_col, is_ckk = get_metric_column(df)
    df_metric = compute_metric_at_kappa_c(df, metric_col, pcrit)
    df_metric["ln_metric"] = np.log(df_metric["metric"])

    x = df_metric["N"].values * np.log(2.0)
    y = df_metric["ln_metric"].values

    if is_ckk:
        slope, intercept = fit_ckk_line(x, y)
        label = f"{label_prefix}CKK: slope={slope:.3f} (paper: {PAPER_CKK_SLOPE})"
        ax.scatter(x, y, color=color, marker="o", s=70, edgecolors="black",
                   linewidths=1.0, zorder=4, alpha=0.85)
        x_line = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_line, slope * x_line + intercept, color=color, linewidth=2.0,
                linestyle="-", label=label, zorder=3)
    else:
        regime = df["regime"].iloc[0] if "regime" in df.columns else "SA"
        label = f"{label_prefix}SA ({regime})"
        ax.plot(x, y, color=color, marker="s", markersize=7,
                markeredgecolor="black", markeredgewidth=1.0,
                linewidth=1.8, linestyle="-", label=label, zorder=4, alpha=0.85)

    return df_metric


def create_plot(dfs, csv_files, pcrit, show=True, save=True, output_path=None):
    """
    Plot ln(search cost) at kappa_c against N*ln(2), for one or more
    experiments overlaid on the same axes (CKK and/or SA).

    Arguments:
        dfs: list of DataFrames, one per experiment
        csv_files: list of Path, same order as dfs (used for output filename
                   and the single-experiment info box)
        pcrit: P_sat threshold defining kappa_c
        show: display interactively if True
        save: export to PNG if True
        output_path: explicit output PNG path (default: derived timestamp)

    Returns:
        (fig, ax)
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = plt.colormaps.get_cmap("tab10")
    all_metric_tables = []
    for i, (df, csv_file) in enumerate(zip(dfs, csv_files)):
        label_prefix = f"[{Path(csv_file).stem}] " if len(dfs) > 1 else ""
        df_metric = plot_one_experiment(ax, df, csv_file, pcrit, cmap(i % 10), label_prefix)
        all_metric_tables.append(df_metric)

    style_axes(ax, pcrit)

    all_x = np.concatenate([t["N"].values * np.log(2.0) for t in all_metric_tables])
    all_y = np.concatenate([t["ln_metric"].values for t in all_metric_tables])
    _safe_xlim(ax, all_x)
    y_margin = max((all_y.max() - all_y.min()) * 0.05, 0.05)
    ax.set_ylim(all_y.min() - y_margin, all_y.max() + y_margin)

    ax.legend(fontsize=10, loc="best", framealpha=0.95)

    if len(dfs) == 1:
        metric_col, is_ckk = get_metric_column(dfs[0])
        info_text = describe_experiment(dfs[0], metric_col, is_ckk) + f"\nData points: {len(all_metric_tables[0])}"
    else:
        info_text = f"Experiments overlaid: {len(dfs)}"
    ax.text(0.98, 0.02, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom", horizontalalignment="right",
            family="monospace", bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

    plt.tight_layout()

    if save:
        if output_path is None:
            FOLDER_PATH.mkdir(exist_ok=True)
            if len(csv_files) == 1:
                match = TIMESTAMP_RE.search(Path(csv_files[0]).name)
                timestamp = match.group(1) if match else Path(csv_files[0]).stem
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = FOLDER_PATH / f"npp_{timestamp}_log_steps_vs_N_pcrit_{pcrit:g}.png"
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
    needs: the shared REQUIRED_COLUMNS plus at least one of the two known
    search-cost columns (checked via get_metric_column).

    Arguments:
        csv_path: Path to a *_aggregated.csv file

    Returns:
        pd.DataFrame

    Raises:
        ValueError: if any required column, or both search-cost columns, are missing
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    get_metric_column(df)  # raises ValueError with a clear message if neither is present
    return df


# ==============================================================================
# 7. CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot ln(search cost) at kappa_c vs N*ln(2), for CKK (nodes_visited) "
                    "or SA (MCS_used) aggregated NPP experiments.",
    )
    parser.add_argument(
        "csv_files", nargs="*",
        help="Path(s) to aggregated CSV file(s). Zero: use the latest experiment "
             "in Experiments/. One: single-experiment plot. Two or more: overlay "
             "them on one plot (e.g. a CKK run and an SA run).",
    )
    parser.add_argument("--pcrit", type=float, default=DEFAULT_PCRIT,
                        help=f"P_sat threshold defining kappa_c (default: {DEFAULT_PCRIT})")
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
        metric_col, is_ckk = get_metric_column(df)
        print(f"  {len(df)} rows, regime(s): {sorted(df['regime'].unique())}, "
              f"metric: {metric_col} ({'CKK' if is_ckk else 'SA'})")
        dfs.append(df)

    print(f"\nEstimating kappa_c (P_sat={args.pcrit}) and interpolating search cost ...")
    create_plot(dfs, csv_paths, args.pcrit, show=args.show, save=not args.no_save)

    if not args.show:
        print("(Use --show flag to display interactively)")
    print("✓ Done.")


if __name__ == "__main__":
    main()
