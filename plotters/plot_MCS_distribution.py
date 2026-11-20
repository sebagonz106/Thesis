"""
plot_MCS_distribution.py

Plots a grouped histogram (+ trend line through bin midpoints) of the number of
iterations spent solving the NPP for a single (N, b) configuration, from an
npp_TIMESTAMP_data.csv produced by either SA4NPP_phase_diagram.jl (column
`n_MCS_used`) or Exhaustive_NPP.jl (column `nodes_visited`). Whichever of the two
columns is present in the file is used automatically.

Runs that did not find a perfect partition hit their iteration cap (n_MCS_limit /
max_nodes), so their values pile up at or near the maximum observed value -- they
are NOT excluded, they simply land in the last bin(s) along with everything else.

Usage:
    python plot_MCS_distribution.py                         # fully interactive
    python plot_MCS_distribution.py npp_20260101_120000      # pick file, rest interactive
    python plot_MCS_distribution.py npp_20260101_120000_data.csv --N 27 --b 25 --group-count 20
    python plot_MCS_distribution.py --N 27 --b 25 --group-count 20 --log-scale
"""

import sys
import re
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FOLDER_PATH = Path(__file__).parent / "Experiments"
TIMESTAMP_RE = re.compile(r"npp_(\d{8}_\d{6})_data\.csv$")

# Either of these may be the "iterations to solve" column, depending on which
# generator produced the file. First one found (in this order) is used.
ITER_COL_CANDIDATES = ["n_MCS_used", "nodes_visited"]


# ==============================================================================
# FILE DISCOVERY
# ==============================================================================

def find_latest_data_csv(folder=None):
    """Find the most recent npp_*_data.csv by embedded timestamp."""
    folder = Path(folder) if folder else FOLDER_PATH
    candidates = []
    for f in folder.glob("npp_*_data.csv"):
        m = TIMESTAMP_RE.search(f.name)
        if m:
            candidates.append((m.group(1), f))
    if not candidates:
        raise FileNotFoundError(f"No npp_*_data.csv files found in {folder}")
    candidates.sort(key=lambda p: p[0])
    return candidates[-1][1]


def find_data_csv(folder=None, identifier=None):
    """Locate a single npp_*_data.csv: by identifier/timestamp/filename, or latest if None."""
    folder = Path(folder) if folder else FOLDER_PATH
    if identifier is None:
        return find_latest_data_csv(folder)
    identifier = str(identifier).strip()
    candidate = Path(identifier)
    if candidate.is_file():
        return candidate
    if identifier.endswith("_data.csv") or identifier.endswith(".csv"):
        candidate = folder / identifier
    else:
        candidate = folder / f"{identifier}_data.csv"
    if not candidate.exists():
        raise FileNotFoundError(f"No data CSV matching '{identifier}' found in {folder}")
    return candidate


def find_params_json(data_csv_path):
    """Best-effort: locate the matching npp_TIMESTAMP_params.json for extra metadata."""
    m = TIMESTAMP_RE.search(Path(data_csv_path).name)
    if not m:
        return None
    params_path = Path(data_csv_path).parent / f"npp_{m.group(1)}_params.json"
    if params_path.exists():
        try:
            with open(params_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_data(csv_file):
    """Load an npp_*_data.csv into a DataFrame and detect the iterations column."""
    df = pd.read_csv(csv_file)

    iter_col = None
    for cand in ITER_COL_CANDIDATES:
        if cand in df.columns:
            iter_col = cand
            break
    if iter_col is None:
        raise ValueError(
            f"Neither of {ITER_COL_CANDIDATES} found as a column in {csv_file}. "
            f"Columns present: {list(df.columns)}"
        )

    required = ["N", "b"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_file}: {missing}")

    return df, iter_col


# ==============================================================================
# INTERACTIVE SELECTION
# ==============================================================================

def prompt_choice(prompt, choices):
    """Print numbered choices and return the one the user picks."""
    for c in choices:
        print(f"  - {c}")
    while True:
        raw = input(prompt).strip()
        if not raw:
            continue
        for c in choices:
            if str(c) == raw:
                return c
        print(f"  '{raw}' is not one of the listed options, try again.")


def select_N(df):
    N_values = sorted(df["N"].unique())
    print("\nAvailable N values:")
    return prompt_choice("Select N (by value): ", N_values)


def select_b(df, N):
    b_values = sorted(df[df["N"] == N]["b"].unique())
    print(f"\nAvailable b values for N={N}:")
    return prompt_choice("Select b (by value): ", b_values)


def select_group_count():
    while True:
        raw = input("\nNumber of groups (bins) to use: ").strip()
        try:
            gc = int(raw)
            if gc >= 1:
                return gc
        except ValueError:
            pass
        print("  Please enter a positive integer.")


# ==============================================================================
# PLOTTING
# ==============================================================================

def build_legend_text(sub, iter_col, params):
    """Assemble the legend/info block: N, b, cooling scheme/factor, A, regime, plus extras."""
    row0 = sub.iloc[0]
    N = row0["N"]
    b = row0["b"]
    regime = row0["regime"] if "regime" in sub.columns else "n/a"

    lines = [f"N = {N}", f"b = {b}", f"regime = {regime}"]

    if "cooling_strategy" in sub.columns:
        cooling_strategy = row0["cooling_strategy"]
        lines.append(f"cooling = {cooling_strategy}")
        if cooling_strategy == "geometric" and "cooling_factor" in sub.columns:
            cf = row0["cooling_factor"]
            if pd.notna(cf):
                lines.append(f"cooling_factor = {cf:.6g}")

    if "A_prefactor" in sub.columns:
        lines.append(f"A = {row0['A_prefactor']:.4g}")

    count = len(sub)
    lines.append(f"count = {count}")

    if "success" in sub.columns:
        p_sat = sub["success"].astype(bool).mean()
        lines.append(f"P_sat = {p_sat:.3f}")

    if params:
        extra_keys = ["min_T", "n_MCS_limit", "samples"]
        for k in extra_keys:
            if k in params and params[k] is not None and k not in ("samples",):
                lines.append(f"{k} = {params[k]}")

    return "\n".join(lines)


def plot_MCS_distribution(df, iter_col, N, b, group_count, log_scale, csv_file,
                           params, show, save, output_path):
    sub = df[(df["N"] == N) & (df["b"] == b)].copy()
    if sub.empty:
        raise ValueError(f"No rows found for N={N}, b={b} in {csv_file}")

    values = sub[iter_col].to_numpy(dtype=float)
    vmax = values.max()
    vmin = values.min()

    if log_scale:
        lo = max(vmin, 1.0)  # log scale needs a strictly-positive left edge
        bin_edges = np.logspace(np.log10(lo), np.log10(max(vmax, lo)), group_count + 1)
        bin_edges[0] = min(bin_edges[0], vmin)  # make sure the lowest value is captured
    else:
        bin_edges = np.linspace(0.0, vmax, group_count + 1)

    # np.histogram is closed-open except the LAST bin, which is closed-closed
    # (includes the rightmost edge) -- exactly the behavior requested.
    counts, edges = np.histogram(values, bins=bin_edges)
    midpoints = (edges[:-1] + edges[1:]) / 2.0
    widths = np.diff(edges)

    mean_val = values.mean()
    std_val = values.std()

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(midpoints, counts, width=widths, align="center",
           color="#4C72B0", alpha=0.75, edgecolor="black", linewidth=0.6,
           label=f"{iter_col} histogram ({group_count} groups)")

    ax.plot(midpoints, counts, "o-", color="#C44E52", linewidth=1.8, markersize=5,
            label="trend (bin-midpoint line)")

    # Mean as a dashed vertical line, +/- std as a shaded band in a distinct color.
    ax.axvspan(max(mean_val - std_val, 0), mean_val + std_val,
               color="#55A868", alpha=0.18, label=f"mean ± std ({std_val:.3g})")
    ax.axvline(mean_val, color="#55A868", linestyle="--", linewidth=1.6,
               label=f"mean = {mean_val:.3g}")

    if log_scale:
        ax.set_xscale("log")

    ax.set_xlabel(iter_col, fontsize=11)
    ax.set_ylabel("count", fontsize=11)
    ax.set_title(f"NPP iterations-to-solve distribution -- N={N}, b={b}",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both" if log_scale else "major")

    legend_text = build_legend_text(sub, iter_col, params)
    handles, labels = ax.get_legend_handles_labels()
    main_legend = ax.legend(handles, labels, fontsize=8, loc="upper right")
    ax.add_artist(main_legend)
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=8,
            va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"))

    plt.tight_layout()

    match = TIMESTAMP_RE.search(Path(csv_file).name)
    timestamp = match.group(1) if match else Path(csv_file).stem

    if save:
        FOLDER_PATH.mkdir(exist_ok=True)
        logx= "_logx" if log_scale else ""
        out = output_path or FOLDER_PATH / f"MCS_dist_{timestamp}_{N}_{b}_c{group_count}{logx}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"✓ Plot saved: {out}")

    if show:
        plt.show()

    return fig


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Histogram + trend of NPP iterations-to-solve (n_MCS_used / nodes_visited)."
    )
    parser.add_argument("identifier", nargs="?", default=None,
                         help="timestamp/basename/filename/path of a specific experiment "
                              "(default: latest npp_*_data.csv in ./Experiments)")
    parser.add_argument("--N", type=int, default=None, help="N value to plot")
    parser.add_argument("--b", type=int, default=None, help="b value to plot")
    parser.add_argument("--group-count", type=int, default=None,
                         help="number of histogram groups/bins")
    parser.add_argument("--log-scale", action="store_true",
                         help="use log-spaced bins / log x-axis (default: linear)")
    parser.add_argument("--show", "-s", action="store_true", help="display interactively")
    parser.add_argument("--no-save", action="store_true", help="don't save a PNG")
    args = parser.parse_args()

    if args.show:
        import matplotlib
        matplotlib.use("TkAgg")
    else:
        import matplotlib
        matplotlib.use("Agg")

    csv_file = find_data_csv(identifier=args.identifier)
    print(f"{'Using latest experiment' if args.identifier is None else 'Loading'}: {csv_file.name}")

    df, iter_col = load_data(csv_file)
    print(f"Detected iterations column: '{iter_col}'")
    print(f"Loaded {len(df)} individual runs.")

    params = find_params_json(csv_file)

    N = args.N if args.N is not None else select_N(df)
    if N not in df["N"].unique():
        sys.exit(f"N={N} not present in {csv_file.name}")

    b = args.b if args.b is not None else select_b(df, N)
    if b not in df[df["N"] == N]["b"].unique():
        sys.exit(f"b={b} not present for N={N} in {csv_file.name}")

    group_count = args.group_count if args.group_count is not None else select_group_count()

    plot_MCS_distribution(df, iter_col, N, b, group_count, args.log_scale, csv_file,
                           params, args.show, not args.no_save, None)

    if not args.show:
        print("(Use --show or -s flag to display interactively)")
    print("\n✓ Plot complete!")


if __name__ == "__main__":
    main()
