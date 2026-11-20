#!/usr/bin/env python3
"""
merge_experiments.py

Merges several small NPP phase-diagram experiments (e.g. separate runs that
each covered a different block of N values, run independently so that a
mistake or a mistuned parameter in one block doesn't cost you a larger run)
into a single consolidated experiment, written out under a fresh timestamp
-- the timestamp of the moment the merge itself is performed, not any of
the source experiments.

----------------------------------------------------------------------------
TWO EXPERIMENT FAMILIES
----------------------------------------------------------------------------
This script merges experiments from either algorithm family produced by this
project, auto-detected per experiment from its params.json:

    "sa"     -- SA4NPP_phase_diagram.jl (simulated annealing). Identified by
                the presence of a "cooling_strategy" key. regime is one of
                "linear"/"quadratic"/"cubic" (the A*N^z budget exponent);
                cooling_strategy is one of "fixed"/"geometric"/"logarithmic".

    "search" -- Exhaustive_NPP.jl (exact CKK branch-and-bound, or the pure KK
                differencing heuristic). Identified by the presence of an
                "algorithm" key (no "cooling_strategy" key). regime is
                "exhaustive" (algorithm="ckk") or "kk" (algorithm="kk"); there
                is no cooling_strategy axis for either.

A merge is only meaningful WITHIN one family -- "nodes visited" and "Monte
Carlo sweeps performed" are not the same kind of quantity, and mixing them
into one aggregated.csv would silently corrupt it. Every experiment selected
for one merge run must therefore belong to the SAME family; the merge aborts
with a clear error (listing which timestamp is which family) otherwise. A
params.json with neither "cooling_strategy" nor "algorithm" is the old
pre-cooling-strategy schema and is rejected with a pointer to
migrate_legacy_data.py, exactly as before.

Everything below the file-format sections generalizes over both families via
a small set of per-family tables/branches (headers, the "which fields must be
identical" list, the winner tie-break key, and which extra per-point
statistic gets recomputed) -- the discovery/tie-break/recompute/CLI machinery
itself is shared code, unchanged in spirit from the original SA-only script.

----------------------------------------------------------------------------
WHAT "MERGE" MEANS FOR EACH FILE
----------------------------------------------------------------------------
params.json
    Fields required to be IDENTICAL across every source experiment (the
    merge aborts otherwise, since these describe the physical experiment
    itself, not a per-batch choice):
        sa:     regime, cooling_strategy, seed
        search: regime, algorithm, seed
    Fields combined numerically across source experiments:
        sa:     A_prefactor, cooling_factor, n_MCS_limit : MAX
                min_T, samples                           : MIN
                T0        : the heuristic string, if any source used it (it
                            takes precedence over a numeric T0 wherever both
                            appear); otherwise the MIN of the numeric values
                            present
        search: max_nodes : MAX (ignored / stays null for algorithm="kk",
                            which has no node cap to begin with)
                samples   : MIN
    elapsed_time_seconds : SUM across source experiments (both families)
    N_values, b_values, kappa_values, n_individual_runs : derived from the
                merged aggregated/data results themselves (see below), not
                copied from any one source's params.json (both families)

aggregated.csv
    - One row per (N, b) pair found across ALL source experiments.
    - When the same (N, b) pair appears in more than one source experiment,
      only ONE of them survives in the merge; the other's rows for that
      exact pair are discarded entirely (from both aggregated.csv AND
      data.csv -- see below). The winner is chosen by, in order:
          sa:     1. highest P_sat
                  2. lowest cooling_factor
                  3. lowest A_prefactor
                  4. lowest mean_Uf      (mean best_energy)
                  5. lowest mean_final_energy
                  6. lowest n_MCS
                  7. arbitrary (first encountered), if every one of the
                     above ties
          search: 1. highest P_sat
                  2. lowest fraction of runs that hit the node cap
                     (n_limit_hit / count) -- fewer capped runs means a more
                     reliable (less truncated) exact CKK result; always 0 for
                     "kk", which never hits a cap, so this step is a no-op
                     there
                  3. lowest mean_nodes_visited (cheaper search reaching the
                     same P_sat -- for "kk" this is essentially always N-1
                     and so is also a no-op there)
                  4. lowest mean_Uf
                  5. lowest mean_final_energy
                  6. arbitrary (first encountered), if every one of the
                     above ties
      All numeric comparisons are rounded to SIG_FIGS significant figures
      first, to avoid floating-point noise from manufacturing a spurious
      "winner" out of what is really a tie.
    - The winning row's aggregated STATISTICS are not simply copied over --
      they are RECOMPUTED from the merged data.csv rows that survive for
      that (N, b) pair:
          sa:     P_sat, mean_Uf, std_Uf, stderr_Uf, mean_final_energy,
                  std_final_energy, mean_MCS_used, std_MCS_used, count
          search: P_sat, mean_Uf, std_Uf, stderr_Uf, mean_final_energy,
                  std_final_energy, mean_nodes_visited, n_limit_hit, count
      In the ordinary case (nothing filtered) this reproduces the source
      numbers exactly; it also guarantees, by construction, that running
      aggregate_results-equivalent logic on the merged data.csv reproduces
      the merged aggregated.csv exactly, which is the self-consistency
      property this script is required to preserve.
      The winning row's PARAMETERS are copied verbatim from the winning
      source, since those describe how that (N, b) pair was produced and
      aren't statistics to recompute:
          sa:     regime, cooling_strategy, A_prefactor, cooling_factor,
                  n_MCS, kappa_actual
          search: regime, kappa_actual

data.csv
    - For every (N, b) pair, only the individual-run rows from the WINNING
      source experiment (per the aggregated.csv tie-break above) are
      included -- never a mix of two experiments' rows for the same pair.
    - Ordered first by N, then by b (matching the convention already used
      by the Julia scripts), with each (N, b) group's rows kept in their
      original relative order from the winning source file.

----------------------------------------------------------------------------
USAGE
----------------------------------------------------------------------------
    python merge_experiments.py [directory] [options]

    directory   Folder containing the npp_*_{data,aggregated,params} files
                to merge (default: the directory this script itself lives
                in -- so dropping this script inside e.g.
                "Experiments/quadratic1/" and running it there merges
                whatever experiments are in that folder, with no knowledge
                of any wider directory structure).

Options:
    --timestamps TS [TS ...]
                Merge only these specific experiments instead of every
                experiment found in `directory`. Accepts either form, e.g.
                "npp_20260801_204632" or bare "20260801_204632".
    --dry-run   Report what would be merged (and which experiment wins each
                (N, b) pair) without writing any output files.

By default (no --timestamps), EVERY experiment found in `directory` is
merged -- the family check plus the regime/cooling_strategy|algorithm/seed
equality check is what keeps this safe: unrelated experiments accidentally
left in the same folder will cause the merge to abort with a clear error
rather than silently blend incompatible physics (or incompatible
algorithms) together.
"""

import argparse
import csv
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

TIMESTAMP_RE = re.compile(r"npp_(\d{8}_\d{6})_(data|aggregated|params)\.(csv|json)$")
HEURISTIC_T0 = "heuristic (sum(numbers)/(4N))"

# Significant figures used when rounding numeric values before comparing
# them for tie-breaking or equality -- guards against floating-point noise
# (e.g. 0.8999999999999999 vs 0.9) manufacturing a spurious winner or a
# spurious mismatch out of what is really the same value.
SIG_FIGS = 6

FAMILY_SA = "sa"
FAMILY_SEARCH = "search"  # Exhaustive_NPP.jl: exact CKK (regime="exhaustive") or pure KK (regime="kk")

# ------------------------------------------------------------------------
# Per-family file schemas. Keeping these as plain tables (rather than
# scattering family-specific field lists through the functions below) is
# what makes adding a third family later a matter of adding one more entry
# to each of these, plus one more branch in sort_key_for_winner /
# recompute_aggregated_row / build_merged_params.
# ------------------------------------------------------------------------

DATA_HEADER = {
    FAMILY_SA: [
        "N", "b", "kappa_requested", "kappa_actual", "regime", "cooling_strategy",
        "A_prefactor", "cooling_factor", "n_MCS", "n_MCS_used", "T0",
        "best_energy", "final_energy", "success", "seed",
    ],
    FAMILY_SEARCH: [
        "N", "b", "kappa_requested", "kappa_actual", "regime",
        "nodes_visited", "max_nodes", "limit_hit",
        "best_energy", "final_energy", "success", "seed",
    ],
}

AGG_HEADER = {
    FAMILY_SA: [
        "N", "regime", "cooling_strategy", "kappa_actual", "b", "A_prefactor",
        "cooling_factor", "n_MCS", "mean_MCS_used", "std_MCS_used", "count",
        "P_sat", "mean_Uf", "std_Uf", "stderr_Uf",
        "mean_final_energy", "std_final_energy",
    ],
    FAMILY_SEARCH: [
        "N", "regime", "kappa_actual", "b", "count", "P_sat", "mean_Uf",
        "std_Uf", "stderr_Uf", "mean_final_energy", "std_final_energy",
        "mean_nodes_visited", "n_limit_hit",
    ],
}

PARAMS_KEY_ORDER = {
    FAMILY_SA: [
        "N_values", "kappa_values", "b_values", "regime", "cooling_strategy",
        "A_prefactor", "cooling_factor", "min_T", "n_MCS_limit", "samples",
        "T0", "seed", "use_kk_init", "timestamp", "elapsed_time_seconds", 
        "n_individual_runs",
    ],
    FAMILY_SEARCH: [
        "N_values", "kappa_values", "b_values", "regime", "algorithm",
        "samples", "max_nodes", "seed", "timestamp", "elapsed_time_seconds",
        "n_individual_runs",
    ],
}

# Fields that must be IDENTICAL across every source experiment in one merge
# (beyond the family itself, which is always checked).
IDENTITY_FIELDS = {
    FAMILY_SA: ("regime", "cooling_strategy", "seed", "use_kk_init"),
    FAMILY_SEARCH: ("regime", "algorithm", "seed"),
}


# ==============================================================================
# NUMERIC HELPERS
# ==============================================================================

def round_sig(x, sig=SIG_FIGS):
    """Round `x` to `sig` significant figures. None/NaN pass through unchanged."""
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return x
    if xf != xf:  # NaN
        return xf
    if xf == 0:
        return 0.0
    from math import log10, floor
    digits = sig - 1 - floor(log10(abs(xf)))
    return round(xf, digits)


def to_float_or_none(s):
    if s is None or s == "":
        return None
    return float(s)


def to_bool(s):
    """Parse a CSV boolean field (Julia prints 'true'/'false')."""
    return str(s).strip().lower() in ("true", "1")


# ==============================================================================
# DISCOVERY
# ==============================================================================

def discover_experiments(directory):
    """
    Find every complete npp_TIMESTAMP_{data,aggregated,params} triple in
    `directory`. Experiments missing one or more of the three files are
    reported and excluded (a merge needs all three to be meaningful).

    Returns:
        dict[str, dict[str, Path]], keyed by timestamp, sorted by timestamp
    """
    directory = Path(directory)
    found = {}
    for f in directory.glob("npp_*"):
        m = TIMESTAMP_RE.search(f.name)
        if m:
            ts, kind, _ext = m.groups()
            found.setdefault(ts, {})[kind] = f

    complete, incomplete = {}, {}
    for ts, members in found.items():
        if {"data", "aggregated", "params"} <= members.keys():
            complete[ts] = members
        else:
            incomplete[ts] = members

    for ts, members in incomplete.items():
        missing = {"data", "aggregated", "params"} - members.keys()
        print(f"WARNING: experiment {ts} is missing {sorted(missing)} -- excluded.",
              file=sys.stderr)

    return dict(sorted(complete.items()))


def normalize_timestamp_arg(raw):
    """Accepts 'npp_20260801_204632' or bare '20260801_204632'."""
    raw = raw.strip()
    return raw[4:] if raw.startswith("npp_") else raw


# ==============================================================================
# FAMILY DETECTION
# ==============================================================================

def detect_family(params, source):
    """
    Determine which experiment family `params` (a loaded params.json) belongs
    to, by which marker key is present:
        "cooling_strategy" present -> "sa"      (SA4NPP_phase_diagram.jl)
        "algorithm"        present -> "search"  (Exhaustive_NPP.jl: CKK or KK)
        neither                    -> pre-cooling-strategy legacy schema

    Arguments:
        params: the loaded params.json dict
        source: the params.json Path, used only for the error message

    Returns:
        FAMILY_SA or FAMILY_SEARCH

    Raises:
        ValueError: on the legacy (pre-family) schema
    """
    if "cooling_strategy" in params:
        return FAMILY_SA
    if "algorithm" in params:
        return FAMILY_SEARCH
    raise ValueError(
        f"{source} uses the pre-cooling-strategy schema (no 'cooling_strategy' "
        f"or 'algorithm' key). Run migrate_legacy_data.py on it before merging."
    )


# ==============================================================================
# LOADING
# ==============================================================================

def load_experiment(members):
    """Load one experiment's params.json + both CSVs into memory."""
    with open(members["params"], "r", encoding="utf-8") as fh:
        params = json.load(fh)
    family = detect_family(params, members["params"])
    if family == FAMILY_SA and "use_kk_init" not in params:
        params["use_kk_init"] = False
    with open(members["data"], "r", newline="", encoding="utf-8") as fh:
        data_rows = list(csv.DictReader(fh))
    with open(members["aggregated"], "r", newline="", encoding="utf-8") as fh:
        agg_rows = list(csv.DictReader(fh))
    return family, params, data_rows, agg_rows


# ==============================================================================
# VALIDATION
# ==============================================================================

def validate_consistent_fields(all_families, all_params):
    """
    Abort with a clear error if:
      (a) the experiments being merged don't all belong to the same family
          ("sa" vs "search" -- their row schemas are incompatible), or
      (b) that family's identity fields (see IDENTITY_FIELDS) differ across
          the experiments -- these describe the physical experiment itself
          and must be identical for a merge to be meaningful.

    Arguments:
        all_families: dict[ts -> family]
        all_params:   dict[ts -> params dict]

    Returns:
        (family, shared_fields: dict) -- the single shared family, and the
        single shared value of each of that family's IDENTITY_FIELDS
    """
    unique_families = set(all_families.values())
    if len(unique_families) > 1:
        detail = ", ".join(f"{ts}={fam!r}" for ts, fam in sorted(all_families.items()))
        raise ValueError(
            f"Cannot merge: experiments belong to different algorithm families "
            f"(sa = SA4NPP_phase_diagram.jl, search = Exhaustive_NPP.jl CKK/KK): "
            f"{detail}"
        )
    family = next(iter(unique_families))

    shared_fields = {}
    for field in IDENTITY_FIELDS[family]:
        values = {ts: p[field] for ts, p in all_params.items()}
        unique = set(values.values())
        if len(unique) > 1:
            detail = ", ".join(f"{ts}={v!r}" for ts, v in values.items())
            raise ValueError(
                f"Cannot merge: '{field}' differs across experiments: {detail}"
            )
        shared_fields[field] = next(iter(unique))

    return family, shared_fields


# ==============================================================================
# AGGREGATED TIE-BREAK
# ==============================================================================

def sort_key_for_winner(agg_row, family):
    """
    Build the ascending sort key used to pick the winning row for a (N, b)
    pair, with SIG_FIGS rounding applied to every numeric field to avoid
    floating-point noise deciding a "tie". The fields compared (beyond
    P_sat, always first) depend on `family` -- see the module docstring's
    "WHAT MERGE MEANS" / aggregated.csv section for the full rationale.

    sa: highest P_sat, then lowest cooling_factor/A_prefactor/mean_Uf/
        mean_final_energy/n_MCS.
        cooling_factor is blank for non-geometric cooling; treated as 0 (a
        neutral, always-equal value within one merge, since cooling_strategy
        is already validated to be identical across every source experiment)
        so it never spuriously breaks a tie when the field simply doesn't
        apply.

    search: highest P_sat, then lowest node-cap-hit fraction, then lowest
        mean_nodes_visited, then lowest mean_Uf/mean_final_energy. For "kk"
        rows the cap-hit fraction is always 0 and mean_nodes_visited is
        always ~N-1, so in practice both reduce to a no-op there and the
        decision falls through to mean_Uf/mean_final_energy, exactly as
        intended (KK is deterministic per seed, so this only matters if
        sources used different seeds and got different single-pass results).
    """
    p_sat = round_sig(agg_row["P_sat"])
    mean_uf = round_sig(agg_row["mean_Uf"])
    mean_final = round_sig(agg_row["mean_final_energy"])

    if family == FAMILY_SA:
        cooling_factor = to_float_or_none(agg_row.get("cooling_factor"))
        cooling_factor = 0.0 if cooling_factor is None else round_sig(cooling_factor)
        A = round_sig(agg_row["A_prefactor"])
        n_mcs = round_sig(agg_row["n_MCS"])
        return (-p_sat, cooling_factor, A, mean_uf, mean_final, n_mcs)

    # FAMILY_SEARCH
    count = to_float_or_none(agg_row.get("count")) or 1.0
    n_limit_hit = to_float_or_none(agg_row.get("n_limit_hit")) or 0.0
    limit_hit_frac = round_sig(n_limit_hit / count)
    mean_nodes = round_sig(agg_row.get("mean_nodes_visited"))
    return (-p_sat, limit_hit_frac, mean_nodes, mean_uf, mean_final)


def pick_winners(experiments, family):
    """
    For every (N, b) pair across all source experiments' aggregated.csv
    rows, pick the winning (timestamp, row) per the tie-break rules.

    Arguments:
        experiments: dict[ts -> (params, data_rows, agg_rows)]
        family: FAMILY_SA or FAMILY_SEARCH, selects the tie-break key

    Returns:
        dict[(N, b) -> (winning_ts, winning_agg_row)]
    """
    candidates = {}  # (N, b) -> list of (ts, row)
    for ts, (_params, _data_rows, agg_rows) in experiments.items():
        for row in agg_rows:
            key = (int(row["N"]), int(row["b"]))
            candidates.setdefault(key, []).append((ts, row))

    winners = {}
    for key, rows in candidates.items():
        best_ts, best_row = min(rows, key=lambda tr: sort_key_for_winner(tr[1], family))
        winners[key] = (best_ts, best_row)
        if len(rows) > 1:
            contenders = ", ".join(f"{ts}(P_sat={r['P_sat']})" for ts, r in rows)
            print(f"  (N={key[0]}, b={key[1]}): {len(rows)} candidates "
                  f"[{contenders}] -> winner {best_ts}")

    return winners


# ==============================================================================
# DATA MERGE + RECOMPUTATION
# ==============================================================================

def build_merged_data(winners, experiments):
    """
    Pull every individual-run row for each winning (N, b) pair from its
    winning source experiment's data.csv, ordered first by N then by b.

    Returns:
        list[dict] -- merged data.csv rows, in final write order
    """
    merged = []
    for (N, b) in sorted(winners.keys()):
        winning_ts, _ = winners[(N, b)]
        _params, data_rows, _agg_rows = experiments[winning_ts]
        for row in data_rows:
            if int(row["N"]) == N and int(row["b"]) == b:
                merged.append(row)
    return merged


def recompute_aggregated_row(N, b, winning_row, data_subset, family):
    """
    Recompute the statistical fields of one aggregated.csv row from the
    actual merged data.csv rows for that (N, b) pair, so the merged
    aggregated.csv is guaranteed self-consistent with the merged data.csv.
    Non-statistical ("parameter") fields are carried over from the winning
    source's own aggregated row verbatim -- see IDENTITY_FIELDS/docstring.

    P_sat/mean_Uf/std_Uf/stderr_Uf/mean_final_energy/std_final_energy/count
    are common to both families and always recomputed the same way, from the
    success/best_energy/final_energy columns present in either data schema.
    The remaining, family-specific search-cost statistic is recomputed too:
        sa:     mean_MCS_used, std_MCS_used  (from n_MCS_used)
        search: mean_nodes_visited           (from nodes_visited)
                n_limit_hit                  (count of limit_hit=true)

    Uses sample standard deviation (ddof=1), matching Julia's std().
    """
    successes = [to_bool(row["success"]) for row in data_subset]
    best_energies = [float(row["best_energy"]) for row in data_subset]
    final_energies = [float(row["final_energy"]) for row in data_subset]
    n = len(data_subset)

    p_sat = sum(successes) / n
    mean_uf = statistics.mean(best_energies)
    std_uf = statistics.stdev(best_energies) if n > 1 else 0.0
    stderr_uf = std_uf / (n ** 0.5)
    mean_final = statistics.mean(final_energies)
    std_final = statistics.stdev(final_energies) if n > 1 else 0.0

    common = {
        "N": N,
        "regime": winning_row["regime"],
        "kappa_actual": winning_row["kappa_actual"],
        "b": b,
        "count": n,
        "P_sat": p_sat,
        "mean_Uf": mean_uf,
        "std_Uf": std_uf,
        "stderr_Uf": stderr_uf,
        "mean_final_energy": mean_final,
        "std_final_energy": std_final,
    }

    if family == FAMILY_SA:
        mcs_used = [float(row["n_MCS_used"]) for row in data_subset]
        mean_mcs_used = statistics.mean(mcs_used)
        std_mcs_used = statistics.stdev(mcs_used) if n > 1 else 0.0
        return {
            **common,
            "cooling_strategy": winning_row["cooling_strategy"],
            "A_prefactor": winning_row["A_prefactor"],
            "cooling_factor": winning_row.get("cooling_factor", ""),
            "n_MCS": winning_row["n_MCS"],
            "mean_MCS_used": mean_mcs_used,
            "std_MCS_used": std_mcs_used,
        }

    # FAMILY_SEARCH
    nodes_visited = [float(row["nodes_visited"]) for row in data_subset]
    mean_nodes = statistics.mean(nodes_visited)
    n_limit_hit = sum(to_bool(row["limit_hit"]) for row in data_subset)
    return {
        **common,
        "mean_nodes_visited": mean_nodes,
        "n_limit_hit": n_limit_hit,
    }


# ==============================================================================
# PARAMS MERGE
# ==============================================================================

def merge_numeric_field(all_params, field, how):
    """how: 'max' or 'min'. None values are ignored; if all are None, result is None."""
    values = [p[field] for p in all_params.values() if p.get(field) is not None]
    if not values:
        return None
    return max(values) if how == "max" else min(values)


def merge_t0(all_params):
    t0_values = [p["T0"] for p in all_params.values()]
    if any(v == HEURISTIC_T0 for v in t0_values):
        return HEURISTIC_T0
    numeric = [float(v) for v in t0_values]
    return min(numeric)


def build_merged_params(family, shared_fields, all_params, merged_data_rows, timestamp, elapsed_time):
    """
    Assemble the merged params.json dict for `family`, combining each
    source's params.json per the rules in the module docstring. N_values,
    b_values, and kappa_values are always derived from the merged data rows
    themselves (never copied from a source), so they can never drift out of
    sync with what was actually merged.
    """
    N_values = sorted({int(r["N"]) for r in merged_data_rows})
    b_by_N = {N: sorted({int(r["b"]) for r in merged_data_rows if int(r["N"]) == N})
              for N in N_values}
    b_values = [b_by_N[N] for N in N_values]
    kappa_values = sorted({round_sig(float(r["kappa_actual"])) for r in merged_data_rows})

    merged = {
        "N_values": N_values,
        "kappa_values": kappa_values,
        "b_values": b_values,
        "timestamp": timestamp,
        "elapsed_time_seconds": elapsed_time,
        "n_individual_runs": len(merged_data_rows),
        **shared_fields,
    }

    if family == FAMILY_SA:
        merged.update({
            "A_prefactor": merge_numeric_field(all_params, "A_prefactor", "max"),
            "cooling_factor": merge_numeric_field(all_params, "cooling_factor", "max"),
            "min_T": merge_numeric_field(all_params, "min_T", "min"),
            "n_MCS_limit": merge_numeric_field(all_params, "n_MCS_limit", "max"),
            "samples": merge_numeric_field(all_params, "samples", "min"),
            "T0": merge_t0(all_params),
        })
    else:  # FAMILY_SEARCH
        merged.update({
            "samples": merge_numeric_field(all_params, "samples", "min"),
            "max_nodes": merge_numeric_field(all_params, "max_nodes", "max"),
        })

    order = PARAMS_KEY_ORDER[family]
    return {k: merged[k] for k in order if k in merged}


# ==============================================================================
# WRITING
# ==============================================================================

def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Merge several small NPP phase-diagram experiments "
                     "(SA, or Exhaustive_NPP.jl CKK/KK -- e.g. split by "
                     "N-block) into one consolidated experiment.",
    )
    parser.add_argument("directory", nargs="?", default=None,
                        help="Folder to merge (default: this script's own directory)")
    parser.add_argument("--timestamps", nargs="+", default=None,
                        help="Merge only these experiments, e.g. "
                             "npp_20260801_204632 20261003_121614")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the merge plan without writing output files.")
    args = parser.parse_args()

    directory = Path(args.directory) if args.directory else Path(__file__).parent
    directory = directory.resolve()

    available = discover_experiments(directory)
    if not available:
        print(f"No complete npp_* experiment triples found in {directory}.", file=sys.stderr)
        sys.exit(1)

    if args.timestamps:
        wanted = {normalize_timestamp_arg(t) for t in args.timestamps}
        missing = wanted - available.keys()
        if missing:
            print(f"ERROR: requested timestamp(s) not found as complete "
                  f"experiments in {directory}: {sorted(missing)}", file=sys.stderr)
            sys.exit(1)
        selected = {ts: available[ts] for ts in wanted}
    else:
        selected = available

    print(f"Merging {len(selected)} experiment(s) from {directory}:")
    for ts in sorted(selected):
        print(f"  - npp_{ts}")

    experiments = {}
    all_families = {}
    all_params = {}
    for ts, members in selected.items():
        try:
            family, params, data_rows, agg_rows = load_experiment(members)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        experiments[ts] = (params, data_rows, agg_rows)
        all_families[ts] = family
        all_params[ts] = params

    try:
        family, shared_fields = validate_consistent_fields(all_families, all_params)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    family_label = "SA (simulated annealing)" if family == FAMILY_SA else "search (CKK/KK, Exhaustive_NPP.jl)"
    print(f"\nFamily: {family_label}")
    print("Shared fields: " + ", ".join(f"{k}={v!r}" for k, v in shared_fields.items()))

    print("\nResolving (N, b) winners across experiments:")
    winners = pick_winners(experiments, family)

    merged_data_rows = build_merged_data(winners, experiments)

    merged_agg_rows = []
    for (N, b) in sorted(winners.keys()):
        winning_ts, winning_row = winners[(N, b)]
        subset = [r for r in merged_data_rows if int(r["N"]) == N and int(r["b"]) == b]
        merged_agg_rows.append(recompute_aggregated_row(N, b, winning_row, subset, family))

    elapsed_time = sum(p.get("elapsed_time_seconds", 0) or 0 for p in all_params.values())
    merge_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_params = build_merged_params(family, shared_fields, all_params, merged_data_rows,
                                         merge_timestamp, elapsed_time)

    print(f"\nMerged result: {len(merged_data_rows)} individual runs across "
          f"{len(merged_agg_rows)} (N, b) pairs, N in {merged_params['N_values']}.")

    if args.dry_run:
        print("\n(dry-run: no files written)")
        return

    data_path = directory / f"npp_{merge_timestamp}_data.csv"
    agg_path = directory / f"npp_{merge_timestamp}_aggregated.csv"
    params_path = directory / f"npp_{merge_timestamp}_params.json"

    write_csv(data_path, DATA_HEADER[family], merged_data_rows)
    write_csv(agg_path, AGG_HEADER[family], merged_agg_rows)
    write_json(params_path, merged_params)

    print(f"\nWrote:\n  {data_path}\n  {agg_path}\n  {params_path}")


if __name__ == "__main__":
    main()
