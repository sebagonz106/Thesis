"""
Exhaustive4NPP_phase_diagram.jl

Exact exhaustive-search phase-diagram study for the Number Partitioning Problem (NPP).
Uses Korf's Complete Karmarkar-Karp (CKK) algorithm: a pruned, depth-first branch-and-bound search
that is guaranteed to find the true optimal discrepancy E1 (not just an approximation).

ALSO supports a PURE Karmarkar-Karp (KK) heuristic run: the classic, single-pass
"largest differencing method" with no branching and no backtracking (Karmarkar &
Karp, 1982). This is NOT exact -- it is a fast O(N log N) greedy heuristic -- and is
offered here purely as a cheap baseline to compare against CKK (exact) and SA
(stochastic) on the same (N, kappa) phase-diagram grid. Select it via
`algorithm=:kk` in `run_phase_diagram_sweep` (default remains `algorithm=:ckk`).
Both algorithms share the exact same instance generation, CSV schema, and
aggregation/plotting pipeline; only the `regime` column differs ("exhaustive" for
CKK, "kk" for the pure heuristic), so every downstream plotter that already groups
by `regime` works unchanged on either kind of run (or a mix of both).

- Control parameter: kappa = b / N.

- Two equivalent ways to parametrize an instance:
    (a) give `kappa` directly            -> b is derived as b = round(kappa * N)
    (b) give `b` directly (integer bits)  -> kappa_actual = b / N is reported
  If both are supplied, kappa wins and b is ignored (a warning is printed once).

- "Perfect" partition: for integer-valued instances the ground-state energy is 0 or 1
  depending on the parity of sum(numbers). Convention: E <= 1 => success
  (SUCCESS_THRESHOLD below).

- A node-count safety valve (`max_nodes`) is provided because, deep in the "hard" phase,
  CKK can in the worst case visit O(2^N) nodes. If the cap is hit before the search
  completes, the run is flagged `n_MCS_limit_hit=true` in the data CSV and the best
  discrepancy found so far is reported (a valid UPPER BOUND on E1, not necessarily the
  exact optimum).

- Multithreading: samples for a given (N, kappa) point are embarrassingly parallel. 
Threads.@threads with a per-task RNG (MersenneTwister seeded deterministically from 
the global seed + indices) keeps results reproducible regardless of thread count. Launch with
      julia -t auto Exhaustive_NPP.jl
  to use all available cores.

WHY THE PRUNING IS CORRECT:
  CKK explores a binary tree where, at each node, the two largest numbers currently in
  the (shrinking) multiset are combined either by DIFFERENCE (a1-a2, "left": they go to
  opposite subsets) or by SUM (a1+a2, "right": they go to the same subset). Every one of
  the N! / ... leaves of the full tree (after N-1 combination steps) corresponds to one
  feasible partition, and the single number left at a leaf IS that partition's
  discrepancy. So an unpruned depth-first traversal already visits every partition
  exactly once up to the symmetry s -> -s, and taking the min over all leaves is
  correct by exhaustion.

  The pruning rule removes subtrees without removing any leaf that could beat the
  current best: at any node, if the largest remaining number `l` is higher than or equal 
  to the sum `r` of all the other remaining numbers, then no matter how the rest of the 
  numbers are subsequently split between the two subsets, `l` alone exceeds anything the 
  other elements could cancel it with, so every leaf below this node has discrepancy exactly
  `l - r` (put everything else against `l`) or larger (any split that doesn't fully
  oppose `l` leaves an even bigger imbalance). Formally: for any sign assignment of the
  remaining elements, |l*s_l + sum(a_i*s_i, i != l)| >= l - sum(a_i, i!=l) = l - r, with
  equality iff all the other elements are given the sign opposite to s_l. So the minimum
  achievable discrepancy in that subtree is exactly l - r, which is computed directly
  without recursing further. This is a lower bound that is simultaneously achieved, so
  no optimal solution is ever lost -- the pruning is exact, not heuristic.
"""

using Random
using Statistics
using Printf
using JSON
using Dates
using Base.Threads

EXPERIMENTS_FOLDER = joinpath(@__DIR__, "Experiments")
SUCCESS_THRESHOLD = 1.1              # E <= 1.1 counts as a "perfect" partition
BIGINT_THRESHOLD_BITS = 100          # switch Int128 -> BigInt above this many bits
REGIME_LABEL = "exhaustive"          # regime tag for CKK (exact) runs
KK_REGIME_LABEL = "kk"               # regime tag for pure KK (heuristic) runs
VALID_ALGORITHMS = (:ckk, :kk)       # :ckk = Korf's exact branch-and-bound
                                      # :kk  = classic single-pass differencing heuristic

const _bigint_warned = Ref(false)

# ==============================================================================
# 1. PROBLEM INSTANCE GENERATION
# ==============================================================================

"""
    resolve_b_kappa(N::Int; kappa=nothing, b=nothing)

Resolve the (b, kappa) pair for an instance of size N, given either one.
If both are supplied, `kappa` takes precedence and `b` is ignored (warning printed).

Returns:
    (b_used::Int, kappa_requested::Union{Float64,Nothing}, kappa_actual::Float64)
"""
function resolve_b_kappa(N::Int; kappa::Union{Real,Nothing}=nothing, b::Union{Int,Nothing}=nothing)
    if kappa === nothing && b === nothing
        error("resolve_b_kappa: must supply at least one of `kappa` or `b`.")
    end
    if kappa !== nothing && b !== nothing
        @warn "Both kappa and b were supplied; kappa takes precedence, b=$b is ignored." maxlog=1
    end
    if kappa !== nothing
        b_used = max(1, round(Int, kappa * N))
        kappa_actual = b_used / N
        return b_used, Float64(kappa), kappa_actual
    else
        kappa_actual = b / N
        return b, nothing, kappa_actual
    end
end

"""
    generate_instance(N::Int, b::Int, rng::AbstractRNG)

Generate a random NPP instance of N non-negative integers, each with exactly `b` bits
of resolution, i.e. uniformly drawn from {0, ..., 2^b - 1}.

Uses Int128 for b <= BIGINT_THRESHOLD_BITS, and BigInt above that (prints a one-time
warning, since BigInt arithmetic is noticeably slower).
"""
function generate_instance(N::Int, b::Int, rng::AbstractRNG)
    if b <= BIGINT_THRESHOLD_BITS
        M = Int128(2)^b
        return rand(rng, Int128(0):(M - 1), N)
    else
        if !_bigint_warned[]
            @warn "b=$b exceeds $(BIGINT_THRESHOLD_BITS) bits; switching to BigInt (slower)." maxlog=1
            _bigint_warned[] = true
        end
        M = BigInt(2)^b
        return [rand(rng, big(0):(M - 1)) for _ in 1:N]
    end
end

# ==============================================================================
# 2. COMPLETE KARMARKAR-KARP (CKK) EXACT SOLVER
# ==============================================================================

"""
    CKKResult

Outcome of one exact CKK search.

Fields:
    best_energy::T   - the true optimal discrepancy E1 (or, if `max_nodes` was hit
                        before completion, the best UPPER BOUND found so far)
    nodes_visited::Int
    limit_hit::Bool  - true if the node-count safety cap was reached before the search
                       completed (so best_energy may not be the exact optimum)
"""
struct CKKResult{T<:Integer}
    best_energy::T
    nodes_visited::Int
    limit_hit::Bool
end

"""
    ckk_search(numbers::Vector{T}; max_nodes::Int=typemax(Int)) where {T<:Integer}

Korf's Complete Karmarkar-Karp algorithm: exact minimum discrepancy via depth-first
branch-and-bound.

At each node we hold a multiset of "remaining" numbers (conceptually: the original list
after some pairs have already been irrevocably committed to same/different subsets).
We repeatedly take the two largest, a1 >= a2, and branch:
    - LEFT:  replace {a1, a2} by |a1 - a2|   (a1, a2 end up in DIFFERENT subsets)
    - RIGHT: replace {a1, a2} by  a1 + a2    (a1, a2 end up in the SAME subset)
Recursing N-1 times collapses the list to one number: that leaf's partition discrepancy.

Pruning: if the largest remaining number `l` is >= the sum `r` of all other remaining numbers, 
the best possible discrepancy in this subtree is exactly l - r; we record it and do not recurse further.

Early stop: once a perfect partition (E <= SUCCESS_THRESHOLD) has been found anywhere in
the tree, the whole search halts immediately.

A sorted Vector{T} is kept at every node (descending order) via binary-search insertion,
which is O(log k) per insertion for a node with k remaining elements -- asymptotically
irrelevant next to the O(2^N) worst-case branching factor, but keeps the two largest
elements accessible in O(1) at every node.

Arguments:
    numbers::Vector{T}  - the N input numbers (T <: Integer, e.g. Int128 or BigInt)
    max_nodes::Int       - safety cap on the number of search-tree nodes visited

Returns:
    CKKResult{T}
"""
function ckk_search(numbers::Vector{T}; max_nodes::Int=typemax(Int)) where {T<:Integer}
    nums = sort(numbers, rev=true)
    total = sum(nums)
    success_cutoff = T(1)  # E <= 1 is "perfect"

    best = Ref(total)       # trivial upper bound: all in one set vs. empty set -> E = total
    nodes = Ref(0)
    limit_hit = Ref(false)
    done = Ref(false)       # set true once a perfect partition is found (global early stop)

    function recurse(lst::Vector{T})
        if done[] || limit_hit[]
            return
        end
        nodes[] += 1
        if nodes[] > max_nodes
            limit_hit[] = true
            return
        end

        n = length(lst)
        if n <= 1
            e = n == 0 ? zero(T) : lst[1]
            if e < best[]
                best[] = e
            end
            if best[] <= success_cutoff
                done[] = true
            end
            return
        end

        s = sum(lst)
        largest = lst[1]
        rest = s - largest

        # Exact pruning rule: largest dominates the rest -> stop branching, the
        # subtree's minimum discrepancy is exactly largest - rest.
        if largest >= rest
            e = largest - rest
            if e < best[]
                best[] = e
            end
            if best[] <= success_cutoff
                done[] = true
            end
            return
        end

        a1, a2 = lst[1], lst[2]
        remainder = lst[3:end]

        # LEFT branch: different subsets -> |a1 - a2|
        diff = a1 - a2
        left_list = _insert_sorted_desc(remainder, diff)
        recurse(left_list)

        if done[] || limit_hit[]
            return
        end

        # RIGHT branch: same subset -> a1 + a2
        ssum = a1 + a2
        right_list = _insert_sorted_desc(remainder, ssum)
        recurse(right_list)
    end

    recurse(nums)

    return CKKResult{T}(best[], nodes[], limit_hit[])
end

"""
    _insert_sorted_desc(sorted_desc::Vector{T}, value::T) where {T<:Integer}

Insert `value` into a descending-sorted vector via binary search, returning a new
vector (the parent node's list is never mutated, since it is also used by the sibling
branch).
"""
function _insert_sorted_desc(sorted_desc::Vector{T}, value::T) where {T<:Integer}
    lo, hi = 1, length(sorted_desc) + 1
    while lo < hi
        mid = (lo + hi) ÷ 2
        if sorted_desc[mid] >= value
            lo = mid + 1
        else
            hi = mid
        end
    end
    out = Vector{T}(undef, length(sorted_desc) + 1)
    @inbounds begin
        out[1:lo-1] .= @view sorted_desc[1:lo-1]
        out[lo] = value
        out[lo+1:end] .= @view sorted_desc[lo:end]
    end
    return out
end

# ==============================================================================
# 2b. PURE KARMARKAR-KARP (KK) HEURISTIC -- FAST BASELINE
# ==============================================================================

"""
    KKResult

Outcome of one pure KK heuristic run on a single instance.

Fields:
    final_energy::T  - the discrepancy produced by the single greedy differencing
                        pass (an UPPER BOUND on the true optimum E1, generally not
                        tight -- KK has no optimality guarantee)
    nodes_visited::Int - always N-1 (one combination step per pass); kept only so
                        the CSV schema is identical to CKKResult/ExhaustiveResult
"""
struct KKResult{T<:Integer}
    final_energy::T
    nodes_visited::Int
end

"""
    kk_heuristic(numbers::Vector{T}) where {T<:Integer}

Classic Karmarkar-Karp "largest differencing method" (Karmarkar & Karp, 1982):
repeatedly take the two largest remaining numbers a1 >= a2 and replace them by
their difference a1 - a2 (i.e. commit them to OPPOSITE subsets), until a single
number remains. That number is the heuristic's discrepancy.

This is exactly the LEFT branch of `ckk_search`'s tree, taken greedily and
EXCLUSIVELY at every node: there is no RIGHT (sum) branch, no backtracking, and
no pruning test, because there is nothing left to prune -- a single deterministic
root-to-leaf path is walked instead of the whole tree. It is therefore a pure
O(N log N) heuristic baseline (sorted-list insertion per step), not an
optimization procedure: the result is an upper bound on the true optimal
discrepancy E1, usually not tight, and (unlike CKK) never improved upon.

Arguments:
    numbers::Vector{T}  - the N input numbers (T <: Integer, e.g. Int128 or BigInt)

Returns:
    KKResult{T}
"""
function kk_heuristic(numbers::Vector{T}) where {T<:Integer}
    nums = sort(numbers, rev=true)
    steps = 0
    while length(nums) > 1
        a1, a2 = nums[1], nums[2]
        diff = a1 - a2
        nums = _insert_sorted_desc(nums[3:end], diff)
        steps += 1
    end
    final = isempty(nums) ? zero(T) : nums[1]
    return KKResult{T}(final, steps)
end

# ==============================================================================
# 3. EXPERIMENT EXECUTION
# ==============================================================================

"""
    Results of a single exact CKK (or heuristic KK) run on one instance. The
    `regime` field distinguishes which algorithm produced the row
    ("exhaustive" = CKK, "kk" = pure KK heuristic); every other field keeps
    the same meaning and type across both, so CSV consumers never need to
    branch on `regime` to know how to read a row.
"""
struct ExhaustiveResult
    N::Int
    b::Int
    kappa_requested::Union{Float64,Nothing}
    kappa_actual::Float64
    regime::String                 # always REGIME_LABEL = "exhaustive"
    nodes_visited::Int
    max_nodes::Int
    limit_hit::Bool
    best_energy::Float64           # == final_energy for CKK (exact algorithm);
    final_energy::Float64          # kept distinct only to mirror SAResult's CSV schema
    success::Bool
    seed::Int
end

"""
    run_ckk_on_instance(numbers::Vector{T}; max_nodes::Int) where {T<:Integer}

Thin wrapper around ckk_search returning the fields needed to build an
ExhaustiveResult, with Float64 casts applied, so downstream aggregation 
code is interchangeable with the SA script's.
"""
function run_ckk_on_instance(numbers::Vector{T}; max_nodes::Int) where {T<:Integer}
    res = ckk_search(numbers; max_nodes=max_nodes)
    success = res.best_energy <= SUCCESS_THRESHOLD
    return (best_energy=Float64(res.best_energy), final_energy=Float64(res.best_energy),
            nodes_visited=res.nodes_visited, limit_hit=res.limit_hit, success=success)
end

"""
    run_kk_on_instance(numbers::Vector{T}) where {T<:Integer}

Thin wrapper around `kk_heuristic`, mirroring `run_ckk_on_instance`'s return shape
(same field names/types) so both algorithms can be dispatched through one code
path in `run_phase_diagram_sweep`. `limit_hit` is always `false` (KK is a single
deterministic pass, never capped), and `best_energy == final_energy` is kept only
for CSV-schema parity with the CKK/SA scripts -- for KK both are simply the one
discrepancy the heuristic found.
"""
function run_kk_on_instance(numbers::Vector{T}) where {T<:Integer}
    res = kk_heuristic(numbers)
    success = res.final_energy <= SUCCESS_THRESHOLD
    return (best_energy=Float64(res.final_energy), final_energy=Float64(res.final_energy),
            nodes_visited=res.nodes_visited, limit_hit=false, success=success)
end

"""
    aggregate_results(all_results::Vector{ExhaustiveResult})

Group results by (N, regime, kappa_actual rounded to 6 digits) and compute:
    P_sat   = fraction of instances with a perfect partition (E<=1)
    U_f     = mean(best_energy), std(best_energy), stderr(best_energy)
  -- same statistics as SA4NPP_phase_diagram.jl's aggregate_results, but since CKK is
     exact, P_sat here is the TRUE probability of a perfect partition (up to sampling
     noise over instances), not an SA-success-rate proxy for it.

Returns:
    Vector{Dict} - one entry per (N, regime, kappa) group
"""
function aggregate_results(all_results::Vector{ExhaustiveResult})
    groups = Dict{Tuple{Int,String,Float64}, Vector{ExhaustiveResult}}()
    for r in all_results
        key = (r.N, r.regime, round(r.kappa_actual, digits=6))
        push!(get!(groups, key, ExhaustiveResult[]), r)
    end

    aggregated = Vector{Dict}()
    for key in sort(collect(keys(groups)))
        N, regime, kappa = key
        runs = groups[key]
        n_runs = length(runs)

        successes = count(r -> r.success, runs)
        p_sat = successes / n_runs

        energies = [r.best_energy for r in runs]
        mean_uf = mean(energies)
        std_uf = std(energies)
        stderr_uf = std_uf / sqrt(n_runs)

        final_energies = [r.final_energy for r in runs]
        mean_final = mean(final_energies)
        std_final = std(final_energies)

        mean_nodes = mean(r.nodes_visited for r in runs)
        n_limit_hit = count(r -> r.limit_hit, runs)

        push!(aggregated, Dict(
            "N" => N,
            "regime" => regime,
            "kappa_actual" => kappa,
            "b" => runs[1].b,
            "count" => n_runs,
            "P_sat" => p_sat,
            "mean_Uf" => mean_uf,
            "std_Uf" => std_uf,
            "stderr_Uf" => stderr_uf,
            "mean_final_energy" => mean_final,
            "std_final_energy" => std_final,
            "mean_nodes_visited" => mean_nodes,
            "n_limit_hit" => n_limit_hit,
        ))
    end
    return aggregated
end

# ==============================================================================
# 4. FILE OUTPUT
# ==============================================================================

"""
    save_results(results, aggregated, parameters, start)

Save:
- npp_TIMESTAMP_data.csv       : one row per individual exact CKK run
- npp_TIMESTAMP_aggregated.csv : one row per (N, regime, kappa) group (P_sat, U_f, ...)
- npp_TIMESTAMP_params.json    : experiment configuration + metadata
Same filename pattern and folder (`Experiments/`) as SA4NPP_phase_diagram.jl.
"""
function save_results(results::Vector{ExhaustiveResult}, aggregated::Vector{Dict}, parameters::Dict, start::DateTime)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    elapsed = now() - start

    mkpath(EXPERIMENTS_FOLDER)

    data_file = joinpath(EXPERIMENTS_FOLDER, "npp_$(timestamp)_data.csv")
    agg_file = joinpath(EXPERIMENTS_FOLDER, "npp_$(timestamp)_aggregated.csv")
    params_file = joinpath(EXPERIMENTS_FOLDER, "npp_$(timestamp)_params.json")

    open(data_file, "w") do io
        println(io, "N,b,kappa_requested,kappa_actual,regime,nodes_visited,max_nodes,limit_hit,best_energy,final_energy,success,seed")
        for r in results
            kreq = r.kappa_requested === nothing ? "" : @sprintf("%.6f", r.kappa_requested)
            println(io, "$(r.N),$(r.b),$kreq,$(r.kappa_actual),$(r.regime),$(r.nodes_visited),$(r.max_nodes),$(r.limit_hit),$(r.best_energy),$(r.final_energy),$(r.success),$(r.seed)")
        end
    end

    open(agg_file, "w") do io
        println(io, "N,regime,kappa_actual,b,count,P_sat,mean_Uf,std_Uf,stderr_Uf,mean_final_energy,std_final_energy,mean_nodes_visited,n_limit_hit")
        for a in aggregated
            println(io, "$(a["N"]),$(a["regime"]),$(a["kappa_actual"]),$(a["b"]),$(a["count"]),$(a["P_sat"]),$(a["mean_Uf"]),$(a["std_Uf"]),$(a["stderr_Uf"]),$(a["mean_final_energy"]),$(a["std_final_energy"]),$(a["mean_nodes_visited"]),$(a["n_limit_hit"])")
        end
    end

    params_dict = merge(parameters, Dict(
        "timestamp" => timestamp,
        "elapsed_time_seconds" => Dates.value(elapsed) / 1000.0,
        "n_individual_runs" => length(results),
    ))
    open(params_file, "w") do io
        write(io, JSON.json(params_dict, 2))
    end

    println("\n$(length(results)) individual runs saved to CSV")
    return data_file, agg_file, params_file
end

# ==============================================================================
# 5. MAIN SWEEP
# ==============================================================================

"""
    resolve_b_values_for_N(b_values, idx_N::Int)

Same semantics as in SA4NPP_phase_diagram.jl (see that file's docstring):
    (a) Vector{Int}          -> the same list applies to every N
    (b) Vector{Vector{Int}}  -> per-N lists, selected by position, last one reused
                                if fewer inner lists than N_values are given
"""
function resolve_b_values_for_N(b_values::Union{Vector{Int},Vector{Vector{Int}}}, idx_N::Int)
    if b_values isa Vector{Vector{Int}}
        return idx_N <= length(b_values) ? b_values[idx_N] : b_values[end]
    else
        return b_values
    end
end

"""
    run_phase_diagram_sweep(
        N_values, kappa_values;
        b_values=nothing, samples=100, seed=16, max_nodes=10_000_000, algorithm=:ckk,
    )

Run the full (N x kappa) sweep, threaded over samples, with either algorithm:
    :ckk  (default) - Korf's exact CKK branch-and-bound (see file docstring);
                       rows tagged regime="exhaustive"
    :kk             - the pure, single-pass KK differencing heuristic (no
                       branching, no backtracking, no optimality guarantee);
                       rows tagged regime="kk"
Mirrors SA4NPP_phase_diagram.jl's run_phase_diagram_sweep, minus the
cooling_strategy axis (neither CKK nor KK has one -- see file-level docstring).

Arguments:
    N_values::Vector{Int}                  - problem sizes to scan
    kappa_values::Vector{Float64}          - kappa values to scan (`nothing` to use
                                              b_values instead); single flat list shared
                                              across every N
    b_values::Union{Vector{Int},Vector{Vector{Int}},Nothing}
                                            - alternative to kappa_values (see
                                              resolve_b_values_for_N)
    samples::Int                           - independent instances per (N, kappa) point
    seed::Int                              - master seed (per-run RNGs derived from this)
    max_nodes::Int                         - safety cap on CKK search-tree nodes per run
                                              (deep in the hard phase, worst case is
                                              O(2^N); this caps wall-clock time at the
                                              cost of possibly not proving optimality --
                                              flagged via `limit_hit`/`n_limit_hit`).
                                              Ignored when algorithm=:kk (a single KK
                                              pass always takes exactly N-1 steps, so
                                              there is nothing to cap).
    algorithm::Symbol                      - :ckk (exact, default) or :kk (pure
                                              heuristic baseline); see above

Returns:
    (all_results::Vector{ExhaustiveResult}, aggregated::Vector{Dict}, data_file, agg_file, params_file)
"""
function run_phase_diagram_sweep(
    N_values::Vector{Int},
    kappa_values::Union{Vector{Float64},Nothing};
    b_values::Union{Vector{Int},Vector{Vector{Int}},Nothing}=nothing,
    samples::Int=100,
    seed::Int=16,
    max_nodes::Int=10_000_000,
    algorithm::Symbol=:ckk,
)
    if kappa_values === nothing && b_values === nothing
        error("Must supply either kappa_values or b_values.")
    end
    if !(algorithm in VALID_ALGORITHMS)
        error("algorithm must be one of $VALID_ALGORITHMS, got :$algorithm")
    end
    using_kappa = kappa_values !== nothing
    is_ckk = algorithm === :ckk
    regime_label = is_ckk ? REGIME_LABEL : KK_REGIME_LABEL
    algo_name = is_ckk ? "exact CKK" : "pure KK heuristic"

    scan_values_by_N = if using_kappa
        [kappa_values for _ in N_values]
    else
        [Float64.(resolve_b_values_for_N(b_values, idx)) for (idx, _) in enumerate(N_values)]
    end
    total_points = sum(length(sv) for sv in scan_values_by_N) * samples

    parameters = Dict(
        "N_values" => N_values,
        "kappa_values" => using_kappa ? kappa_values : nothing,
        "b_values" => using_kappa ? nothing : b_values,
        "regime" => regime_label,
        "algorithm" => string(algorithm),
        "samples" => samples,
        "max_nodes" => is_ckk ? max_nodes : nothing,
        "seed" => seed,
    )

    println("\n" * "="^80)
    println("NPP PHASE-DIAGRAM SWEEP -- regime=$regime_label ($algo_name), samples=$samples" *
            (is_ckk ? ", max_nodes=$max_nodes" : ""))
    println("N values:     $N_values")
    println(using_kappa ? "kappa values: $kappa_values" : "b values:     $b_values")
    println("threads:      $(Threads.nthreads())")
    println("="^80 * "\n")

    all_results = Vector{ExhaustiveResult}(undef, total_points)
    start_time = now()

    idx_counter = 0
    for (idx_N, N) in enumerate(N_values)
        scan_values = scan_values_by_N[idx_N]

        for sv in scan_values
            b_used, kappa_requested, kappa_actual = using_kappa ?
                resolve_b_kappa(N; kappa=sv) : resolve_b_kappa(N; b=round(Int, sv))

            base_idx = idx_counter
            @threads for s in 1:samples
                run_seed = seed + hash((N, b_used, s)) % 1_000_000_000
                rng = MersenneTwister(run_seed)

                numbers = generate_instance(N, b_used, rng)
                res = is_ckk ? run_ckk_on_instance(numbers; max_nodes=max_nodes) :
                               run_kk_on_instance(numbers)

                all_results[base_idx + s] = ExhaustiveResult(
                    N, b_used, kappa_requested, kappa_actual, regime_label,
                    res.nodes_visited, (is_ckk ? max_nodes : res.nodes_visited), res.limit_hit,
                    res.best_energy, res.final_energy, res.success, run_seed,
                )
            end
            idx_counter += samples

            n_success = count(r -> r.success, all_results[base_idx+1:base_idx+samples])
            n_hit = count(r -> r.limit_hit, all_results[base_idx+1:base_idx+samples])
            @printf("  N=%-5d kappa=%.4f (b=%-2d)  P_perfect=%.3f  <best_E>=%.3e  limit_hit=%d/%d  [%.1fs elapsed]\n",
                    N, kappa_actual, b_used, n_success / samples,
                    mean(r.best_energy for r in all_results[base_idx+1:base_idx+samples]),
                    n_hit, samples,
                    Dates.value(now() - start_time) / 1000.0)
        end
    end

    println("\n" * "="^80)
    println("AGGREGATING AND SAVING RESULTS")
    println("="^80)

    aggregated = aggregate_results(all_results)
    data_file, agg_file, params_file = save_results(all_results, aggregated, parameters, start_time)

    println("Data file:       $data_file")
    println("Aggregated file: $agg_file")
    println("Params file:     $params_file")
    println("="^80 * "\n")

    return all_results, aggregated, data_file, agg_file, params_file
end

# ==============================================================================
# 6. EXAMPLE USAGE
# ==============================================================================

if abspath(PROGRAM_FILE) == @__FILE__

    samples = 1000
    max_nodes = 100_000_000_000   # safety cap; raise for deeper "hard-phase" exploration

    N_values = collect(10:1:35)
    b_values = Vector{Vector{Int64}}()

    for N in N_values
        b_list = collect(floor(Int, N*0.2):1:ceil(Int, N*0.6))
        push!(b_values, b_list)
    end

    # Exact CKK run (default algorithm=:ckk) -- regime="exhaustive" in the output CSVs.
    # results, aggregated, df, af, pf = run_phase_diagram_sweep(
    #     N_values, nothing;
    #     samples=samples, seed=16, b_values=b_values, max_nodes=max_nodes,
    # )

    # Pure KK heuristic run -- same (N, kappa) grid, same CSV schema, regime="kk".
    # Fast O(N log N) baseline for comparing against CKK (exact) and SA (stochastic)
    # on identical instances/seeds.
    #
    results_kk, aggregated_kk, df_kk, af_kk, pf_kk = run_phase_diagram_sweep(
        N_values, nothing;
        samples=samples, seed=16, b_values=b_values, algorithm=:kk,
    )
end
