"""
SA4NPP_phase_diagram.jl

Simulated Annealing phase-diagram study for the Number Partitioning Problem (NPP),
mirroring the methodology used for K-SAT: run SA with a fixed computational budget 
that scales as A*N^z Monte Carlo Sweeps (MCS), z = 1 (linear), 2 (quadratic), 3 (cubic), 
and measure P_sat(kappa) and U_f(kappa) at that fixed budget, for several N, to look for 
a (possibly N-dependent, possibly fixed) threshold.

- Control parameter: kappa = b / N (bits per element), the NPP analogue of alpha = n/m
  in K-SAT.

- Two equivalent ways to parametrize an instance:
    (a) give `kappa` directly            -> b is derived as b = round(kappa * N)
    (b) give `b` directly (integer bits)  -> kappa_actual = b / N is reported
  If both are supplied, kappa wins and b is ignored (a warning is printed once).

- Each instance is a vector of N independent integers uniformly drawn from
  {0, 1, ..., 2^b - 1}. Int128 is used while b <= 100 (safe margin under Int128's
  127-bit range); for b > 100 the code automatically switches to BigInt and prints
  a one-time warning, since BigInt arithmetic is slower.

- "Perfect" partition: for integer-valued instances the ground-state energy is 0 or 1
  depending on the parity of sum(numbers). Convention used here: E <= 1 => success
  (SUCCESS_THRESHOLD below).

- Two independent axes control the SA schedule:
    (1) `regime` ("linear"/"quadratic"/"cubic") sets the exponent z in A*N^z, as before.
    (2) `cooling_strategy` ("fixed"/"geometric"/"logarithmic") sets HOW temperature
        decreases over the course of a run. Each strategy is implemented by its own
        independent SA function:

      - "fixed" (simulated_annealing_fixed_budget, UNCHANGED from before):
            n_MCS   = round(A * N^z)              # total budget, set by `regime`
            T(step) = T0 * (1 - step / n_MCS)      # constant ΔT = T0/n_MCS per MCS
            1 MCS   = N single-spin-flip attempts
        Temperature is distributed uniformly over the whole budget: `regime` and `A`
        directly determine the TOTAL number of MCS performed.

      - "geometric" (simulated_annealing_geometric) and "logarithmic"
        (simulated_annealing_logarithmic): here `regime` and `A` instead determine
        MCS_per_T = round(A * N^z), the number of Monte Carlo Sweeps performed AT
        EACH temperature stage, not the total budget. The schedule then advances
        through as many temperature stages as needed until either a perfect
        partition is found, the temperature drops below `min_T`, or the total MCS
        performed reaches the safety cap `n_MCS_limit` -- whichever comes first.
        `min_T` and `n_MCS_limit` are plain safety settings, the same for every
        (N, kappa) point in a sweep regardless of `regime`/`A`/N (per design: "the
        cap is independent of the regime"), so they are recorded once in the
        experiment's params.json rather than duplicated in every CSV row.
          * geometric:    T(stage) = T0 * cooling_factor^stage
          * logarithmic:  T(stage) = T0 / (1 + cooling_factor * ln(1 + stage))   (stage = 0, 1, 2, ...)
            A beta-generalized Hajek-style logarithmic schedule (Hajek, 1988): the
            `cooling_factor` argument plays the role of beta (β), the time-scale
            parameter controlling the cooling rate. beta << 1 keeps the temperature
            close to T0 for many more stages (more exploration at high T, the
            intended regime here); beta = 1 cools at roughly the rate of the
            classical Hajek schedule T0*ln(2)/ln(stage+2); beta > 1 cools faster.
            T(0) = T0 exactly, matching how the other two strategies start. Hajek's
            asymptotic convergence guarantee requires the effective constant T0/beta
            to be at least the depth of the deepest non-global local minimum, so
            smaller beta is the safe direction; this is why low beta is recommended
            rather than beta > 1.
    `cooling_factor` IS recorded per-row in the CSV for BOTH "geometric" and
    "logarithmic" strategies (the same field is reused for beta rather than adding
    a new column), since in both cases it is physically meaningful to compare
    across points.

- U_f is defined as the mean of `best_energy` reached during the run (the best
  configuration ever visited within the run), not the energy of the literal final
  configuration. Both are still recorded in the raw per-run CSV.

- b_values accepts two shapes for flexibility in which (N, b) combinations to scan:
    (a) Vector{Int}          -> the same list of b values is used for every N
    (b) Vector{Vector{Int}}  -> one list of b values per N (by position); if fewer
                                inner lists than N values are given, the LAST inner
                                list provided is reused for all remaining N values.
  kappa_values, by contrast, always stays a single flat list shared across every N.

- Multithreading: samples for a given (N, kappa, regime, cooling_strategy) point are
  embarrassingly parallel. The code uses Threads.@threads with a per-task RNG
  (MersenneTwister seeded deterministically from the global seed + indices) so
  results stay reproducible regardless of thread scheduling / thread count. Launch
  Julia with
      julia -t auto SA4NPP_phase_diagram.jl
  to use all available cores.
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

Arguments:
    N::Int           - number of elements
    b::Int           - bits of resolution per element (b = kappa * N)
    rng::AbstractRNG - RNG to draw from (pass a per-task RNG for thread safety)

Returns:
    Vector{<:Integer} - N random integers in [0, 2^b - 1]
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

# ------------------------------------------------------------------------
# Minimal binary max-heap (1-indexed array, sift-up/sift-down), used by
# kk_partition_signs below. Written by hand rather than pulling in
# DataStructures.jl, to keep this file dependency-free like the rest of the
# project (only Random/Statistics/Printf/JSON/Dates/Base.Threads are used
# anywhere here)
# ------------------------------------------------------------------------

@inline _heap_left(i::Int) = 2i
@inline _heap_right(i::Int) = 2i + 1
@inline _heap_parent(i::Int) = i >> 1

function _sift_down!(heap::Vector{Tuple{T,Int}}, i::Int) where {T}
    n = length(heap)
    while true
        l, r, largest = _heap_left(i), _heap_right(i), i
        if l <= n && heap[l][1] > heap[largest][1]
            largest = l
        end
        if r <= n && heap[r][1] > heap[largest][1]
            largest = r
        end
        largest == i && break
        heap[i], heap[largest] = heap[largest], heap[i]
        i = largest
    end
end

function _sift_up!(heap::Vector{Tuple{T,Int}}, i::Int) where {T}
    while i > 1
        p = _heap_parent(i)
        heap[p][1] >= heap[i][1] && break
        heap[p], heap[i] = heap[i], heap[p]
        i = p
    end
end

function _heapify_max!(heap::Vector{Tuple{T,Int}}) where {T}
    for i in (length(heap) ÷ 2):-1:1
        _sift_down!(heap, i)
    end
    return heap
end

function _heap_push_max!(heap::Vector{Tuple{T,Int}}, item::Tuple{T,Int}) where {T}
    push!(heap, item)
    _sift_up!(heap, length(heap))
    return heap
end

function _heap_pop_max!(heap::Vector{Tuple{T,Int}}) where {T}
    top = heap[1]
    last_item = pop!(heap)
    if !isempty(heap)
        heap[1] = last_item
        _sift_down!(heap, 1)
    end
    return top
end

"""
    kk_partition_signs(numbers::Vector{T}) where {T<:Integer}

Given an NPP instance `numbers`, compute the +1/-1 SUBSET ASSIGNMENT the
classic Karmarkar-Karp ("largest differencing") heuristic produces for it,
via a proper O(N log N) max-heap (same algorithmic shape as Python's
`heapq`), by tracking, for every merge of the two currently-largest
"groups", which original indices ride along inside each group and flipping
their signs exactly as the smaller-valued group's contribution requires --
this mirrors the sign-bookkeeping in the Python reference implementation.

Intended use: a fast, informed alternative to a uniformly random initial
`state` for simulated annealing -- instead of `rand(rng, Int8[-1, 1], N)`,
call `kk_partition_signs(numbers)` to start SA already at (or very near) the
KK heuristic's own solution. See `simulated_annealing_fixed_budget`,
`simulated_annealing_geometric`, and `simulated_annealing_logarithmic`'s
`use_kk_init` keyword below, which wire this in directly.

At each step the two largest groups x (value x_val) and y (value y_val,
x_val >= y_val by heap order) are merged: x's members keep their current
sign (first time touched: assigned +1), y's members have their current sign
flipped (first time touched: assigned -1), and the merged group's value
becomes x_val - y_val = |x_val - y_val|, pushed back onto the heap under a
fresh group id. After N-1 merges a single group remains and every index has
been assigned a sign.

Arguments:
    numbers::Vector{T} - the N input numbers (T <: Integer)

Returns:
    Vector{Int8} -- signs[i] in {-1, +1}; sum(signs .* numbers) equals, in
    absolute value, the KK heuristic's discrepancy for `numbers`.
"""
function kk_partition_signs(numbers::Vector{T}) where {T<:Integer}
    N = length(numbers)
    signs = zeros(Int8, N)
    N == 0 && return signs
    if N == 1
        signs[1] = 1
        return signs
    end

    # groups[gid] = original indices folded into merged cluster `gid`.
    # Singletons occupy ids 1:N; each of the N-1 merges allocates a fresh id
    # above N rather than mutating a parent in place, so group lookups by id
    # stay O(1) throughout.
    groups = Vector{Vector{Int}}(undef, 2N - 1)
    for i in 1:N
        groups[i] = [i]
    end
    next_group_id = N

    heap = Vector{Tuple{T,Int}}(undef, N)
    for i in 1:N
        heap[i] = (numbers[i], i)
    end
    _heapify_max!(heap)

    while length(heap) > 1
        x_val, x_gid = _heap_pop_max!(heap)
        y_val, y_gid = _heap_pop_max!(heap)  # x_val >= y_val (max-heap pop order)

        for idx in groups[x_gid]
            signs[idx] = signs[idx] == 0 ? Int8(1) : signs[idx]
        end
        for idx in groups[y_gid]
            signs[idx] = signs[idx] == 0 ? Int8(-1) : -signs[idx]
        end

        next_group_id += 1
        groups[next_group_id] = vcat(groups[x_gid], groups[y_gid])
        _heap_push_max!(heap, (x_val - y_val, next_group_id))
    end

    return signs
end

# ==============================================================================
# 2. ENERGY LANDSCAPE
# ==============================================================================

"""
    signed_sum(state::Vector{Int8}, numbers::Vector{<:Integer})

Compute sum(state[j] * numbers[j]) (SIGNED, not absolute). Kept separate from the
energy (which is |signed_sum|) so that single-spin-flip deltas can be applied in O(1)
instead of recomputing the full sum on every proposal.
"""
function signed_sum(state::Vector{Int8}, numbers::Vector{T}) where {T<:Integer}
    total = zero(T)
    # Eliminates array bounds checking within expressions.
    @inbounds for j in eachindex(numbers)
        total += T(state[j]) * numbers[j]
    end
    return total
end

"""
    residue(signed::Integer)

Cost function E(s) = |sum(a_j * s_j)|, given the signed sum.
"""
residue(signed::Integer) = abs(signed)

# ==============================================================================
# 3. SIMULATED ANNEALING
# ==============================================================================

"""
    Results of a single fixed-budget SA run.
"""
struct SAResult
    N::Int
    b::Int
    kappa_requested::Union{Float64,Nothing}
    kappa_actual::Float64
    regime::String
    cooling_strategy::String
    A_prefactor::Float64
    cooling_factor::Union{Float64,Nothing}  # geometric: cooling ratio; logarithmic: beta (β), time-scale; nothing for "fixed"
    n_MCS::Int
    n_MCS_used::Int
    T0::Float64
    best_energy::Float64
    final_energy::Float64
    success::Bool
    seed::Int
end

"""
    regime_exponent(regime::String)

Map a regime name to its MCS-scaling exponent z, where n_MCS = A * N^z.
"""
function regime_exponent(regime::String)
    regime == "linear"    && return 1
    regime == "quadratic" && return 2
    regime == "cubic"     && return 3
    error("regime must be one of \"linear\", \"quadratic\", \"cubic\"; got \"$regime\"")
end

"""
    simulated_annealing_fixed_budget(numbers, N, regime, A, rng; T0=nothing, use_kk_init=false)

Run SA for exactly n_MCS = round(A * N^z) Monte Carlo Sweeps (z set by `regime`),
1 MCS = N single-spin-flip attempts, with LINEAR cooling
T(step) = T0 * (1 - step / n_MCS). Stops early ONLY if a perfect partition
(E <= SUCCESS_THRESHOLD) is found.

Arguments:
    numbers::Vector{<:Integer} - problem instance (length N)
    N::Int                     - instance size
    regime::String              - "linear", "quadratic", or "cubic"
    A::Float64                  - prefactor in n_MCS = A * N^z
    rng::AbstractRNG            - RNG for this run (must be thread-local, not shared)
    T0::Union{Float64,Nothing}  - initial temperature (auto: mean(numbers)/4 if nothing)
    use_kk_init::Bool           - false (default): start from a uniformly random state
                                   (rand(rng, Int8[-1, 1], N)); true: start from the KK
                                   heuristic's own partition of `numbers` (see
                                   `kk_partition_signs`), a fast informed warm start
                                   instead of a blind random one

Returns:
    NamedTuple with fields: best_energy, final_energy, success, n_MCS, n_MCS_used, T0_used
"""
function simulated_annealing_fixed_budget(
    numbers::Vector{T},
    N::Int,
    regime::String,
    A::Float64,
    rng::AbstractRNG;
    T0::Union{Float64,Nothing}=nothing,
    use_kk_init::Bool=false,
) where {T<:Integer}

    z = regime_exponent(regime)
    n_MCS = max(1, round(Int, A * (N^z)))

    T0_used = T0 !== nothing ? T0 : Float64(sum(numbers)) / (4.0 * N)
    T0_used = max(T0_used, eps(Float64))  # guard against degenerate all-zero instances

    state = use_kk_init ? kk_partition_signs(numbers) : rand(rng, Int8[-1, 1], N)
    s_sum = signed_sum(state, numbers)
    current_energy = residue(s_sum)
    best_energy = current_energy

    success = current_energy <= SUCCESS_THRESHOLD
    n_used = 0

    if !success
        for step in 1:n_MCS
            n_used = step
            Temp = T0_used * (1.0 - step / n_MCS)  # -> 0 at step == n_MCS

            for _ in 1:N
                i = rand(rng, 1:N)
                # Flipping spin i changes the signed sum by -2*state[i]*numbers[i]
                delta_signed = -2 * T(state[i]) * numbers[i]
                proposed_s_sum = s_sum + delta_signed
                proposed_energy = residue(proposed_s_sum)
                delta_E = proposed_energy - current_energy

                accept = delta_E <= 0 || (Temp > 0.0 && rand(rng) < exp(-Float64(delta_E) / Temp))

                if accept
                    state[i] *= Int8(-1)
                    s_sum = proposed_s_sum
                    current_energy = proposed_energy
                    if current_energy < best_energy
                        best_energy = current_energy
                    end
                end
            end

            if best_energy <= SUCCESS_THRESHOLD
                success = true
                break
            end
        end
    end

    return (best_energy=Float64(best_energy), final_energy=Float64(current_energy),
            success=success, n_MCS=n_MCS, n_MCS_used=n_used, T0_used=T0_used)
end

"""
    simulated_annealing_geometric(numbers, N, regime, A, rng; T0=nothing, cooling_factor=0.995, min_T=1e-20, n_MCS_limit=1_000_000)

Run SA with GEOMETRIC temperature cooling: T(stage) = T0 * cooling_factor^stage,
stage = 0, 1, 2, .... Unlike `simulated_annealing_fixed_budget`, here `regime` and
`A` do NOT set a total budget -- they set

    MCS_per_T = round(A * N^z)     # z from `regime`, via regime_exponent

the number of Monte Carlo Sweeps (1 MCS = N single-spin-flip attempts) performed
AT EACH temperature stage before cooling to the next one. The run advances through
as many stages as needed until one of three stopping conditions is met, whichever
comes first:
    1. a perfect partition is found (E <= SUCCESS_THRESHOLD);
    2. the temperature drops below `min_T`;
    3. the total number of MCS performed (summed across all stages so far) reaches
       the safety cap `n_MCS_limit`.
`min_T` and `n_MCS_limit` are plain safety settings, independent of `regime`/`A`/N,
so they are reported once in the experiment's params.json rather than per CSV row
(see this file's header docstring); `cooling_factor` IS reported per row, since it
is physically meaningful to compare across points.

Arguments:
    numbers::Vector{<:Integer} - problem instance (length N)
    N::Int                      - instance size
    regime::String               - "linear", "quadratic", or "cubic" (sets MCS_per_T's exponent z)
    A::Float64                   - prefactor in MCS_per_T = A * N^z
    rng::AbstractRNG             - RNG for this run (must be thread-local, not shared)
    T0::Union{Float64,Nothing}   - initial temperature (auto: sum(numbers)/(4N) if nothing)
    cooling_factor::Float64      - geometric cooling ratio, 0 < cooling_factor < 1
                                    (default 0.995 is a conservative starting point;
                                    tune per problem size)
    min_T::Float64                - stage loop stops once Temp falls below this
    n_MCS_limit::Int               - safety cap on total MCS performed across all stages
    use_kk_init::Bool             - false (default): start from a uniformly random state
                                      (rand(rng, Int8[-1, 1], N)); true: start from the KK
                                      heuristic's own partition of `numbers` (see
                                      `kk_partition_signs`), a fast informed warm start
                                      instead of a blind random one

Returns:
    NamedTuple with fields: best_energy, final_energy, success, n_MCS, n_MCS_used, T0_used
    Here n_MCS = MCS_per_T (the per-stage budget) and n_MCS_used = the TOTAL MCS
    actually performed across every completed stage -- a different meaning from the
    "fixed" strategy's n_MCS/n_MCS_used (see the header docstring), but kept under
    the same field names so SAResult/the CSV schema stays uniform across strategies.
"""
function simulated_annealing_geometric(
    numbers::Vector{T},
    N::Int,
    regime::String,
    A::Float64,
    rng::AbstractRNG;
    T0::Union{Float64,Nothing}=nothing,
    cooling_factor::Float64=0.995,
    min_T::Float64=1e-20,
    n_MCS_limit::Int=1_000_000,
    use_kk_init::Bool=false,
) where {T<:Integer}

    z = max(0, regime_exponent(regime) - 1)
    MCS_per_T = max(1, round(Int, A * (N^z)))
    n_MCS_limit *= N

    T0_used = T0 !== nothing ? T0 : Float64(sum(numbers)) / (4.0 * N)
    T0_used = max(T0_used, eps(Float64))  # guard against degenerate all-zero instances

    state = use_kk_init ? kk_partition_signs(numbers) : rand(rng, Int8[-1, 1], N)
    s_sum = signed_sum(state, numbers)
    current_energy = residue(s_sum)
    best_energy = current_energy

    success = current_energy <= SUCCESS_THRESHOLD
    total_MCS_done = 0
    Temp = T0_used

    if !success
        while total_MCS_done < n_MCS_limit
            for _ in 1:(MCS_per_T * N)
                i = rand(rng, 1:N)
                # Flipping spin i changes the signed sum by -2*state[i]*numbers[i]
                delta_signed = -2 * T(state[i]) * numbers[i]
                proposed_s_sum = s_sum + delta_signed
                proposed_energy = residue(proposed_s_sum)
                delta_E = proposed_energy - current_energy

                accept = delta_E <= 0 || (Temp > min_T*(cooling_factor^2) && rand(rng) < exp(-Float64(delta_E) / Temp))

                if accept
                    state[i] *= Int8(-1)
                    s_sum = proposed_s_sum
                    current_energy = proposed_energy
                    if current_energy < best_energy
                        best_energy = current_energy
                    end
                end
            end
            total_MCS_done += MCS_per_T

            if best_energy <= SUCCESS_THRESHOLD
                success = true
                break
            end

            if Temp < min_T
                break
            end

            Temp *= cooling_factor
        end
    end

    return (best_energy=Float64(best_energy), final_energy=Float64(current_energy),
            success=success, n_MCS=MCS_per_T, n_MCS_used=total_MCS_done, T0_used=T0_used)
end

"""
    simulated_annealing_logarithmic(numbers, N, regime, A, rng; T0=nothing, cooling_factor=1.0, min_T=1e-20, n_MCS_limit=1_000_000)

Run SA with generalized logarithmic temperature cooling:
    T(stage) = T0 / (1 + cooling_factor * ln(1 + stage)),    stage = 0, 1, 2, ...
a beta-parameterized generalization of the classical Hajek (1988) schedule. The
`cooling_factor` argument plays the role of beta (β), the time-scale parameter
that controls how fast the schedule cools:
    - beta << 1  -> temperature stays close to T0 for many more stages (slow
                    cooling, far more exploration at high temperature);
    - beta == 1  -> cools at roughly the rate of the classical Hajek schedule
                    T0*ln(2)/ln(stage+2);
    - beta  > 1  -> cools faster than classical Hajek.
Well suited to combinatorial optimization problems like NPP because, for small
enough beta, it retains the theoretical (asymptotic) Hajek guarantee of escaping
local optima, at the cost of needing many more stages (and hence a larger
`n_MCS_limit`) to reach a comparably low temperature in practice. Normalized here
so that T(0) = T0 exactly, matching how `simulated_annealing_fixed_budget` and
`simulated_annealing_geometric` both start.

As with `simulated_annealing_geometric`, `regime` and `A` set

    MCS_per_T = round(A * N^z)     # z from `regime`, via regime_exponent

the number of Monte Carlo Sweeps performed AT EACH temperature stage, and the run
advances through stages until a perfect partition is found, Temp falls below
`min_T`, or the total MCS performed reaches the safety cap `n_MCS_limit` --
whichever comes first. See `simulated_annealing_geometric`'s docstring for the
rationale behind keeping `min_T`/`n_MCS_limit` out of the per-row CSV.

Arguments:
    numbers::Vector{<:Integer} - problem instance (length N)
    N::Int                      - instance size
    regime::String               - "linear", "quadratic", or "cubic" (sets MCS_per_T's exponent z)
    A::Float64                   - prefactor in MCS_per_T = A * N^z
    rng::AbstractRNG             - RNG for this run (must be thread-local, not shared)
    T0::Union{Float64,Nothing}   - initial temperature (auto: sum(numbers)/(4N) if nothing)
    cooling_factor::Float64      - beta (β), the logarithmic schedule's time-scale
                                    parameter, must be > 0 (default 1.0; pass
                                    beta << 1, e.g. 0.01-0.1, to spend much more
                                    time near T0 -- tune per N/kappa, see the file
                                    header for the rationale)
    min_T::Float64                - stage loop stops once Temp falls below this
    n_MCS_limit::Int               - safety cap on total MCS performed across all stages
    use_kk_init::Bool             - false (default): start from a uniformly random state
                                      (rand(rng, Int8[-1, 1], N)); true: start from the KK
                                      heuristic's own partition of `numbers` (see
                                      `kk_partition_signs`), a fast informed warm start
                                      instead of a blind random one

Returns:
    NamedTuple with fields: best_energy, final_energy, success, n_MCS, n_MCS_used, T0_used
    (same n_MCS/n_MCS_used convention as simulated_annealing_geometric, i.e.
    n_MCS = MCS_per_T and n_MCS_used = total MCS performed across all stages)
"""
function simulated_annealing_logarithmic(
    numbers::Vector{T},
    N::Int,
    regime::String,
    A::Float64,
    rng::AbstractRNG;
    T0::Union{Float64,Nothing}=nothing,
    cooling_factor::Float64=1.0,
    min_T::Float64=1e-20,
    n_MCS_limit::Int=1_000_000,
    use_kk_init::Bool=false,
) where {T<:Integer}

    cooling_factor > 0 || error("cooling_factor (beta) must be > 0 for cooling_strategy=\"logarithmic\"; got $cooling_factor")

    z = regime_exponent(regime)-1
    MCS_per_T = max(1, round(Int, A * (N^z)))
    n_MCS_limit *= N

    T0_used = T0 !== nothing ? T0 : Float64(sum(numbers)) / (4.0 * N)
    T0_used = max(T0_used, eps(Float64))  # guard against degenerate all-zero instances

    state = use_kk_init ? kk_partition_signs(numbers) : rand(rng, Int8[-1, 1], N)
    s_sum = signed_sum(state, numbers)
    current_energy = residue(s_sum)
    best_energy = current_energy

    success = current_energy <= SUCCESS_THRESHOLD
    total_MCS_done = 0
    Temp = T0_used
    stage = 0

    if !success
        while total_MCS_done < n_MCS_limit
            for _ in 1:(MCS_per_T * N)
                i = rand(rng, 1:N)
                # Flipping spin i changes the signed sum by -2*state[i]*numbers[i]
                delta_signed = -2 * T(state[i]) * numbers[i]
                proposed_s_sum = s_sum + delta_signed
                proposed_energy = residue(proposed_s_sum)
                delta_E = proposed_energy - current_energy

                accept = delta_E <= 0 || (Temp > 0.0 && rand(rng) < exp(-Float64(delta_E) / Temp))

                if accept
                    state[i] *= Int8(-1)
                    s_sum = proposed_s_sum
                    current_energy = proposed_energy
                    if current_energy < best_energy
                        best_energy = current_energy
                    end
                end
            end
            total_MCS_done += MCS_per_T

            if best_energy <= SUCCESS_THRESHOLD
                success = true
                break
            end

            stage += 1
            Temp = T0_used / (1.0 + cooling_factor * log(1.0 + stage))
            if Temp < min_T
                break
            end
        end
    end

    return (best_energy=Float64(best_energy), final_energy=Float64(current_energy),
            success=success, n_MCS=MCS_per_T, n_MCS_used=total_MCS_done, T0_used=T0_used)
end

# ==============================================================================
# 4. EXPERIMENT EXECUTION
# ==============================================================================

"""
    aggregate_results(all_results::Vector{SAResult})

Group results by (N, regime, cooling_strategy, kappa_actual rounded to 6 digits)
and compute:
    P_sat   = fraction of successful runs
    U_f     = mean(best_energy), std(best_energy), stderr(best_energy)

cooling_strategy is included in the grouping key (not just regime) so that, e.g.,
"fixed" and "geometric" runs at the same (N, regime, kappa) are never averaged
together -- they are physically different experiments.

Returns:
    Vector{Dict} - one entry per (N, regime, cooling_strategy, kappa) group
"""
function aggregate_results(all_results::Vector{SAResult})
    groups = Dict{Tuple{Int,String,String,Float64}, Vector{SAResult}}()
    for r in all_results
        key = (r.N, r.regime, r.cooling_strategy, round(r.kappa_actual, digits=6))
        push!(get!(groups, key, SAResult[]), r)
    end

    aggregated = Vector{Dict}()
    for key in sort(collect(keys(groups)))
        N, regime, cooling_strategy, kappa = key
        runs = groups[key]
        n_runs = length(runs)

        successes = count(r -> r.success, runs)
        p_sat = successes / n_runs

        energies = [r.best_energy for r in runs]
        mean_uf = mean(energies)
        std_uf = std(energies)
        stderr_uf = std_uf / sqrt(n_runs)

        final_energies = [r.final_energy for r in runs]
        MCS_used = [r.n_MCS_used for r in runs]
        mean_final = mean(final_energies)
        std_final = std(final_energies)
        mean_MCS_used = mean(MCS_used)
        std_MCS_used = std(MCS_used)

        push!(aggregated, Dict(
            "N" => N,
            "regime" => regime,
            "cooling_strategy" => cooling_strategy,
            "kappa_actual" => kappa,
            "b" => runs[1].b,
            "A_prefactor" => runs[1].A_prefactor,
            "cooling_factor" => runs[1].cooling_factor,
            "n_MCS" => runs[1].n_MCS,
            "mean_MCS_used" => mean_MCS_used,
            "std_MCS_used" => std_MCS_used,
            "count" => n_runs,
            "P_sat" => p_sat,
            "mean_Uf" => mean_uf,
            "std_Uf" => std_uf,
            "stderr_Uf" => stderr_uf,
            "mean_final_energy" => mean_final,
            "std_final_energy" => std_final,
        ))
    end
    return aggregated
end

# ==============================================================================
# 5. FILE OUTPUT
# ==============================================================================

"""
    save_results(results, aggregated, parameters, start)

Save:
- npp_TIMESTAMP_data.csv       : one row per individual SA run
- npp_TIMESTAMP_aggregated.csv : one row per (N, regime, kappa) group (P_sat, U_f, ...)
- npp_TIMESTAMP_params.json    : experiment configuration + metadata
"""
function save_results(results::Vector{SAResult}, aggregated::Vector{Dict}, parameters::Dict, start::DateTime)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    elapsed = now() - start

    mkpath(EXPERIMENTS_FOLDER)

    data_file = joinpath(EXPERIMENTS_FOLDER, "npp_$(timestamp)_data.csv")
    agg_file = joinpath(EXPERIMENTS_FOLDER, "npp_$(timestamp)_aggregated.csv")
    params_file = joinpath(EXPERIMENTS_FOLDER, "npp_$(timestamp)_params.json")

    open(data_file, "w") do io
        println(io, "N,b,kappa_requested,kappa_actual,regime,cooling_strategy,A_prefactor,cooling_factor,n_MCS,n_MCS_used,T0,best_energy,final_energy,success,seed")
        for r in results
            kreq = r.kappa_requested === nothing ? "" : @sprintf("%.6f", r.kappa_requested)
            cfac = r.cooling_factor === nothing ? "" : @sprintf("%.6f", r.cooling_factor)
            println(io, "$(r.N),$(r.b),$kreq,$(r.kappa_actual),$(r.regime),$(r.cooling_strategy),$(r.A_prefactor),$cfac,$(r.n_MCS),$(r.n_MCS_used),$(r.T0),$(r.best_energy),$(r.final_energy),$(r.success),$(r.seed)")
        end
    end

    open(agg_file, "w") do io
        # mean_MCS_used/std_MCS_used were always computed by aggregate_results (see
        # above) but previously dropped on the way to disk -- restored here so the
        # aggregated CSV actually carries the search-cost metric (MCS performed per
        # (N, kappa) point) alongside P_sat/U_f, mirroring Exhaustive_NPP.jl's
        # mean_nodes_visited column for CKK.
        println(io, "N,regime,cooling_strategy,kappa_actual,b,A_prefactor,cooling_factor,n_MCS,mean_MCS_used,std_MCS_used,count,P_sat,mean_Uf,std_Uf,stderr_Uf,mean_final_energy,std_final_energy")
        for a in aggregated
            cfac = a["cooling_factor"] === nothing ? "" : a["cooling_factor"]
            println(io, "$(a["N"]),$(a["regime"]),$(a["cooling_strategy"]),$(a["kappa_actual"]),$(a["b"]),$(a["A_prefactor"]),$cfac,$(a["n_MCS"]),$(a["mean_MCS_used"]),$(a["std_MCS_used"]),$(a["count"]),$(a["P_sat"]),$(a["mean_Uf"]),$(a["std_Uf"]),$(a["stderr_Uf"]),$(a["mean_final_energy"]),$(a["std_final_energy"])")
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
# 6. MAIN SWEEP
# ==============================================================================

"""
    resolve_b_values_for_N(b_values, idx_N::Int)

Resolve which list of b values applies to the N_values[idx_N]-th problem size,
given the two shapes `b_values` can take (see this file's header docstring):
    (a) Vector{Int}          -> the same list applies to every N; always returned as-is.
    (b) Vector{Vector{Int}}  -> per-N lists, selected by position. If idx_N exceeds
                                length(b_values), the LAST inner list provided is
                                reused (so a shorter list than N_values is valid and
                                simply repeats its final entry for the remaining N).

Arguments:
    b_values::Union{Vector{Int},Vector{Vector{Int}}} - as described above
    idx_N::Int                                        - 1-based position of the
                                                         current N within N_values

Returns:
    Vector{Int} - the b values to scan for this particular N
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
        N_values, kappa_values, regime;
        b_values=nothing, A=10.0, samples=100, T0=nothing, seed=16,
        cooling_strategy="fixed", cooling_factor=0.995, min_T=1e-20, n_MCS_limit=1_000_000,
    )

Run the full (N x kappa) sweep for a single `regime` and `cooling_strategy`,
threaded over samples.

Arguments:
    N_values::Vector{Int}                  - problem sizes to scan
    kappa_values::Vector{Float64}          - kappa values to scan (`nothing` to use b_values
                                              instead); always a single flat list shared
                                              across every N (unlike b_values, see below)
    regime::String                         - "linear", "quadratic", or "cubic". For
                                              cooling_strategy="fixed" this sets the TOTAL
                                              budget (n_MCS = A*N^z, as before); for
                                              "geometric"/"logarithmic" it instead sets the
                                              per-stage budget (MCS_per_T = A*N^z) -- see
                                              this file's header docstring for the full
                                              rationale.
    b_values::Union{Vector{Int},Vector{Vector{Int}},Nothing}
                                            - alternative to kappa_values: explicit bit
                                              counts (used only where `kappa_values ===
                                              nothing`). Either a single flat list (shared
                                              across every N) or a list of per-N lists (see
                                              `resolve_b_values_for_N`).
    A::Float64                             - prefactor in n_MCS or MCS_per_T = A * N^z
                                              (meaning depends on cooling_strategy, as above)
    samples::Int                           - independent instances per (N, kappa) point
    T0::Union{Float64,Nothing}             - initial temperature (auto per-instance if nothing)
    seed::Int                              - master seed (per-run RNGs are derived from this)
    cooling_strategy::String               - "fixed", "geometric", or "logarithmic"
                                              (see this file's header docstring)
    cooling_factor::Float64                - geometric: cooling ratio (0 < cooling_factor < 1).
                                              logarithmic: beta (β), the time-scale parameter
                                              of T(stage) = T0/(1 + beta*ln(1+stage)); use
                                              beta << 1 (e.g. 0.01-0.1) to spend much more time
                                              near T0. Not used when cooling_strategy="fixed".
    min_T::Float64                         - stage-loop stopping temperature, used only
                                              when cooling_strategy is "geometric" or
                                              "logarithmic"; recorded in params.json only
                                              (not per CSV row -- it is the same for every
                                              point in the sweep, see header docstring)
    n_MCS_limit::Int                       - safety cap on total MCS per run, used only
                                              when cooling_strategy is "geometric" or
                                              "logarithmic"; recorded in params.json only,
                                              same rationale as min_T
    use_kk_init::Bool                      - false (default): every run starts from a
                                              uniformly random state; true: every run
                                              starts from the KK heuristic's own
                                              partition of that run's instance (see
                                              `kk_partition_signs`), recorded in
                                              params.json

Returns:
    (all_results::Vector{SAResult}, aggregated::Vector{Dict}, data_file, agg_file, params_file)
"""
function run_phase_diagram_sweep(
    N_values::Vector{Int},
    kappa_values::Union{Vector{Float64},Nothing},
    regime::String;
    b_values::Union{Vector{Int},Vector{Vector{Int}},Nothing}=nothing,
    A::Float64=10.0,
    samples::Int=100,
    T0::Union{Float64,Nothing}=nothing,
    seed::Int=16,
    cooling_strategy::String="fixed",
    cooling_factor::Float64=0.995,
    min_T::Float64=1e-20,
    n_MCS_limit::Int=1_000_000,
    use_kk_init::Bool=false,
)
    if kappa_values === nothing && b_values === nothing
        error("Must supply either kappa_values or b_values.")
    end
    cooling_strategy in ("fixed", "geometric", "logarithmic") ||
        error("cooling_strategy must be one of \"fixed\", \"geometric\", \"logarithmic\"; got \"$cooling_strategy\"")
    using_kappa = kappa_values !== nothing

    # Precompute the scan values for each N up front
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
        "regime" => regime,
        "cooling_strategy" => cooling_strategy,
        "A_prefactor" => A,
        "cooling_factor" => cooling_strategy in ("geometric", "logarithmic") ? cooling_factor : nothing,
        "min_T" => cooling_strategy == "fixed" ? nothing : min_T,
        "n_MCS_limit" => cooling_strategy == "fixed" ? nothing : n_MCS_limit,
        "samples" => samples,
        "T0" => T0 === nothing ? "heuristic (sum(numbers)/(4N))" : T0,
        "seed" => seed,
        "use_kk_init" => use_kk_init,
    )

    println("\n" * "="^80)
    println("NPP PHASE-DIAGRAM SWEEP -- regime=$regime, cooling_strategy=$cooling_strategy, A=$A, samples=$samples")
    println("N values:     $N_values")
    println(using_kappa ? "kappa values: $kappa_values" : "b values:     $b_values")
    if cooling_strategy != "fixed"
        print_cooling_factor = cooling_strategy in ("geometric", "logarithmic") ? ", cooling_factor=$cooling_factor" : ""
        print("min_T=$min_T, n_MCS_limit=$n_MCS_limit")
        println(print_cooling_factor)
    end
    println("threads:      $(Threads.nthreads())")
    println("="^80 * "\n")

    all_results = Vector{SAResult}(undef, total_points)
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
                res = if cooling_strategy == "geometric"
                    simulated_annealing_geometric(numbers, N, regime, A, rng;
                        T0=T0, cooling_factor=cooling_factor, min_T=min_T, n_MCS_limit=n_MCS_limit,
                        use_kk_init=use_kk_init)
                elseif cooling_strategy == "logarithmic"
                    simulated_annealing_logarithmic(numbers, N, regime, A, rng;
                        T0=T0, cooling_factor=cooling_factor, min_T=min_T, n_MCS_limit=n_MCS_limit,
                        use_kk_init=use_kk_init)
                else
                    simulated_annealing_fixed_budget(numbers, N, regime, A, rng; T0=T0, use_kk_init=use_kk_init)
                end

                run_cooling_factor = cooling_strategy in ("geometric", "logarithmic") ? cooling_factor : nothing
                all_results[base_idx + s] = SAResult(
                    N, b_used, kappa_requested, kappa_actual, regime, cooling_strategy,
                    A, run_cooling_factor, res.n_MCS, res.n_MCS_used, res.T0_used,
                    res.best_energy, res.final_energy, res.success, run_seed,
                )
            end
            idx_counter += samples

            n_success = count(r -> r.success, all_results[base_idx+1:base_idx+samples])
            @printf("  N=%-5d kappa=%.4f (b=%-2d)  P_sat=%.3f  <best_E>=%.3e  [%.1fs elapsed]\n",
                    N, kappa_actual, b_used, n_success / samples,
                    mean(r.best_energy for r in all_results[base_idx+1:base_idx+samples]),
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
# 8. EXAMPLE USAGE
# ==============================================================================

if abspath(PROGRAM_FILE) == @__FILE__

    regime = "linear"         # "linear", "quadratic", "cubic"
    cooling_strategy = "geometric"  # "fixed", "geometric", "logarithmic"
    A = 1.0
    samples = 100
    multi = false

    if multi
        cooling_rates = cooling_strategy == "logarithmic" ?
            [0.1, 0.05, 0.03, 0.01] :
            [0.99999999, 0.9999999995, 0.9999999995, 0.99999999999]
        for (i, cooling_rate) in enumerate(cooling_rates)
            base_N = 28 + 2*(i-1)
            N_values = collect(base_N:(base_N+1))
            b_values = [collect((base_N-4):(base_N-2)), collect((base_N-3):(base_N-1))]

            results, aggregated, df, af, pf = run_phase_diagram_sweep(
                N_values, nothing, regime;
                A=A, samples=samples, b_values=b_values, seed=16,
                cooling_strategy=cooling_strategy, cooling_factor=cooling_rate,
                min_T=1e-20, n_MCS_limit=(i*10_000_000),
            )
            println("Sweep completed successfully!")
        end
    else
        N_values = collect(29:1:31)
        b_values = Vector{Vector{Int}}()

        for N in N_values
            b_values_N = collect((N-4):(N-1))
            push!(b_values, b_values_N)
        end

        # for N in 28:35
        #     push!(N_values, N)
        #     push!(b_values, collect((N-5):(N-1)))
        # end

        cooling_rate = 0.9999999999999999
        results, aggregated, df, af, pf = run_phase_diagram_sweep(
            N_values, nothing, regime;
            A=A, samples=samples, b_values=b_values, seed=16,
            cooling_strategy=cooling_strategy, cooling_factor=cooling_rate,
            min_T=1e-12, n_MCS_limit=10_000_000, use_kk_init=true,
        )
    end
end

# Geometric, quadratic, A=25, n_MCS_limit=1_000_000 * N^(z-1)
#   10-15 0.995
#   16-18 0.999
#   19-21 0.9995
#   22-24 0.9999
#   25-27 0.99995
#   28-29 0.99999
#   30-30 0.999995
#   31-32 0.999999
#   33-35 0.9999999

# Geometric, quadratic, A=1, fixed n_MCS_limit=1_000_000
#   10-17 0.9999
#   18-20 0.99999
#   21-23 0.999999
#   24-25 0.9999995
#   26-27 0.9999999
#   28-29 0.99999995
#   30-31 0.99999999
#   32-33 0.999999995
#   34-35 0.999999999

# Geometric, linear, A=1, n_MCS_limit=1_000_000 * N^(z-1)
#   10-13 0.99999
#   14-17 0.999995
#   18-20 0.999999
#   21-22 0.9999999
#   23-24 0.99999995
#   25-26 0.99999999
#   27-28 0.999999995
#   29-30 0.999999999
#   31-32 0.9999999995
#   33-34 0.9999999999
#   35-35 0.99999999999

# Geometric, quadratic, A=1, n_MCS_limit=1_000_000 * N^(z-1)
#   10-17 0.9999
#   18-20 0.99999
#   21-24 0.999999
#   25-28 0.9999999
#   28-31 0.99999999
#   32-35 0.999999999

# Geometric cubic A=1
#   10-17 0.9999
#   18-20 0.99999
#   21-24 0.999999
#   25-28 0.9999999
#   28-31 0.99999999
#   32-35 0.999999999