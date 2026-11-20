"""
trajectory_SA4NPP.jl

Energy-trajectory experiment for SA applied to the Number Partitioning Problem (NPP).
Companion to SA4NPP_phase_diagram.jl, reusing the same instance model and SA engines,
but instead of reporting only the end-of-run summary (best_energy, success, ...), it
logs E(t) -- both the running incumbent (`best_energy`) and the literal current state's
energy (`current_energy`) -- at a log-spaced grid of MCS checkpoints over the course of
each run. This is the data needed to plot energy-vs-runtime stagnation curves and to
estimate kappa_c from the log(E) vs log(t) family of curves at fixed N.

- Supports all three cooling strategies from SA4NPP_phase_diagram.jl ("fixed",
  "geometric", "logarithmic") and all three regimes ("linear"/"quadratic"/"cubic"),
  chosen explicitly by the user (no implicit restriction).

- Checkpoint grid: for a given (N, b, regime, cooling_strategy) configuration, a single
  SHARED log-spaced grid of MCS-step checkpoints is built once, from step 1 up to a
  config-dependent T_max:
      "fixed"                    -> T_max = n_MCS = round(A * N^z)      (the total budget)
      "geometric"/"logarithmic"  -> T_max = n_MCS_limit                 (the safety cap)
  Every sample (replicate) of that configuration is checkpointed on EXACTLY this same
  grid. This means aggregation across samples is a plain groupby-mean over (config, step)
  -- no interpolation is needed in Julia, and the Python side only has to plot / fit.

- Early stop handling: if a run finds a perfect partition (or otherwise stops) 
  before reaching a later checkpoint, the remaining checkpoints for that run 
  are filled FORWARD with the run's final (best_energy, current_energy) -- a flat
  continuation, which is the physically sensible choice (once solved/stopped, the
  incumbent energy does not change) and keeps every sample's trajectory defined on the
  full shared grid for clean averaging.

- Both `best_energy` (monotonic running incumbent) and `current_energy` (literal,
  possibly non-monotonic state energy) are logged at every checkpoint, so the Python
  plotters can choose either.

- Output files (Experiments/ folder), using the naming convention:
      trajectory_TIMESTAMP_data.csv        - one row per (sample, checkpoint): raw
      trajectory_TIMESTAMP_aggregated.csv  - one row per (config, checkpoint): stats
      trajectory_TIMESTAMP_params.json     - experiment configuration + metadata
  All statistics (mean/std/stderr across samples at each step) are computed here in
  Julia; the Python plotters only plot and (for kappa_c) do a simple linear-interpolation
  crossing on the already-aggregated data.

- Configuration of which (N, b) points to trace is always MANUAL: the user supplies
  `configs::Vector{Pair{Int,Vector{Int}}}` so "easy" and "hard" b's can be mixed per 
  N based on the phase-diagram results.

- Multithreading: samples for a given (N, b) config are embarrassingly parallel, same
  per-task-RNG approach as SA4NPP_phase_diagram.jl. Launch Julia with
    julia -t auto trajectory_SA4NPP.jl
  to use all available cores.
"""

using Random
using Statistics
using Printf
using JSON
using Dates
using Base.Threads

EXPERIMENTS_FOLDER = joinpath(@__DIR__, "Experiments")
SUCCESS_THRESHOLD = 1.1
BIGINT_THRESHOLD_BITS = 100
const _bigint_warned = Ref(false)

# ==============================================================================
# 1. PROBLEM INSTANCE GENERATION
# ==============================================================================

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
# 2. ENERGY LANDSCAPE
# ==============================================================================

function signed_sum(state::Vector{Int8}, numbers::Vector{T}) where {T<:Integer}
    total = zero(T)
    @inbounds for j in eachindex(numbers)
        total += T(state[j]) * numbers[j]
    end
    return total
end

residue(signed::Integer) = abs(signed)

function regime_exponent(regime::String)
    regime == "linear"    && return 1
    regime == "quadratic" && return 2
    regime == "cubic"     && return 3
    error("regime must be one of \"linear\", \"quadratic\", \"cubic\"; got \"$regime\"")
end

# ==============================================================================
# 3. CHECKPOINT GRID
# ==============================================================================

"""
    log_checkpoints(T_max::Int, n_points::Int)

Build a log-spaced, strictly increasing grid of Int MCS-step checkpoints covering
[1, T_max], with at most `n_points` distinct points (duplicates from rounding are
removed, which is why the returned vector can be shorter than `n_points`). T_max
itself is always included as the last point, so every trajectory's final (plateau)
value is captured exactly at the config's budget / safety cap.
"""
function log_checkpoints(T_max::Int, n_points::Int)
    T_max = max(T_max, 1)
    n_points = max(2, min(n_points, T_max))
    raw = exp.(range(log(1.0), log(Float64(T_max)), length=n_points))
    pts = sort(unique(round.(Int, raw)))
    pts[1] < 1 && (pts[1] = 1)
    if pts[end] != T_max
        push!(pts, T_max)
    end
    return pts
end

struct TrajPoint
    step::Int
    best_energy::Float64
    current_energy::Float64
end

# ==============================================================================
# 4. SIMULATED ANNEALING WITH TRAJECTORY LOGGING
# ==============================================================================

"""
    simulated_annealing_fixed_budget_traj(numbers, N, regime, A, rng, checkpoints; T0=nothing)

Same schedule as `simulated_annealing_fixed_budget` in SA4NPP_phase_diagram.jl
(T(step) = T0*(1 - step/n_MCS), n_MCS = round(A*N^z)), but additionally records
(step, best_energy, current_energy) at every step in `checkpoints` that is reached,
and fills any checkpoints beyond an early stop with the run's final values.
"""
function simulated_annealing_fixed_budget_traj(
    numbers::Vector{T},
    N::Int,
    regime::String,
    A::Float64,
    rng::AbstractRNG,
    checkpoints::Vector{Int};
    T0::Union{Float64,Nothing}=nothing,
) where {T<:Integer}

    z = regime_exponent(regime)
    n_MCS = max(1, round(Int, A * (N^z)))

    T0_used = T0 !== nothing ? T0 : Float64(sum(numbers)) / (4.0 * N)
    T0_used = max(T0_used, eps(Float64))

    state = rand(rng, Int8[-1, 1], N)
    s_sum = signed_sum(state, numbers)
    current_energy = residue(s_sum)
    best_energy = current_energy
    success = current_energy <= SUCCESS_THRESHOLD

    traj = TrajPoint[]
    cp_idx = 1
    ncp = length(checkpoints)
    n_used = 0

    if !success
        for step in 1:n_MCS
            n_used = step
            Temp = T0_used * (1.0 - step / n_MCS)

            for _ in 1:N
                i = rand(rng, 1:N)
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

            while cp_idx <= ncp && checkpoints[cp_idx] == step
                push!(traj, TrajPoint(step, Float64(best_energy), Float64(current_energy)))
                cp_idx += 1
            end

            if best_energy <= SUCCESS_THRESHOLD
                success = true
                break
            end
        end
    end

    # Flat continuation for any checkpoints beyond an early stop (or beyond step 0
    # if the initial state was already a success).
    while cp_idx <= ncp
        push!(traj, TrajPoint(checkpoints[cp_idx], Float64(best_energy), Float64(current_energy)))
        cp_idx += 1
    end

    return (best_energy=Float64(best_energy), final_energy=Float64(current_energy),
            success=success, n_MCS=n_MCS, n_MCS_used=n_used, T0_used=T0_used, traj=traj)
end

"""
    _sa_staged_traj(numbers, N, MCS_per_T, temp_fn, rng, checkpoints; T0, min_T, n_MCS_limit)

Shared engine for the "geometric" and "logarithmic" cooling strategies (they differ
only in how temperature evolves from stage to stage). `temp_fn(stage, T0)` returns
the temperature for a given stage (stage = 0, 1, 2, ...). Stops at whichever of the
three original stopping conditions comes first (perfect partition / Temp < min_T /
total MCS >= n_MCS_limit), logs trajectory checkpoints every single MCS (not just at
stage boundaries), and flat-fills any remaining checkpoints on early stop.
"""
function _sa_staged_traj(
    numbers::Vector{T},
    N::Int,
    MCS_per_T::Int,
    temp_fn::Function,
    rng::AbstractRNG,
    checkpoints::Vector{Int};
    T0::Union{Float64,Nothing}=nothing,
    min_T::Float64=1e-20,
    n_MCS_limit::Int=1_000_000,
) where {T<:Integer}

    T0_used = T0 !== nothing ? T0 : Float64(sum(numbers)) / (4.0 * N)
    T0_used = max(T0_used, eps(Float64))

    state = rand(rng, Int8[-1, 1], N)
    s_sum = signed_sum(state, numbers)
    current_energy = residue(s_sum)
    best_energy = current_energy
    success = current_energy <= SUCCESS_THRESHOLD

    traj = TrajPoint[]
    cp_idx = 1
    ncp = length(checkpoints)
    total_done = 0
    stage = 0
    Temp = T0_used

    if !success
        while total_done < n_MCS_limit && Temp >= min_T
            for _ in 1:MCS_per_T
                for _ in 1:N
                    i = rand(rng, 1:N)
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
                total_done += 1

                while cp_idx <= ncp && checkpoints[cp_idx] == total_done
                    push!(traj, TrajPoint(total_done, Float64(best_energy), Float64(current_energy)))
                    cp_idx += 1
                end

                (best_energy <= SUCCESS_THRESHOLD || total_done >= n_MCS_limit) && break
            end

            if best_energy <= SUCCESS_THRESHOLD
                success = true
                break
            end
            total_done >= n_MCS_limit && break

            stage += 1
            Temp = temp_fn(stage, T0_used, Temp)
        end
    end

    while cp_idx <= ncp
        push!(traj, TrajPoint(checkpoints[cp_idx], Float64(best_energy), Float64(current_energy)))
        cp_idx += 1
    end

    return (best_energy=Float64(best_energy), final_energy=Float64(current_energy),
            success=success, n_MCS=MCS_per_T, n_MCS_used=total_done, T0_used=T0_used, traj=traj)
end

geometric_temp_fn(cooling_factor::Float64) = (_stage, _T0, Temp) -> Temp * cooling_factor

"""
    logarithmic_temp_fn(cooling_factor::Float64)

Generalized logarithmic cooling: T(stage) = T0 / (1 + cooling_factor * ln(1 + stage)).
`cooling_factor` plays the role of beta (β) -- see SA4NPP_phase_diagram.jl's
`simulated_annealing_logarithmic` docstring for the full rationale: beta << 1 keeps
the temperature near T0 for far more stages than the classical Hajek schedule
(T0*ln(2)/ln(stage+2), recovered approximately at beta == 1).
"""
logarithmic_temp_fn(cooling_factor::Float64) = (stage, T0, _Temp) -> T0 / (1.0 + cooling_factor * log(1.0 + stage))

# ==============================================================================
# 5. PER-CONFIG / PER-SAMPLE RESULT RECORD
# ==============================================================================

struct TrajRun
    N::Int
    b::Int
    kappa_actual::Float64
    regime::String
    cooling_strategy::String
    A_prefactor::Float64
    cooling_factor::Union{Float64,Nothing}
    sample_id::Int
    seed::Int
    success::Bool
    n_MCS_used::Int
    T0::Float64
    traj::Vector{TrajPoint}
end

"""
    resolve_b_values_for_N(b_values, idx_N::Int)

Identical helper to the one in SA4NPP_phase_diagram.jl (see that file for the full
docstring): resolves which list of b values applies to N_values[idx_N].
"""
function resolve_b_values_for_N(b_values::Union{Vector{Int},Vector{Vector{Int}}}, idx_N::Int)
    if b_values isa Vector{Vector{Int}}
        return idx_N <= length(b_values) ? b_values[idx_N] : b_values[end]
    else
        return b_values
    end
end

# ==============================================================================
# 6. EXPERIMENT EXECUTION
# ==============================================================================

"""
    run_trajectory_experiment(
        N_values, b_values, regime, cooling_strategy;
        A=10.0, samples=20, n_points=60, T0=nothing, seed=16,
        cooling_factor=0.995, min_T=1e-20, n_MCS_limit=1_000_000,
    )

Run the manually-specified (N, b) trajectory sweep for a single `regime` and
`cooling_strategy`. For every (N, b) pair, builds one shared log-spaced checkpoint
grid (see `log_checkpoints`), runs `samples` independent replicates (threaded), and
returns both the raw per-sample trajectories and the per-step aggregated statistics.

Arguments:
    N_values::Vector{Int}                 - problem sizes to trace
    b_values::Union{Vector{Int},Vector{Vector{Int}}}
                                           - bits per N; flat list shared across every
                                             N, or one list per N (see
                                             `resolve_b_values_for_N`). This is how you
                                             manually pick "easy" and "hard" b's per N.
    regime::String                        - "linear", "quadratic", or "cubic"
    cooling_strategy::String              - "fixed", "geometric", or "logarithmic"
    A::Float64                            - prefactor in n_MCS/MCS_per_T = A*N^z
    samples::Int                          - independent replicate runs per (N,b), averaged
    n_points::Int                         - max number of log-spaced checkpoints per run
    T0::Union{Float64,Nothing}            - initial temperature (auto per-instance if nothing)
    seed::Int                             - master seed
    cooling_factor::Float64               - geometric: cooling ratio. logarithmic: beta (β),
                                             the time-scale parameter of
                                             T(stage)=T0/(1+beta*ln(1+stage)); use beta << 1
                                             to spend much more time near T0. Unused for "fixed".
    min_T::Float64                        - stage-loop stop temperature (geometric/log only)
    n_MCS_limit::Int                      - safety cap AND checkpoint-grid upper bound for
                                             geometric/logarithmic (see file header)

Returns:
    (all_runs::Vector{TrajRun}, aggregated::Vector{Dict}, data_file, agg_file, params_file)
"""
function run_trajectory_experiment(
    N_values::Vector{Int},
    b_values::Union{Vector{Int},Vector{Vector{Int}}},
    regime::String,
    cooling_strategy::String;
    A::Float64=10.0,
    samples::Int=20,
    n_points::Int=60,
    T0::Union{Float64,Nothing}=nothing,
    seed::Int=16,
    cooling_factor::Float64=0.995,
    min_T::Float64=1e-20,
    n_MCS_limit::Int=1_000_000,
)
    cooling_strategy in ("fixed", "geometric", "logarithmic") ||
        error("cooling_strategy must be one of \"fixed\", \"geometric\", \"logarithmic\"; got \"$cooling_strategy\"")

    b_values_by_N = [resolve_b_values_for_N(b_values, idx) for (idx, _) in enumerate(N_values)]
    total_configs = sum(length(bs) for bs in b_values_by_N)

    parameters = Dict(
        "N_values" => N_values,
        "b_values" => b_values,
        "regime" => regime,
        "cooling_strategy" => cooling_strategy,
        "A_prefactor" => A,
        "samples" => samples,
        "n_points" => n_points,
        "cooling_factor" => cooling_strategy in ("geometric", "logarithmic") ? cooling_factor : nothing,
        "min_T" => cooling_strategy == "fixed" ? nothing : min_T,
        "n_MCS_limit" => cooling_strategy == "fixed" ? nothing : n_MCS_limit,
        "T0" => T0 === nothing ? "heuristic (sum(numbers)/(4N))" : T0,
        "seed" => seed,
    )

    cooling_factor_info = cooling_factor === nothing ? "" : " cooling_factor=$cooling_factor,"
    println("\n" * "="^80)
    println("NPP TRAJECTORY EXPERIMENT -- regime=$regime, A=$A, cooling_strategy=$cooling_strategy,$cooling_factor_info samples=$samples")
    println("N values: $N_values")
    println("b values: $b_values")
    println("threads:  $(Threads.nthreads())")
    println("="^80 * "\n")

    all_runs = Vector{TrajRun}(undef, total_configs * samples)
    start_time = now()

    temp_fn = cooling_strategy == "geometric" ? geometric_temp_fn(cooling_factor) :
              cooling_strategy == "logarithmic" ? logarithmic_temp_fn(cooling_factor) : nothing

    idx_counter = 0
    for (idx_N, N) in enumerate(N_values)
        z = regime_exponent(regime)
        for b in b_values_by_N[idx_N]
            _, _, kappa_actual = resolve_b_kappa(N; b=b)

            T_max = cooling_strategy == "fixed" ? max(1, round(Int, A * (N^z))) : n_MCS_limit
            checkpoints = log_checkpoints(T_max, n_points)

            base_idx = idx_counter
            @threads for s in 1:samples
                run_seed = seed + hash((N, b, s)) % 1_000_000_000
                rng = MersenneTwister(run_seed)
                numbers = generate_instance(N, b, rng)

                res = if cooling_strategy == "fixed"
                    simulated_annealing_fixed_budget_traj(numbers, N, regime, A, rng, checkpoints; T0=T0)
                else
                    MCS_per_T = max(1, round(Int, A * (N^(z-1))))
                    _sa_staged_traj(numbers, N, MCS_per_T, temp_fn, rng, checkpoints;
                                     T0=T0, min_T=min_T, n_MCS_limit=n_MCS_limit)
                end

                all_runs[base_idx + s] = TrajRun(
                    N, b, kappa_actual, regime, cooling_strategy, A,
                    cooling_strategy in ("geometric", "logarithmic") ? cooling_factor : nothing,
                    s, run_seed, res.success, res.n_MCS_used, res.T0_used, res.traj,
                )
            end
            idx_counter += samples

            n_success = count(r -> r.success, all_runs[base_idx+1:base_idx+samples])
            @printf("  N=%-5d b=%-4d (kappa=%.4f)  checkpoints=%-4d P_sat=%.2f  [%.1fs elapsed]\n",
                    N, b, kappa_actual, length(checkpoints), n_success / samples,
                    Dates.value(now() - start_time) / 1000.0)
        end
    end

    println("\n" * "="^80)
    println("AGGREGATING AND SAVING RESULTS")
    println("="^80)

    aggregated = aggregate_trajectories(all_runs)
    data_file, agg_file, params_file = save_trajectory_results(all_runs, aggregated, parameters, start_time)

    println("Data file:       $data_file")
    println("Aggregated file: $agg_file")
    println("Params file:     $params_file")
    println("="^80 * "\n")

    return all_runs, aggregated, data_file, agg_file, params_file
end

# ==============================================================================
# 7. AGGREGATION
# ==============================================================================

"""
    aggregate_trajectories(all_runs::Vector{TrajRun})

Group by (N, b, regime, cooling_strategy, step) and compute mean/std/stderr of both
best_energy and current_energy across samples, plus P_sat (fraction of runs that had
already succeeded by that step -- derived from n_MCS_used <= step). Since every run
in a group shares the exact same checkpoint grid (see `log_checkpoints`), this is a
plain groupby-mean: no interpolation is needed.

Returns:
    Vector{Dict} - one entry per (N, b, regime, cooling_strategy, step) group, sorted
"""
function aggregate_trajectories(all_runs::Vector{TrajRun})
    groups = Dict{Tuple{Int,Int,String,String,Int}, Vector{Tuple{TrajRun,TrajPoint}}}()
    for r in all_runs
        for tp in r.traj
            key = (r.N, r.b, r.regime, r.cooling_strategy, tp.step)
            push!(get!(groups, key, Tuple{TrajRun,TrajPoint}[]), (r, tp))
        end
    end

    aggregated = Vector{Dict}()
    for key in sort(collect(keys(groups)))
        N, b, regime, cooling_strategy, step = key
        entries = groups[key]
        n = length(entries)

        best_es = [tp.best_energy for (_, tp) in entries]
        cur_es = [tp.current_energy for (_, tp) in entries]
        n_solved_by_now = count(((r, _),) -> r.n_MCS_used <= step && r.success, entries)

        r1 = entries[1][1]
        push!(aggregated, Dict(
            "N" => N,
            "b" => b,
            "kappa_actual" => r1.kappa_actual,
            "regime" => regime,
            "cooling_strategy" => cooling_strategy,
            "A_prefactor" => r1.A_prefactor,
            "cooling_factor" => r1.cooling_factor,
            "step" => step,
            "count" => n,
            "mean_best_energy" => mean(best_es),
            "std_best_energy" => std(best_es),
            "stderr_best_energy" => std(best_es) / sqrt(n),
            "mean_current_energy" => mean(cur_es),
            "std_current_energy" => std(cur_es),
            "stderr_current_energy" => std(cur_es) / sqrt(n),
            "P_sat_by_step" => n_solved_by_now / n,
        ))
    end
    return aggregated
end

# ==============================================================================
# 8. FILE OUTPUT
# ==============================================================================

"""
    save_trajectory_results(all_runs, aggregated, parameters, start)

Save:
- trajectory_TIMESTAMP_data.csv        : one row per (sample, checkpoint) -- raw
- trajectory_TIMESTAMP_aggregated.csv  : one row per (config, checkpoint) -- stats
- trajectory_TIMESTAMP_params.json     : experiment configuration + metadata
"""
function save_trajectory_results(all_runs::Vector{TrajRun}, aggregated::Vector{Dict}, parameters::Dict, start::DateTime)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    elapsed = now() - start

    mkpath(EXPERIMENTS_FOLDER)

    data_file = joinpath(EXPERIMENTS_FOLDER, "trajectory_$(timestamp)_data.csv")
    agg_file = joinpath(EXPERIMENTS_FOLDER, "trajectory_$(timestamp)_aggregated.csv")
    params_file = joinpath(EXPERIMENTS_FOLDER, "trajectory_$(timestamp)_params.json")

    open(data_file, "w") do io
        println(io, "N,b,kappa_actual,regime,cooling_strategy,A_prefactor,cooling_factor,sample_id,seed,success,n_MCS_used,T0,step,best_energy,current_energy")
        for r in all_runs
            cfac = r.cooling_factor === nothing ? "" : @sprintf("%.6f", r.cooling_factor)
            for tp in r.traj
                println(io, "$(r.N),$(r.b),$(r.kappa_actual),$(r.regime),$(r.cooling_strategy),$(r.A_prefactor),$cfac,$(r.sample_id),$(r.seed),$(r.success),$(r.n_MCS_used),$(r.T0),$(tp.step),$(tp.best_energy),$(tp.current_energy)")
            end
        end
    end

    open(agg_file, "w") do io
        println(io, "N,b,kappa_actual,regime,cooling_strategy,A_prefactor,cooling_factor,step,count,mean_best_energy,std_best_energy,stderr_best_energy,mean_current_energy,std_current_energy,stderr_current_energy,P_sat_by_step")
        for a in aggregated
            cfac = a["cooling_factor"] === nothing ? "" : a["cooling_factor"]
            println(io, "$(a["N"]),$(a["b"]),$(a["kappa_actual"]),$(a["regime"]),$(a["cooling_strategy"]),$(a["A_prefactor"]),$cfac,$(a["step"]),$(a["count"]),$(a["mean_best_energy"]),$(a["std_best_energy"]),$(a["stderr_best_energy"]),$(a["mean_current_energy"]),$(a["std_current_energy"]),$(a["stderr_current_energy"]),$(a["P_sat_by_step"])")
        end
    end

    params_dict = merge(parameters, Dict(
        "timestamp" => timestamp,
        "elapsed_time_seconds" => Dates.value(elapsed) / 1000.0,
        "n_individual_runs" => length(all_runs),
    ))
    open(params_file, "w") do io
        write(io, JSON.json(params_dict, 2))
    end

    println("\n$(length(all_runs)) individual runs ($(sum(length(r.traj) for r in all_runs)) checkpoint rows) saved to CSV")
    return data_file, agg_file, params_file
end

# ==============================================================================
# 9. EXAMPLE USAGE
# ==============================================================================

if abspath(PROGRAM_FILE) == @__FILE__

    N_values = [28]
    b_values = [
        [24, 25, 26, 27, 28],   # N=28
    ]
    regime = "quadratic"            # "linear", "quadratic", or "cubic"
    A = 25.0
    cooling_strategy = "geometric"      # "fixed", "geometric", or "logarithmic"
    cooling_factor = 0.999999999999
    samples = 100
    n_points = 100

    runs, aggregated, df, af, pf = run_trajectory_experiment(
        N_values, b_values, regime, cooling_strategy;
        A=A, samples=samples, n_points=n_points, seed=16,
        cooling_factor=cooling_factor, min_T=1e-12, n_MCS_limit=28_000_000,
    )

    println("Trajectory experiment completed successfully!")
end
