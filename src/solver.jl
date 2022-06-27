function solve(
    f::AbstractFormulation,
    ker::Kernel,
    train,
    valid,
    test;
    epoch_max::Int=10,
    checkpoint_every::Int=1,
    p_update::Real=0.9,
    loss_every::Int=100,
    dir::AbstractString=pwd(),
    ε::Real=1e-4,
    verbose::Bool=true,
    save_checkpoints::Bool=true
)

    reset_timer!(TO)

    # Initialization
    @timeit TO "kernel matrix" begin
        @timeit TO "train" begin
            kernel, K = KernelMatrix(ker, f, train...)
        end
        @timeit TO "valid" begin
            Kvalid = kernelmatrix(kernel, train[1][:, K.perm], valid[1]; obsdim=2)
        end
        @timeit TO "test" begin
            Ktest = kernelmatrix(kernel, train[1][:, K.perm], test[1]; obsdim=2)
        end
    end
    state = initialize(f, K)
    iter_max = size(train[1], 2) ÷ 2
    k = 0
    l = 0
    Δ = zero(eltype(K))

    # Progress logging
    p = Progress(; epoch_max, iter_max, verbose)

    # Initial state
    state[:save_checkpoints] = save_checkpoints
    state[:dir] = dir
    state[:p_update] = p_update
    state[:epoch] = 0
    state[:ker] = ker
    state[:train] = train
    state[:valid] = valid
    state[:Kvalid] = Kvalid
    state[:test] = test
    state[:Ktest] = Ktest
    state[:βinds] = inds_β(K)

    Lp, Ld, gap = objective(f, K, state)
    state[:loss_primal] = [Lp]
    state[:loss_dual] = [Ld]
    state[:loss_gap] = [gap]
    solution = checkpoint!(f, K, state)

    optionals = () -> (
        "L primal" => state[:loss_primal][end],
        "L dual" => state[:loss_dual][end],
        "Scaled gap" => state[:loss_gap][end] / state[:loss_gap][1],
    )

    # Training
    start!(p, optionals()...)
    counter = 0
    for epoch in 1:epoch_max
        state[:epoch] += 1
        @timeit TO "Epoch" begin
            for iter in 1:iter_max
                counter += 1
                k = if Δ == 0 || rand() > p_update
                    rand(1:K.n)
                else
                    l
                end
                @timeit TO "rule selection" begin
                    r = select_rule(f, K, state, k)
                end
                @timeit TO "state update" begin
                    update_state!(f, K, state, r)
                end
                Δ, k, l = r.Δ, r.k, r.l

                # progress bar
                progress!(p, iter, epoch, optionals()...)

                # stop condition
                @timeit TO "Loss" begin
                    if counter == loss_every
                        counter = 0
                        Lp, Ld, gap = objective(f, K, state)
                        append!(state[:loss_primal], Lp)
                        append!(state[:loss_dual], Ld)
                        append!(state[:loss_gap], gap)
                    end
                end
                terminate(state, ε) && break
            end
        end

        # checkpoint
        if mod(epoch, checkpoint_every) == 0 || epoch == epoch_max
            solution = checkpoint!(f, K, state)
        end

        # stop condition
        if terminate(state, ε)
            state[:epoch] = epoch_max
            solution = checkpoint!(f, K, state)
            break
        end
    end
    finish!(p, optionals()...)
    return solution
end

function terminate(state::Dict, ε::Real)
    g0, g = state[:loss_gap][1], state[:loss_gap][end]
    return isnan(g) || isinf(g) || g / g0 <= ε
end

function checkpoint!(f, K::KernelMatrix{T}, state) where {T}
    @timeit TO "Evaluation" begin
        @timeit TO "Train scores" begin
            y_train = state[:train][2]
            s_train = compute_scores(f, K, state)
        end
        @timeit TO "Valid scores" begin
            y_valid = state[:valid][2]
            s_valid = predict(f, state, state[:Kvalid])
        end
        @timeit TO "Test scores" begin
            y_test = state[:test][2]
            s_test = predict(f, state, state[:Ktest])
        end

        path = solution_path(state[:dir], state[:epoch])

        @timeit TO "Saving" begin
            solution = Dict(
                :model => extract_solution(f, K, state),
                :p_update => state[:p_update],
                :kernel => state[:ker],
                :epoch => state[:epoch],
                :train => Dict(:y => y_train, :s => s_train),
                :valid => Dict(:y => y_valid, :s => s_valid),
                :test => Dict(:y => y_test, :s => s_test),
                :loss_primal => state[:loss_primal],
                :loss_dual => state[:loss_dual],
                :loss_gap => state[:loss_gap],
            )
            if state[:save_checkpoints]
                mkpath(dirname(path))
                save_checkpoint(path, solution)
            end
        end
    end
    return solution
end

function predict(::AbstractFormulation, state::Dict, K)
    αβ = copy(state[:αβ])
    αβ[state[:βinds]] .*= -1

    return vec(αβ' * K)
end
