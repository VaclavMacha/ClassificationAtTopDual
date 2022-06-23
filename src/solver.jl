function solve(
    f::AbstractFormulation,
    ker::Kernel,
    train,
    valid,
    test;
    epoch_max::Int=10,
    checkpoint_every::Int=1,
    p_update::Real=0.9,
    dir::AbstractString=pwd(),
    ε::Real=1e-4,
    verbose::Bool=true,
    save_checkpoints::Bool=true
)

    reset_timer!(TO)

    # Initialization
    K = KernelMatrix(ker, f, train...)
    state = initialize(f, K)
    iter_max = size(train[1], 2) ÷ 2
    k = 0
    l = 0
    Δ = zero(Float32)

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
    state[:test] = test
    solution = checkpoint!(f, K, state)

    optionals = () -> (
        "L primal" => state[:loss_primal][end],
        "L dual" => state[:loss_dual][end],
        "Scaled gap" => state[:loss_gap][end] / state[:loss_gap][1],
    )

    # Training
    start!(p, optionals()...)
    for epoch in 1:epoch_max
        state[:epoch] += 1
        @timeit TO "Epoch" begin
            for iter in 1:iter_max
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

function checkpoint!(f, K, state)
    @timeit TO "Evaluation" begin
        @timeit TO "Train scores" begin
            y_train = state[:train][2]
            s_train = compute_scores(f, K, state)
        end
        @timeit TO "Valid scores" begin
            y_valid, s_valid = eval_model(f, state, state[:train], state[:valid])
        end
        @timeit TO "Test scores" begin
            y_test, s_test = eval_model(f, state, state[:train], state[:test])
        end

        loss_primal = get!(state, :loss_primal, Float32[])
        loss_dual = get!(state, :loss_dual, Float32[])
        loss_gap = get!(state, :loss_gap, Float32[])

        @timeit TO "Loss" begin
            Lp, Ld, gap = objective(f, K, state)
            append!(loss_primal, Lp)
            append!(loss_dual, Ld)
            append!(loss_gap, gap)
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
                :loss_primal => loss_primal,
                :loss_dual => loss_dual,
                :loss_gap => loss_gap,
            )
            if state[:save_checkpoints]
                mkpath(dirname(path))
                save_checkpoint(path, solution)
            end
        end
    end
    return solution
end

function eval_model(f::AbstractFormulation, state::Dict, train, test; chunksize=10000)
    # unpack data
    X, y = train
    Xt, yt = test

    # init kernel ans dual vars
    αβ = copy(state[:αβ])
    nα, ~, perm = permutation(f, vec(y))
    αβ[(nα+1):end] .*= -1
    kernel = initialize(state[:ker], size(X, 1))

    # compute scores
    s = zeros(Float32, size(Xt, 2))
    for cols in partition(1:size(Xt, 2), chunksize)
        K = kernelmatrix(kernel, X[:, perm], Xt[:, cols]; obsdim=2)
        s[cols] .= vec(αβ' * K)
    end
    return yt, s
end
