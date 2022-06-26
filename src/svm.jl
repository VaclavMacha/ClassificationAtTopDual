struct SVM{T} <: AbstractFormulation
    C::T

    SVM(; C::Real=1) = new{typeof(C)}(C)
end

function solve(
    f::SVM,
    ker::Kernel,
    train,
    valid,
    test;
    verbose::Bool=true,
    ε::Real=1e-4,
    kwargs...
)

    reset_timer!(TO)
    io = IOBuffer()

    # Initialization
    kernel = initialize(ker, size(train[1], 1), Float64)
    @timeit TO "kernel matrix" begin
        @timeit TO "train" begin
            Ktrain = kernelmatrix(kernel, train[1]; obsdim=2)
        end
        @timeit TO "valid" begin
            Kvalid = kernelmatrix(kernel, train[1], valid[1]; obsdim=2)
        end
        @timeit TO "test" begin
            Ktest = kernelmatrix(kernel, train[1], test[1]; obsdim=2)
        end
    end
    if verbose
        print_timer(io, TO; sortby=:name)
        @info String(take!(io))
    end

    # train
    @timeit TO "training" begin
        model = LIBSVM.svmtrain(
            Ktrain,
            Int.(train[2]);
            probability=true,
            cost=Float64(f.C),
            kernel=LIBSVM.Kernel.Precomputed,
            epsilon=Float64(ε)
        )
    end
    if verbose
        print_timer(io, TO; sortby=:name)
        @info String(take!(io))
    end

    # saving solution
    @timeit TO "evaluation" begin
        solution = Dict(
            :kernel => ker,
            :train => Dict(:y => train[2], :s => predict(model, Ktrain)),
            :valid => Dict(:y => valid[2], :s => predict(model, Kvalid)),
            :test => Dict(:y => test[2], :s => predict(model, Ktest)),
        )
    end
    if verbose
        print_timer(io, TO; sortby=:name)
        @info String(take!(io))
    end
    return solution
end

function predict(model, K)
    y, s = LIBSVM.svmpredict(model, K)
    return vec(s[2, :])
end
