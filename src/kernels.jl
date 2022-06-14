struct Linear <: Kernel end

initialize(::Linear, ::Int) = KernelFunctions.LinearKernel(; c=zero(Float32))

struct Gaussian{T<:Real} <: Kernel
    γ::T
    scale::Bool
end

Gaussian(; γ::Real=1, scale::Bool=true) = Gaussian(γ, scale)

function initialize(ker::Gaussian, d::Int)
    T = Float32
    γ = ker.scale ? T(ker.γ / d) : T(ker.γ)
    return KernelFunctions.TransformedKernel(
        KernelFunctions.SqExponentialKernel(),
        KernelFunctions.ScaleTransform(T(sqrt(2) * sqrt(γ))),
    )
end

# KernelMatrix
struct KernelMatrix{M,T<:Real,I}
    X::Matrix{T}
    y::I

    # auxilliary vars
    n::Int
    nα::Int
    nβ::Int
    perm::Vector{Int}

    # kernel matrix
    matrix::M
end

function KernelMatrix(ker::Kernel, f::AbstractFormulation, X::Matrix{T}, y) where {T}
    nα, nβ, perm = permutation(f, vec(y))

    # kernel function
    kernel = initialize(ker, size(X, 1))

    # compute kernel matrix
    @timeit TO "KernelMatrix init" begin
        matrix = kernelmatrix(kernel, X[:, perm]; obsdim=2)
        matrix[1:nα, (nα+1):end] .*= -1
        matrix[(nα+1):end, 1:nα] .*= -1
    end

    return KernelMatrix(X, y, nα + nβ, nα, nβ, perm, matrix)
end

Base.show(io::IO, K::KernelMatrix) = print(io, "$(K.n)x$(K.n) kernel matrix")
Base.size(K::KernelMatrix) = (K.n, K.n)
Base.eltype(::KernelMatrix{M,T}) where {M,T} = T
Base.getindex(K::KernelMatrix, args...) = getindex(K.matrix, args...)
Base.:*(K::KernelMatrix, s::AbstractVector) = K.matrix * s

inds_α(K::KernelMatrix) = 1:K.nα
inds_β(K::KernelMatrix) = (K.nα+1):K.n
