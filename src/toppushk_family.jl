struct TopPush{S<:Surrogate,T} <: TopPushKFamily{S}
    C::T

    TopPush(; C::Real=1, S::Type{<:Surrogate}=Hinge) = new{S,typeof(C)}(C)
end

compute_K(::TopPush, ::KernelMatrix) = 1

struct TopPushK{S<:Surrogate,T} <: TopPushKFamily{S}
    K::Int
    C::T

    TopPushK(K::Int; C::Real=1, S::Type{<:Surrogate}=Hinge) = new{S,typeof(C)}(K, C)
end

compute_K(f::TopPushK, ::KernelMatrix) = f.K

struct TopMeanK{S<:Surrogate,T} <: TopPushKFamily{S}
    τ::T
    C::T

    TopMeanK(τ::Real; C::Real=1, S::Type{<:Surrogate}=Hinge) = new{S,typeof(τ)}(τ, C)
end

compute_K(f::TopMeanK, K::KernelMatrix) = max(1, round(Int, f.τ * K.nβ))

struct tauFPL{S<:Surrogate,T} <: TopPushKFamily{S}
    τ::T
    C::T

    tauFPL(τ::Real; C::Real=1, S::Type{<:Surrogate}=Hinge) = new{S,typeof(τ)}(τ, C)
end

compute_K(f::tauFPL, K::KernelMatrix) = max(1, round(Int, f.τ * K.nβ))

# ------------------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------------------
function permutation(::TopPushKFamily, y)
    perm_α = findall(==(1), y)
    perm_β = findall(==(0), y)
    return length(perm_α), length(perm_β), vcat(perm_α, perm_β)
end

function permutation(::TopMeanK, y)
    perm_α = findall(==(1), y)
    perm_β = 1:length(y)
    return length(perm_α), length(perm_β), vcat(perm_α, perm_β)
end

function threshold(::TopPush, K::KernelMatrix, state::Dict)
    s = compute_scores(f, K, state)[inds_β(K)]
    return maximum(s)
end

function threshold(f::TopPushKFamily, K::KernelMatrix, state::Dict)
    s = compute_scores(f, K, state)[inds_β(K)]
    return mean(partialsort(s, 1:compute_K(f, K); rev = true))
end

function objective(f::TopPushKFamily{S}, K::KernelMatrix, state::Dict) where {S<:Surrogate}
    s = state[:s]
    αβ = state[:αβ]
    C = Float32(f.C)

    s_pos = s[inds_α(K)]
    α = αβ[inds_α(K)]

    w_norm = s' * αβ / 2
    t = threshold(f, K, state)

    L_primal = w_norm + C * sum(value.(S, t .- s_pos))
    L_dual = -w_norm - C * sum(conjugate.(S, α ./ C))
    return L_primal, L_dual, L_primal - L_dual
end

function compute_scores(::TopPushKFamily, K::KernelMatrix, state::Dict)
    s = copy(state[:s])
    s[inds_β(K)] .*= -1
    return s[invperm(K.perm)]
end

function compute_scores(::TopMeanK, K::KernelMatrix, state::Dict)
    s = copy(state[:s])
    return .-s[inds_β(K)]
end

function extract_solution(::TopPushKFamily, K::KernelMatrix, state::Dict)
    αβ = copy(state[:αβ])
    return Dict(:α => αβ[inds_α(K)], :β => αβ[inds_β(K)])
end

# ------------------------------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------------------------------
function initialize(::TopPush, K::KernelMatrix)
    return Dict{Symbol,Any}(
        :s => zeros(Float32, K.n),
        :αβ => zeros(Float32, K.n),
    )
end

function ffunc(::TopPushKFamily, μ::Real, s::AbstractVector, K::Integer)
    i, j, d = 2, 1, 1
    λ, λ_old = z[1], 0
    g, g_old = -K * μ, -K * μ

    while g < 0
        g_old = g
        λ_old = λ

        if z[i] <= z[j] + μ
            g += d * (z[i] - λ)
            λ = z[i]
            d += 1
            i += 1
        else
            g += d * (z[j] + μ - λ)
            λ = z[j] + μ
            d -= 1
            j += 1
        end
    end
    return -(λ - λ_old) / (g - g_old) * g_old + λ_old
end

function hfunc(f::TopPushKFamily{Hinge}, μ, s, α0, β0, C, K)
    λ = ffunc(f, μ, s, K)
    return sum(min.(max.(α0 .- λ .+ sum(max.(β0 .+ λ .- μ, 0)) / K, 0), C)) - K * μ
end

function initialize(f::TopPushKFamily{Hinge}, K::KernelMatrix)
    α0 = rand(Float32, K.nα)
    β0 = rand(Float32, K.nβ)
    C = Float32(f.C)
    Kf = compute_K(f, K)

    # find feasible solution
    if mean(partialsort(β0, 1:Kf; rev=true)) + maximum(α0) <= 0
        α, β = zero(α0), zero(β0)
    else
        z = vcat(.-sort(β0; rev=true), Inf)
        μ_lb = 1e-10
        μ_ub = length(α0) * C / Kf + 1e-6

        μ = find_root(μ -> hfunc(f, μ, z, α0, β0, C, Kf), (μ_lb, μ_ub))
        λ = ffunc(f, μ, z, Kf)
        δ = sum(max.(β0 .+ λ .- μ, 0)) / Kf

        α = @. max(min(α0 - λ + δ, C), 0)
        β = @. max(min(β0 + λ, μ), 0)
    end

    # retun state
    αβ = vcat(α, β)
    return Dict{Symbol,Any}(
        :s => K * αβ,
        :αβ => αβ,
        :αsum => sum(α),
        :βsort => sort(β, rev=true),
    )
end

function hfunc(f::TopPushKFamily{Quadratic}, μ, s, α0, β0, K)
    λ = ffunc(f, μ, s, K)
    return sum(max.(α0 .- λ .+ sum(max.(β0 .+ λ .- μ, 0)) / K, 0)) - K * μ
end

function initialize(f::TopPushKFamily{Quadratic}, K::KernelMatrix)
    α0 = rand(Float32, K.nα)
    β0 = rand(Float32, K.nβ)
    Kf = compute_K(f, K)

    # find feasible solution
    if mean(partialsort(β0, 1:Kf; rev=true)) + maximum(α0) <= 0
        α, β = zero(α0), zero(β0)
    else
        s = vcat(.-sort(β0; rev=true), Inf)
        μ_lb = 1e-10
        μ_ub = length(α0) * (maximum(α0) + maximum(β0)) / Kf

        μ = find_root(μ -> hfunc(f, μ, s, α0, β0, Kf), (μ_lb, μ_ub))
        λ = ffunc(f, μ, s, Kf)
        δ = sum(max.(β0 .+ λ .- μ, 0)) / Kf

        α = @. max(α0 - λ + δ, 0)
        β = @. max(min(β0 + λ, μ), 0)
    end

    # retun state
    αβ = vcat(α, β)
    return Dict(
        :s => K * αβ,
        :αβ => αβ,
        :αsum => sum(α),
        :βsort => sort(β, rev=true),
    )
end

# ------------------------------------------------------------------------------------------
# Update rules
# ------------------------------------------------------------------------------------------
function Update(
    R::Type{<:RuleType},
    ::TopPushKFamily;
    num::Real,
    den::Real,
    lb::Real,
    ub::Real,
    kwargs...
)
    Δ = compute_Δ(; lb, ub, num, den)
    L = -den * Δ^2 / 2 - num * Δ
    return Update(R; num, den, lb, ub, Δ, L, δ=0, kwargs...)
end

# Hinge loss
function Update(::Type{ααRule}, f::TopPushKFamily{Hinge}, K::KernelMatrix, state, k, l)
    s = state[:s]
    αβ = state[:αβ]
    C = Float32(f.C)

    return Update(ααRule, f; k, l,
        lb=max(-αβ[k], αβ[l] - C),
        ub=min(C - αβ[k], αβ[l]),
        num=s[k] - s[l],
        den=K[k, k] - 2 * K[k, l] + K[l, l]
    )
end

function Update(::Type{αβRule}, f::TopPush{Hinge}, K::KernelMatrix, state, k, l)
    s = state[:s]
    αβ = state[:αβ]
    C = Float32(f.C)

    return Update(αβRule, f; k, l,
        lb=max(-αβ[k], -αβ[l]),
        ub=C - αβ[k],
        num=s[k] + s[l] - 1,
        den=K[k, k] + 2 * K[k, l] + K[l, l]
    )
end

function Update(::Type{αβRule}, f::TopPushKFamily{Hinge}, K::KernelMatrix, state, k, l)
    s = state[:s]
    αβ = state[:αβ]
    αsum = state[:αsum]
    C = Float32(f.C)

    Kf = compute_K(f, K)
    βmax = find_βmax(state, αβ[l])

    return Update(αβRule, f; k, l,
        lb=max(-αβ[k], -αβ[l], Kf * βmax - αsum),
        ub=min(C - αβ[k], (αsum - Kf * αβ[l]) / (Kf - 1)),
        num=s[k] + s[l] - 1,
        den=K[k, k] + 2 * K[k, l] + K[l, l]
    )
end

function Update(::Type{ββRule}, f::TopPush{Hinge}, K::KernelMatrix, state, k, l)
    s = state[:s]
    αβ = state[:αβ]

    return Update(ββRule, f; k, l,
        lb=-αβ[k],
        ub=αβ[l],
        num=s[k] - s[l],
        den=K[k, k] - 2 * K[k, l] + K[l, l]
    )
end

function Update(::Type{ββRule}, f::TopPushKFamily{Hinge}, K::KernelMatrix, state, k, l)
    s = state[:s]
    αβ = state[:αβ]
    αsum = state[:αsum]
    Kf = compute_K(f, K)

    return Update(ββRule, f; k, l,
        lb=max(-αβ[k], αβ[l] - αsum / Kf),
        ub=min(αsum / Kf - αβ[k], αβ[l]),
        num=s[k] - s[l],
        den=K[k, k] - 2 * K[k, l] + K[l, l]
    )
end

# Quadratic Hinge loss
function Update(::Type{ααRule}, f::TopPushKFamily{Quadratic}, K::KernelMatrix, state, k, l)
    s = state[:s]
    αβ = state[:αβ]
    C = Float32(f.C)

    return Update(ααRule, f; k, l,
        lb=-αβ[k],
        ub=αβ[l],
        num=s[k] - s[l] + (αβ[k] - αβ[l]) / (2C),
        den=K[k, k] - 2 * K[k, l] + K[l, l] + 1 / C
    )
end

function Update(::Type{αβRule}, f::TopPush{Quadratic}, K::KernelMatrix, state, k, l)
    s = state[:s]
    αβ = state[:αβ]
    C = Float32(f.C)

    return Update(αβRule, f; k, l,
        lb=max(-αβ[k], -αβ[l]),
        ub=Float32(Inf),
        num=s[k] + s[l] - 1 + αβ[k] / (2C),
        den=K[k, k] + 2 * K[k, l] + K[l, l] + 1 / (2C)
    )
end

function Update(::Type{αβRule}, f::TopPushKFamily{Quadratic}, K::KernelMatrix, state, k, l)
    s = state[:s]
    αβ = state[:αβ]
    αsum = state[:αsum]
    C = Float32(f.C)

    Kf = compute_K(f, K)
    βmax = find_βmax(state, αβ[l])

    return Update(αβRule, f; k, l,
        lb=max(-αβ[k], -αβ[l], Kf * βmax - αsum),
        ub=(αsum - Kf * αβ[l]) / (Kf - 1),
        num=s[k] + s[l] - 1 + αβ[k] / (2C),
        den=K[k, k] + 2 * K[k, l] + K[l, l] + 1 / (2C)
    )
end

function Update(::Type{ββRule}, f::TopPush{Quadratic}, K::KernelMatrix, state, k, l)
    s = state[:s]
    αβ = state[:αβ]

    return Update(ββRule, f; k, l,
        lb=-αβ[k],
        ub=αβ[l],
        num=s[k] - s[l],
        den=K[k, k] - 2 * K[k, l] + K[l, l]
    )
end

function Update(::Type{ββRule}, f::TopPushKFamily{Quadratic}, K::KernelMatrix, state, k, l)
    s = state[:s]
    αβ = state[:αβ]
    αsum = state[:αsum]
    Kf = compute_K(f, K)

    return Update(ββRule, f; k, l,
        lb=max(-αβ[k], αβ[l] - αsum / Kf),
        ub=min(αsum / Kf - αβ[k], αβ[l]),
        num=s[k] - s[l],
        den=K[k, k] - 2 * K[k, l] + K[l, l]
    )
end
