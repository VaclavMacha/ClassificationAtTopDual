struct PatMat{S<:Surrogate,T} <: PatMatFamily{S}
    τ::T
    C::T
    ϑ::T

    function PatMat(τ::Real; ϑ::Real=1, C::Real=1, S::Type{<:Surrogate}=Hinge)
        return new{S,typeof(τ)}(τ, C, ϑ)
    end
end

struct PatMatNP{S<:Surrogate,T} <: PatMatFamily{S}
    τ::T
    C::T
    ϑ::T

    function PatMatNP(τ::Real; ϑ::Real=1, C::Real=1, S::Type{<:Surrogate}=Hinge)
        return new{S,typeof(τ)}(τ, C, ϑ)
    end
end

# ------------------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------------------
function permutation(::PatMat, y)
    perm_α = findall(==(1), y)
    perm_β = 1:length(y)
    return length(perm_α), length(perm_β), vcat(perm_α, perm_β)
end

function permutation(::PatMatNP, y)
    perm_α = findall(==(1), y)
    perm_β = findall(==(0), y)
    return length(perm_α), length(perm_β), vcat(perm_α, perm_β)
end

function threshold(f::PatMat{S}, K::KernelMatrix{T}, state::Dict) where {S,T}
    s = compute_scores(f, K, state)
    τ = T(f.τ)
    ϑ = T(f.ϑ)
    foo(t) = sum(value.(S, ϑ .* (s .- t))) - K.nβ * τ

    return find_root(foo, (-typemax(T), typemax(T)))
end

function threshold(f::PatMatNP{S}, K::KernelMatrix{T}, state::Dict) where {S, T}
    s = compute_scores(f, K, state)[vec(K.y).==0]
    τ = T(f.τ)
    ϑ = T(f.ϑ)
    foo(t) = sum(value.(S, ϑ .* (s .- t))) - K.nβ * τ

    return find_root(foo, (-typemax(T), typemax(T)))
end

function objective(f::PatMatFamily{S}, K::KernelMatrix{T}, state::Dict) where {S, T}
    s = state[:s]
    αβ = state[:αβ]
    δ = state[:δ]
    τ = T(f.τ)
    C = T(f.C)
    ϑ = T(f.ϑ)

    s_pos = s[inds_α(K)]
    α = αβ[inds_α(K)]
    β = αβ[inds_β(K)]

    w_norm = s' * αβ / 2
    t = threshold(f, K, state)

    L_primal = w_norm + C * sum(value.(S, t .- s_pos))
    L_dual = -w_norm - C * sum(conjugate.(S, α ./ C)) - δ * sum(conjugate.(S, β ./ (δ * ϑ))) - δ * K.nβ * τ
    return L_primal, L_dual, L_primal - L_dual
end

function compute_scores(::PatMat, K::KernelMatrix, state::Dict)
    s = copy(state[:s])
    return .-s[inds_β(K)]
end

function compute_scores(::PatMatNP, K::KernelMatrix, state::Dict)
    s = copy(state[:s])
    s[inds_β(K)] .*= -1
    return s[invperm(K.perm)]
end

function extract_solution(::PatMatFamily, K::KernelMatrix, state::Dict)
    αβ = copy(state[:αβ])
    δ = copy(state[:δ])
    return Dict(:α => αβ[inds_α(K)], :β => αβ[inds_β(K)], :δ => δ)
end

# ------------------------------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------------------------------
function ffunc(::PatMatFamily{Hinge}, μ, α0, β0, δ0, C1, C2)
    return C2 * δ0 + C2^2 * sum(max.(β0 .- μ, 0)) - μ
end

function hfunc(f::PatMatFamily{Hinge}, μ, α0, β0, δ0, C1, C2)
    λ = ffunc(f, μ, α0, β0, δ0, C1, C2)
    return sum(min.(max.(α0 .- λ, 0), C1)) - sum(min.(max.(β0 .+ λ, 0), λ + μ))
end

function initialize(f::PatMatFamily{Hinge}, K::KernelMatrix{T}) where {T}
    TT = Float64
    α0 = rand(TT, K.nα)
    β0 = rand(TT, K.nβ)
    δ0 = maximum(β0)
    C1 = TT(f.C)
    C2 = TT(f.ϑ)

    if δ0 <= -C2 * sum(max.(β0 .+ maximum(α0), 0))
        α, β, δ = zero(α0), zero(β0), zero(δ0)
    else
        μ_ub1 = minimum(β0)
        μ_ub2 = minimum(C2 * δ0 + C2^2 * sum(β0) .- α0) / (C2^2 * length(β0) + 1)
        μ_ub = min(μ_ub1, μ_ub2) - TT(1e8)
        μ_lb = maximum(β0) + C2 * max(δ0, 0) + TT(1e8)

        μ = find_root(μ -> hfunc(f, μ, α0, β0, δ0, C1, C2), (μ_lb, μ_ub))
        λ = ffunc(f, μ, α0, β0, δ0, C1, C2)

        α = @. T(min(max(α0 - λ, 0), C1))
        β = @. T(min(max(β0 + λ, 0), λ + μ))
        δ = T((λ + μ) / C2)
    end

    # retun state
    αβ = vcat(α, β)
    return Dict{Symbol,Any}(
        :s => K * αβ,
        :αβ => αβ,
        :δ => δ,
        :βsort => sort(β, rev=true),
    )
end

function ffunc(::PatMatFamily{Quadratic}, λ, α0, β0, δ0)
    return sum(max.(α0 .- λ, 0)) - sum(max.(β0 .+ λ, 0))
end

function initialize(f::PatMatFamily{Quadratic}, K::KernelMatrix{T}) where {T}
    TT = Float64
    α0 = rand(TT, K.nα)
    β0 = rand(TT, K.nβ)
    τ = TT(f.τ)
    ϑ = TT(f.ϑ)
    δ0 = sqrt(max(sum(abs2, β0) / (4 * K.nβ * τ * ϑ^2), 0))

    if -maximum(β0) > maximum(α0)
        α, β, δ = zero(α0), zero(β0), zero(δ0)
    else
        λ = find_root(λ -> ffunc(f, λ, α0, β0, δ0), (-maximum(β0), maximum(α0)))
        α = @. T(max(α0 - λ, 0))
        β = @. T(max(β0 + λ, 0))
        δ = T(max(δ0, TT(1e-4)))
    end

    # retun state
    αβ = vcat(α, β)
    return Dict{Symbol,Any}(
        :s => K * αβ,
        :αβ => αβ,
        :δ => δ,
        :βsort => sort(β, rev=true),
    )
end

# ------------------------------------------------------------------------------------------
# Update rules
# ------------------------------------------------------------------------------------------
function Update(
    R::Type{<:RuleType},
    f::PatMatFamily{Hinge},
    K::KernelMatrix{T},
    αβ::Vector{T},
    δ_old::T;
    num::T,
    den::T,
    lb::T,
    ub::T,
    δ::T,
    kwargs...
) where {T<:Real}

    τ = T(f.τ)
    Δ = compute_Δ(; lb, ub, num, den)
    L = -den * Δ^2 / 2 - num * Δ - (δ - δ_old) * K.nβ * τ
    return Update(R; num, den, lb, ub, Δ, L, δ, kwargs...)
end

function Update(
    R::Type{<:RuleType},
    f::PatMatFamily{Quadratic},
    K::KernelMatrix{T},
    αβ::Vector{T},
    δ_old::T;
    num::T,
    den::T,
    lb::T,
    ub::T,
    δ::T,
    k::Int,
    l::Int,
    kwargs...
) where {T<:Real}

    τ = T(f.τ)
    ϑ = T(f.ϑ)

    if R <: ααRule
        num_new = num
        den_new = den
    elseif R <: αβRule
        num_new = num - (1 / δ_old - 1 / δ) * αβ[l] / (2 * ϑ^2)
        den_new = den - (1 / δ_old - 1 / δ) / (2 * ϑ^2)
    else
        num_new = num - (1 / δ_old - 1 / δ) * (αβ[k] - αβ[l]) / (2 * ϑ^2)
        den_new = den - (1 / δ_old - 1 / δ) / (ϑ^2)
    end

    Δ = compute_Δ(; lb, ub, num, den)
    L = -den_new * Δ^2 / 2 - num_new * Δ + (δ_old - δ) * K.nβ * τ
    return Update(R; num, den, lb, ub, Δ, L, δ, k, l, kwargs...)
end

function Update(
    R::Type{<:RuleType},
    f::PatMatFamily,
    K::KernelMatrix{T},
    state::Dict{Symbol, Any},
    k::Int,
    l::Int
) where {T<:Real}

    τ = T(f.τ)
    C = T(f.C)
    ϑ = T(f.ϑ)
    return Update(R, f, K, k, l, state[:s], state[:αβ], state[:δ], C, τ, ϑ, state[:βsort])
end

# Hinge loss
function Update(
    ::Type{ααRule},
    f::PatMatFamily{Hinge},
    K::KernelMatrix{T},
    k::Int,
    l::Int,
    s::Vector{T},
    αβ::Vector{T},
    δ::T,
    C::T,
    τ::T,
    ϑ::T,
    βsort::Vector{T}
) where {T<:Real}

    return Update(ααRule, f, K, αβ, δ; k, l,
        lb=max(-αβ[k], αβ[l] - C),
        ub=min(C - αβ[k], αβ[l]),
        num=s[k] - s[l],
        den=K[k, k] - 2 * K[k, l] + K[l, l],
        δ=δ
    )
end

function Update(::Type{αβRule}, f::PatMatFamily{Hinge},
    K::KernelMatrix{T},
    k::Int,
    l::Int,
    s::Vector{T},
    αβ::Vector{T},
    δ::T,
    C::T,
    τ::T,
    ϑ::T,
    βsort::Vector{T}
) where {T<:Real}

    βmax = find_βmax(βsort, αβ[l])
    lb = max(-αβ[k], -αβ[l])
    ub = C - αβ[k]
    kwargs = (; lb, ub, k, l)

    # solution 1
    num1 = s[k] + s[l] - 1 - 1 / ϑ
    den1 = K[k, k] + 2 * K[k, l] + K[l, l]
    Δ1 = compute_Δ(; lb, ub, num=num1, den=den1)
    if αβ[l] + Δ1 <= βmax
        r1 = Update(αβRule, f, K, αβ, δ; num=num1, den=den1, δ=βmax / ϑ, kwargs...)
    else
        r1 = (;)
    end

    # solution 2
    num2 = s[k] + s[l] - 1 - (1 - K.nβ * τ) / ϑ
    den2 = K[k, k] + 2 * K[k, l] + K[l, l]
    Δ2 = compute_Δ(; lb, ub, num=num2, den=den2)
    if αβ[l] + Δ2 >= βmax
        r2 = Update(αβRule, f, K, αβ, δ; num=num2, den=den2, δ=(αβ[l] + Δ2) / ϑ, kwargs...)
    else
        r2 = (;)
    end
    return select_rule(r1, r2)
end

function Update(::Type{ββRule}, f::PatMatFamily{Hinge},
    K::KernelMatrix{T},
    k::Int,
    l::Int,
    s::Vector{T},
    αβ::Vector{T},
    δ::T,
    C::T,
    τ::T,
    ϑ::T,
    βsort::Vector{T}
) where {T<:Real}

    βmax = find_βmax(βsort, αβ[l])
    lb = -αβ[k]
    ub = αβ[l]
    kwargs = (; lb, ub, k, l)

    # solution 1
    num1 = s[k] - s[l]
    den1 = K[k, k] - 2 * K[k, l] + K[l, l]
    Δ1 = compute_Δ(; lb, ub, num=num1, den=den1)
    if max(αβ[k] + Δ1, αβ[l] - Δ1) <= βmax
        r1 = Update(ββRule, f, K, αβ, δ; num=num1, den=den1, δ=βmax / ϑ, kwargs...)
    else
        r1 = (;)
    end

    # solution 2
    num2 = s[k] - s[l] + K.nβ * τ / ϑ
    den2 = K[k, k] - 2 * K[k, l] + K[l, l]
    Δ2 = compute_Δ(; lb, ub, num=num2, den=den2)
    if αβ[k] + Δ2 >= max(βmax, αβ[l] - Δ2)
        r2 = Update(ββRule, f, K, αβ, δ; num=num2, den=den2, δ=(αβ[k] + Δ2) / ϑ, kwargs...)
    else
        r2 = (;)
    end

    # solution 3
    num3 = s[k] - s[l] - K.nβ * τ / ϑ
    den3 = K[k, k] - 2 * K[k, l] + K[l, l]
    Δ3 = compute_Δ(; lb, ub, num=num3, den=den3)
    if αβ[l] - Δ3 >= max(βmax, αβ[k] + Δ3)
        r3 = Update(ββRule, f, K, αβ, δ; num=num3, den=den3, δ=(αβ[l] - Δ3) / ϑ, kwargs...)
    else
        r3 = (;)
    end
    return select_rule(select_rule(r1, r2), r3)
end

# Quadratic Hinge loss
function Update(::Type{ααRule}, f::PatMatFamily{Quadratic},
    K::KernelMatrix{T},
    k::Int,
    l::Int,
    s::Vector{T},
    αβ::Vector{T},
    δ::T,
    C::T,
    τ::T,
    ϑ::T,
    βsort::Vector{T}
) where {T<:Real}

    return Update(ααRule, f, K, αβ, δ; k, l,
        lb=-αβ[k],
        ub=αβ[l],
        num=s[k] - s[l] + (αβ[k] - αβ[l]) / (2C),
        den=K[k, k] - 2 * K[k, l] + K[l, l] + 1 / C,
        δ=δ
    )
end

function Update(::Type{αβRule}, f::PatMatFamily{Quadratic},
    K::KernelMatrix{T},
    k::Int,
    l::Int,
    s::Vector{T},
    αβ::Vector{T},
    δ::T,
    C::T,
    τ::T,
    ϑ::T,
    βsort::Vector{T}
) where {T<:Real}

    lb = max(-αβ[k], -αβ[l])
    ub = T(Inf)
    num = s[k] + s[l] - 1 + αβ[k] / (2C) - 1 / ϑ + αβ[l] / (2 * δ * ϑ^2)
    den = K[k, k] + 2 * K[k, l] + K[l, l] + 1 / (2C) + 1 / (2 * δ * ϑ^2)

    Δ = compute_Δ(; num, den, lb, ub)
    δ = sqrt(max(δ^2 + (Δ^2 + 2 * Δ * αβ[l]) / (4 * ϑ^2 * K.nβ * τ), 0))

    return Update(αβRule, f, K, αβ, δ; lb, ub, num, den, k, l, δ)
end

function Update(::Type{ββRule}, f::PatMatFamily{Quadratic},
    K::KernelMatrix{T},
    k::Int,
    l::Int,
    s::Vector{T},
    αβ::Vector{T},
    δ::T,
    C::T,
    τ::T,
    ϑ::T,
    βsort::Vector{T}
) where {T<:Real}

    lb = -αβ[k]
    ub = αβ[l]
    num = s[k] - s[l] + (αβ[k] - αβ[l]) / (2 * δ * ϑ^2)
    den = K[k, k] - 2 * K[k, l] + K[l, l] + 1 / (δ * ϑ^2)

    Δ = compute_Δ(; num, den, lb, ub)
    δ = sqrt(max(δ^2 + (Δ^2 + Δ * (αβ[k] - αβ[l])) / (2 * ϑ^2 * K.nβ * τ), 0))

    return Update(ββRule, f, K, αβ, δ; lb, ub, num, den, k, l, δ)
end
