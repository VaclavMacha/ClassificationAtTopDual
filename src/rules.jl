struct Update{R<:RuleType,T<:Real}
    num::T
    den::T
    lb::T
    ub::T
    k::Int
    l::Int
    Δ::T
    L::T
    δ::T

    function Update(R::Type{<:RuleType}; num, den, lb, ub, k, l, Δ, L, δ)
        return new{R, typeof(num)}(num, den, lb, ub, k, l, Δ, L, δ)
    end
end

# rule αα
abstract type ααRule <: RuleType end

update_βsort!(::Dict, ::Update{ααRule}) = nothing
update_αsum!(::Dict, ::Update{ααRule}) = nothing

function update_αβ!(state::Dict, r::Update{ααRule})
    state[:αβ][r.k] += r.Δ
    state[:αβ][r.l] -= r.Δ
    return
end

function update_scores!(state::Dict, r::Update{ααRule}, K::KernelMatrix)
    state[:s] .+= r.Δ .* (K[r.k, :] - K[r.l, :])
    return
end

# rule αβ
abstract type αβRule <: RuleType end

function update_βsort!(state::Dict, r::Update{αβRule})
    αβ = state[:αβ]
    βsort = state[:βsort]
    l = r.l
    Δ = r.Δ

    deleteat!(βsort, searchsortedfirst(βsort, αβ[l]; rev=true))
    insert!(βsort, searchsortedfirst(βsort, αβ[l] + Δ; rev=true), αβ[l] + Δ)
end

update_αsum!(state::Dict, r::Update{αβRule}) = state[:αsum] += r.Δ

function update_αβ!(state::Dict, r::Update{αβRule})
    state[:αβ][r.k] += r.Δ
    state[:αβ][r.l] += r.Δ
    return
end

function update_scores!(state::Dict, r::Update{αβRule}, K::KernelMatrix)
    state[:s] .+= r.Δ .* (K[r.k, :] + K[r.l, :])
    return
end

# rule ββ
abstract type ββRule <: RuleType end

function update_βsort!(state::Dict, r::Update{ββRule})
    αβ = state[:αβ]
    βsort = state[:βsort]
    k = r.k
    l = r.l
    Δ = r.Δ

    deleteat!(βsort, searchsortedfirst(βsort, αβ[k]; rev=true))
    insert!(βsort, searchsortedfirst(βsort, αβ[k] + Δ; rev=true), αβ[k] + Δ)
    deleteat!(βsort, searchsortedfirst(βsort, αβ[l]; rev=true))
    insert!(βsort, searchsortedfirst(βsort, αβ[l] - Δ; rev=true), αβ[l] - Δ)
    return
end

update_αsum!(::Dict, ::Update{ββRule}) = nothing

function update_αβ!(state::Dict, r::Update{ββRule})
    state[:αβ][r.k] += r.Δ
    state[:αβ][r.l] -= r.Δ
    return
end

function update_scores!(state::Dict, r::Update{ββRule}, K::KernelMatrix)
    state[:s] .+= r.Δ .* (K[r.k, :] - K[r.l, :])
    return
end
