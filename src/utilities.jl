# Surrogates
struct Hinge <: Surrogate end
struct Quadratic <: Surrogate end

value(::Type{Hinge}, s::Real) = max(0, 1 + s)
conjugate(::Type{Hinge}, s::Real) = 0 <= s <= 1 ? -s : one(s)

value(::Type{Quadratic}, s::Real) = (max(0, 1 + s))^2
conjugate(::Type{Quadratic}, s::Real) = 0 <= s ? (s^2)/4 - s : one(s)

# root finding
function find_root(f, lims)
    try
        Roots.find_zero(f, sum(lims) / 2)
    catch
        Roots.find_zero(f, lims)
    end
end


# Select rule
compute_Δ(; lb, ub, num, den, kwargs...) = min(max(lb, -num / den), ub)
select_rule(r1::NamedTuple, ::NamedTuple) = r1
select_rule(r1::Update, ::NamedTuple) = r1
select_rule(::NamedTuple, r2::Update) = r2

function select_rule(r1::Update, r2::Update)
    if isinf(r1.L) || isnan(r1.L) || r1.Δ == 0
        return r2
    end
    if isinf(r2.L) || isnan(r2.L) || r2.Δ == 0
        return r1
    end

    if r1.L == r2.L
        return rand(Bool) ? r1 : r2
    elseif r1.L > r2.L
        return r1
    else
        return r2
    end
end

function select_rule(f::AbstractFormulation, K::KernelMatrix, state::Dict, k::Int)
    best = (;)
    for l = 1:K.n
        l == k && continue
        rule = if k <= K.nα && l <= K.nα
            Update(ααRule, f, K, state, k, l)
        elseif k <= K.nα && l > K.nα
            Update(αβRule, f, K, state, k, l)
        elseif k > K.nα && l <= K.nα
            Update(αβRule, f, K, state, l, k)
        else
            Update(ββRule, f, K, state, k, l)
        end
        best = select_rule(best, rule)
    end
    return best
end

# state update
function find_βmax(βsort, βk)
    return βsort[1] != βk ? βsort[1] : βsort[2]
end

update_state!(::AbstractFormulation, K::KernelMatrix, state::Dict, r) = nothing

function update_state!(::AbstractFormulation, K::KernelMatrix, state::Dict, r::Update)
    iszero(r.Δ) && return

    haskey(state, :αsum) && update_αsum!(state, r)
    haskey(state, :βsort) && update_βsort!(state, r)
    update_scores!(state, r, K)
    update_αβ!(state, r)

    if haskey(state, :δ)
        state[:δ] = r.δ
    end
    return
end
