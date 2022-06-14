struct OnFlyMatrix{T<:Real}
    ind::Vector{Int}
    diagonal::Vector{T}
    row::Vector{T}
end

function update_row!()
