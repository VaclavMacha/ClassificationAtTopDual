function isfeasible(f::TopPushKFamily{Hinge}, K::KernelMatrix{T}, sol; ε::Real=1e-6) where {T}
    α, β = sol[:α], sol[:β]
    C = T(f.C)
    Kf = compute_K(f, K)

    return @testset "constraints" begin
        @test sum(α) ≈ sum(β)
        @testset "0 <= α_$(i) <= C" for i in eachindex(α)
            @test -ε <= α[i] <= C + ε
        end
        @testset "0 <= β_$(i) <= sum(α)/K" for i in eachindex(β)
            @test -ε <= β[i] <= sum(α) / Kf + ε
        end
    end
end

function isfeasible(f::TopPushKFamily{Quadratic}, K::KernelMatrix, sol; ε::Real=1e-6)
    α, β = sol[:α], sol[:β]
    Kf = compute_K(f, K)

    return @testset "constraints" begin
        @test sum(α) ≈ sum(β)
        @testset "0 <= α_$(i)" for i in eachindex(α)
            @test -ε <= α[i]
        end
        @testset "0 <= β_$(i) <= sum(α)/K" for i in eachindex(β)
            @test -ε <= β[i] <= sum(α) / Kf + ε
        end
    end
    return all([
        test_eq(sum(α), sum(β), "sum(α) == sum(β)"),
        test_lb.(α, 0, "0 <= α")...,
        test_lb.(β, 0, "0 <= β")...,
        test_ub.(β, sum(α) / Kf, "β <= sum(α)/K")...,
    ])
end

function isfeasible(f::PatMatFamily{Hinge}, ::KernelMatrix{T}, sol; ε::Real=1e-6) where {T}
    α, β, δ = sol[:α], sol[:β], sol[:δ]
    C = T(f.C)
    ϑ = T(f.ϑ)

    return @testset "constraints" begin
        @test sum(α) ≈ sum(β)
        @testset "0 <= α_$(i) <= C" for i in eachindex(α)
            @test -ε <= α[i] <= C + ε
        end
        @testset "0 <= β_$(i) <= ϑδ" for i in eachindex(β)
            @test -ε <= β[i] <= ϑ * δ + ε
        end
        @test -ε <= δ
    end
end

function isfeasible(::PatMatFamily{Quadratic}, ::KernelMatrix, sol; ε::Real=1e-6)
    α, β, δ = sol[:α], sol[:β], sol[:δ]

    return @testset "constraints" begin
        @test sum(α) ≈ sum(β)
        @testset "0 <= α_$(i)" for i in eachindex(α)
            @test -ε <= α[i]
        end
        @testset "0 <= β_$(i)" for i in eachindex(β)
            @test -ε <= β[i]
        end
        @test -ε <= δ
    end
end

# test initialization
@testset "Initialization:" for f in formulations
    @testset "$(f) with $(ker) kernel" for ker in kernels
        K = KernelMatrix(ker, f, train...)
        state = initialize(f, K)
        isfeasible(f, K, extract_solution(f, K, state); ε)
    end
end

# test rules
@testset "Update rules:" for f in formulations
    @testset "$(f) with $(ker) kernel" for ker in kernels
        K = KernelMatrix(ker, f, train...)
        iα = shuffle(inds_α(K))
        iβ = shuffle(inds_β(K))

        @testset "Rule αα:" begin
            state = initialize(f, K)
            rule = Update(ααRule, f, K, state, iα[1], iα[2])
            update_state!(f, K, state, rule)
            isfeasible(f, K, extract_solution(f, K, state); ε)
        end
        @testset "Rule αβ:" begin
            state = initialize(f, K)
            rule = Update(αβRule, f, K, state, iα[1], iβ[1])
            update_state!(f, K, state, rule)
            isfeasible(f, K, extract_solution(f, K, state); ε)
        end
        @testset "Rule ββ:" begin
            state = initialize(f, K)
            rule = Update(ββRule, f, K, state, iβ[1], iβ[2])
            update_state!(f, K, state, rule)
            isfeasible(f, K, extract_solution(f, K, state); ε)
        end
    end
end

# test final solution
@testset "Final solution:" for f in formulations
    @testset "$(f) with $(ker) kernel" for ker in kernels
        K = KernelMatrix(ker, f, train...)
        sol = solve(f, ker, train, valid, test; verbose=false, save_checkpoints=false)
        isfeasible(f, K, sol[:model]; ε)
    end
end
