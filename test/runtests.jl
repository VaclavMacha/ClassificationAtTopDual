using Test
using Random
using ClassificationAtTopDual

using ClassificationAtTopDual: PatMatFamily, TopPushKFamily
using ClassificationAtTopDual: compute_K
using ClassificationAtTopDual: initialize, extract_solution, update_state!
using ClassificationAtTopDual: Update, ααRule, αβRule, ββRule, inds_α, inds_β

Random.seed!(1234)

# initialization
train = (rand(Float32, 50, 20), rand(Bool, 20))
valid = train
test = train

kernels = (
    Linear(),
    Gaussian(),
)

formulations = (
    TopPush(; S=Hinge),
    TopPush(; S=Quadratic),
    TopPushK(2; S=Hinge),
    TopPushK(2; S=Quadratic),
    TopMeanK(0.01; S=Hinge),
    TopMeanK(0.01; S=Quadratic),
    tauFPL(0.01; S=Hinge),
    tauFPL(0.01; S=Quadratic),
    PatMat(0.01; S=Hinge),
    PatMat(0.01; S=Quadratic),
    PatMatNP(0.01; S=Hinge),
    PatMatNP(0.01; S=Quadratic),
)

# tests
ε = 1e-6 # tolerance

@testset "Feasibility" begin
    include("feasibility.jl")
end
