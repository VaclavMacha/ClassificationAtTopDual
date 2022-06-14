module ClassificationAtTopDual

using BSON
using LIBSVM
using Random
using Roots
using Statistics
using TimerOutputs

import KernelFunctions

import KernelFunctions: kernelmatrix
import ProgressMeter: durationstring, speedstring
using Base.Iterators: partition

abstract type Kernel end
abstract type Surrogate end
abstract type RuleType end

abstract type AbstractFormulation end
abstract type TopPushKFamily{S<:Surrogate} <: AbstractFormulation end
abstract type PatMatFamily{S<:Surrogate} <: AbstractFormulation end

# exports
export solve, isfeasible

export KernelMatrix, Linear, Gaussian
export Hinge, Quadratic

export PatMat, PatMatNP
export TopPush, TopPushK, TopMeanK, tauFPL

# constants
const TO = TimerOutput()

# Defaults paths
function solution_path(dir::AbstractString, epoch::Int=-1)
    if epoch < 0
        joinpath(dir, "solution.bson")
    else
        joinpath(dir, "checkpoints", "checkpoint_epoch=$(epoch).bson")
    end
end

# Saving and loading of checkpoints
load_checkpoint(path) = BSON.load(path, @__MODULE__)
save_checkpoint(path, model) = BSON.bson(path, model)

# includes
include("logging.jl")
include("kernels.jl")
include("rules.jl")
include("utilities.jl")
include("toppushk_family.jl")
include("patmat_family.jl")
include("solver.jl")

end # module
