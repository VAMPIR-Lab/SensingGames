# Vanderbilt Mathematical Programming 
#   and Intelligent Robotics Lab (VaMPIR)


module SensingGames

## Differentiation + function approximation
# using Symbolics, Symbolics.IfElse, SymbolicUtils
# using PATHSolver
using Zygote
using Flux
using ChainRulesCore
# using ForwardDiff
using JLD2

## Visualization
# using Makie, GLMakie
using Plots

## Collections management
# using BlockArrays
using SparseArrays
using ThreadsX
using StaticArrays

## Development tools
using Infiltrator
using Debugger
using Wandb
using Logging

abstract type Policy end
abstract type AbstractState end
abstract type Game end

include("utils.jl")
include("statespaces.jl")
include("games.jl")
include("policies.jl")
include("dynamics.jl")
include("custom_adjoints.jl")

include("../examples/localization.jl")
include("../examples/meetup.jl")
include("../examples/fovtag_planar.jl")
include("../examples/blurtag_planar.jl")

end #module