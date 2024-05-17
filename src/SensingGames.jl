# Vanderbilt Mathematical Programming 
#   and Intelligent Robotics Lab (VaMPIR)


module SensingGames

## Differentiation + function approximation
# using Symbolics, Symbolics.IfElse, SymbolicUtils
# using PATHSolver
using Zygote
using Flux
# using ForwardDiff

## Visualization
# using Makie, GLMakie
using Plots

## Array management
# using BlockArrays
using SparseArrays

## Development tools
using Infiltrator
using Debugger

abstract type Policy end
abstract type AbstractState end
abstract type Game end

include("utils.jl")
include("statespaces.jl")
include("games.jl")
include("policies.jl")
include("dynamics.jl")

include("../examples/localization.jl")

end #module