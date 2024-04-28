# Vanderbilt Mathematical Programming 
#   and Intelligent Robotics Lab (VaMPIR)


module SensingGames

## Differentiation
# using Symbolics, Symbolics.IfElse, SymbolicUtils
# using PATHSolver
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

# include("particles.jl")
include("games.jl")
include("policies.jl")

include("../examples/localization.jl")

end #module
