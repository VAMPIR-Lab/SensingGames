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
using GLMakie
# using Plots
using Printf

## Collections management
# using BlockArrays
using SparseArrays
using ThreadsX
using StaticArrays

# Stats
using StatsBase

## Development tools
using Infiltrator
using Debugger
# using Wandb
using Logging
# using Accessors

include("utils.jl")
include("statespaces.jl")
include("games.jl")
include("game_utils.jl")
include("beliefs.jl")
include("solvers.jl")
include("policies.jl")
include("dynamics.jl")
include("render.jl")
include("custom_adjoints.jl")

include("../examples/localization.jl")
include("../examples/meetup.jl")
# include("../examples/fovtag_planar.jl")
include("../examples/blurtag.jl")
include("../examples/fovtag.jl")
include("../examples/multitag.jl")

end #module