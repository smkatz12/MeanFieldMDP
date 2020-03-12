@everywhere using GridInterpolations
@everywhere using POMDPModelTools
@everywhere using StaticArrays
@everywhere using SparseArrays
@everywhere using JLD2

# These are just for visualization
@everywhere using PGFPlots
@everywhere using Interact
@everywhere using Colors
@everywhere using ColorBrewer

@everywhere include("vMDP.jl")
@everywhere include("hMDP.jl")
@everywhere include("MeanFieldVI.jl")
@everywhere include("policyViz.jl")