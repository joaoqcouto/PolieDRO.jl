# dependencies
using JuMP, GLPK, Distributions

# calculating hulls
include("hulls/hulls-calculation.jl")

# calculating associated probabilities
include("hulls/hulls-probabilities.jl")