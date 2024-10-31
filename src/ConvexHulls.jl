# dependencies
using JuMP, GLPK, Distributions

# calculating hulls
include("hulls/hulls_calculation.jl")

# calculating associated probabilities
include("hulls/hulls_probabilities.jl")