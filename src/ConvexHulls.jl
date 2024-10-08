# dependencies
using JuMP, HiGHS, Distributions

# calculating hulls
include("hulls/hulls-calculation.jl")

# calculating associated probabilities
include("hulls/hulls-probabilities.jl")