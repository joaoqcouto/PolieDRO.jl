using Test
using DataFrames
using Statistics, JuMP, UCIData
using MLJ, MLJLinearModels, MLJLIBSVMInterface
using PolieDRO

# testing utils
include("utils/dataset_aux.jl")
include("utils/hypercubes_gen.jl")

# convex hulls and probabilities tests
include("hulls/hull_tests.jl")

# including model tests
include("models/classification_tests.jl")
include("models/regression_tests.jl")
