# docstring for returned evaluator functions
evaluator_docstring = """
Model evaluator function
Returned by the build_model methods

Given an optimized PolieDRO model and a matrix of points, evaluate these points in the model and returns the associated dependent variable for each one in a vector.

# Arguments
- `model::PolieDROModel`: A PolieDRO model struct, already solved
- `X::Matrix{Float64}`: A matrix of points (each point a row) to be evaluated

# Returns
- The y vector of evaluated points associated with the X matrix
    - OBS: Hinge and Logistic Loss don't return vectors with -1 and 1 values but SVM and sigmoid values

# Assertions
- Model must be optimized
"""
