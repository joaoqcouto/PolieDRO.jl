# Enum to store the implemented loss function values
# Each one is explained in more detail in the README.md file
@enum LossFunctions hinge_loss logistic_loss mse_loss


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


"""
PolieDRO model structure

# Fields
- `model::GenericModel`: JuMP model where the DRO problem is defined
- `β0::Float64`: The intercept term of the model solution
- `β1::Vector{Float64}`: The vector of coefficients of the model solution
- `optimized::Bool`: If the model has been solved or not
"""
mutable struct PolieDROModel
    model::GenericModel
    β0::Float64
    β1::Vector{Float64}
    optimized::Bool
end