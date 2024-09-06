module PolieDRO

using JuMP
include("ConvexHulls.jl")

# Enum to store the implemented loss function values
# Each one is explained in more detail in the README.md file
@enum LossFunctions hinge_loss logistic_loss mse_loss

#=
PolieDRO model structure

# Fields
- 'loss_function::LossFunctions': enum within the given possible implemented loss functions
- 'model::GenericModel': JuMP model where the DRO problem is defined
- 'β0::Float64': The intercept term of the model solution
- 'β1::Vector{Float64}': The vector of coefficients of the model solution
- 'optimized::Bool': If the model has been solved or not
=#
mutable struct PolieDROModel
    loss_function::LossFunctions
    model::GenericModel
    β0::Float64
    β1::Vector{Float64}
    optimized::Bool
end

#=
Build model function

Calculates the convex hulls and probabilities associated with the given data and builds the PolieDRO model for the specified loss function.

# Arguments
- 'X::Matrix{T}': Matrix NxD of points in which the model is trained (N = number of points, D = dimension of points)
- 'y::Vector{T}': Dependent variable vector relative to the points in the matrix X (size N)
- 'loss_function::LossFunctions': One of the given loss functions implemented in the enumerator
    - Default value: hinge_loss (for the Hinge Loss classification model)
- 'significance_level::Float64': Used to define a confidence interval for the probabilities associated to the hulls (read more in the README.md)
    - Default value: 0.05

# Returns
- An unsolved PolieDROModel struct, that can be solved using the solve_model function

# Assertions
- X and Y must match in sizes (NxD and N)
- N must be larger than D (no high dimensional problems)
- No infinite or NaN values in either X or y
- No duplicate points in X
- For classification models (hinge and logistic loss) all values in y must be either 1 or -1
=#
function build_model(X::Matrix{T}, y::Vector{T}, loss_function::LossFunctions=hinge_loss, significance_level::Float64=0.05) where T<:Float64
    N, D = size(X)

    # checks on matrices to assert consistencies
    @assert length(y) == N "X matrix and y vector sizes do not match ($(N)x$(D) and $(length(y)))"
    @assert N>D "Too few points (at least $(D+1) for a dimension of $(D))"

    # no NaN or infinite values allowed
    @assert all(isfinite, X) "No infinite/NaN values allowed in X matrix"
    @assert all(isfinite, y) "No infinite/NaN values allowed in y vector"

    # no duplicate rows in X matrix
    @assert length(unique(eachrow(X))) == N "No duplicate rows allowed in X matrix"

    println("Calculating convex hulls...")
    Xhulls, not_vertices = ConvexHulls.convex_hulls(X)
    println("Calculating associated probabilities...")
    p = ConvexHulls.hulls_probabilities(Xhulls, significance_level)

    println("Building JuMP model...")

    nhulls = length(Xhulls)
    model = Model()

    # base variables
    @variable(model, κ[i=1:nhulls].>=0)
    @variable(model, λ[i=1:nhulls].>=0)
    @variable(model, β0)
    @variable(model, β1[i=1:D])

    # objective
    @objective(model, Min, sum([κ[i]*p[i][2] - λ[i]*p[i][1] for i=1:nhulls]))

    # removing non vertices from last hull (vertex constraints shouldn't apply to them)
    setdiff!(Xhulls[end], not_vertices)

    # different loss functions
    if (loss_function == hinge_loss)
        # classification problem: y values are all either 1 or -1
        @assert all([y[i] == 1 || y[i] == -1 for i in eachindex(y)]) "There is a value in y other than 1 or -1"

        vert_idxs = Iterators.flatten(Xhulls) # every vertex index
        @variable(model, η[j in vert_idxs].>=0) # η associated with each vertex (needs to be indexed like so for constraint ct2)

        # constraint applied for each hull
        @constraint(model, ct1[i in eachindex(Xhulls)],(sum(η[j] for j in Xhulls[i])-sum([κ[l]-λ[l] for l=1:i]))<=0)
        # or should be (sum(η)-sum([κ[l]-λ[l] for l=1:i]))<=0

        # constraint applied for each vertex in each hull
        @constraint(model, ct2[i in eachindex(Xhulls), j in Xhulls[i]],η[j]>=1-y[j]*(sum(β1[k]*X[j,k] for k in eachindex(β1))-β0))

    elseif (loss_function == logistic_loss)
        # classification problem: y values are all either 1 or -1
        @assert all([y[i] == 1 || y[i] == -1 for i in eachindex(y)]) "There is a value in y other than 1 or -1"

        # constraint applied for each vertex in each hull
        @constraint(model, ct[i in eachindex(Xhulls), j in Xhulls[i]], (log(1 + exp(-y[j]*(β0+sum(β1[k]*X[j,k] for k in eachindex(β1)))))-sum([κ[l]-λ[l] for l=1:i]))<=0)
    elseif (loss_function == mse_loss)
        # constraint applied for each vertex in each hull
        @constraint(model, ct[i in eachindex(Xhulls), j in Xhulls[i]], ((y[j]-(β0+sum(β1[k]*X[j,k] for k in eachindex(β1))))^2-sum([κ[l]-λ[l] for l=1:i]))<=0)
    end

    return PolieDROModel(loss_function, model, -Inf64, [], false)
end

#=
Solve model function

Uses the given solver to solve the PolieDRO model.
Modifies the struct with the results and sets the 'optimized' bool in it to true

# Arguments
- 'model::PolieDROModel': A PolieDRO model struct, as given by the build_model function, to be solved
- 'optimizer': An optimizer as the ones used to solve JuMP models
    - NOTE: For the logistic and MSE models, a nonlinear solver is necessary
- 'silent::Bool': Sets the flag to solve the model silently (without logs)
    - Default value: false
=#
function solve_model!(model::PolieDROModel, optimizer; silent::Bool=false)
    set_optimizer(model.model, optimizer)
    if (silent)
        set_silent(model.model)
    end

    optimize!(model.model)
    v = object_dictionary(model.model)

    model.β0 = value(v[:β0])
    model.β1 = value.(v[:β1])
    model.optimized = true

    return
end

#=
Evaluate model function

Given an optimized PolieDRO model and a matrix of points, evaluate these points in the model and returns the associated dependent variable for each one in a vector.

# Arguments
- 'model::PolieDROModel': A PolieDRO model struct, as given by the build_model function, to be solved
- 'optimizer': An optimizer as the ones used to solve JuMP models
    - NOTE: For the logistic and MSE models, a nonlinear solver is necessary
- 'silent::Bool': Sets the flag to solve the model silently (without logs)
    - Default value: false

# Returns
- The y vector of evaluated points associated with the X matrix
    - OBS: Hinge and Logistic Loss don't return vectors with -1 and 1 values but SVM and sigmoid values

# Assertions
- Model must be optimized
=#
function evaluate_model(model::PolieDROModel, X::Matrix{T}) where T<:Float64
    @assert model.optimized "Model has not been optimized"

    if (model.loss_function == hinge_loss)
        # hinge loss evaluation
        return [model.β1'X[i,:] - model.β0 for i=axes(X,1)]

    elseif (model.loss_function == logistic_loss)
        # log loss evaluation
        return [exp(model.β0 + model.β1'X[i,:])/(1+exp(model.β0 + model.β1'X[i,:])) for i=axes(X,1)]

    elseif (model.loss_function == mse_loss)
        # mse loss evaluation
        return [model.β0 + model.β1'X[i,:] for i=axes(X,1)]

    end
end

end # module