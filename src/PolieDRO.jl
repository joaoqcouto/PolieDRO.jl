module PolieDRO

using JuMP
include("ConvexHulls.jl")

# Enum to store the implemented loss function values
# Each one is explained in more detail in the README.md file
@enum LossFunctions hinge_loss logistic_loss mse_loss

#=
PolieDRO model structure

# Fields
- 'model::GenericModel': JuMP model where the DRO problem is defined
- 'β0::Float64': The intercept term of the model solution
- 'β1::Vector{Float64}': The vector of coefficients of the model solution
- 'optimized::Bool': If the model has been solved or not
=#
mutable struct PolieDROModel
    model::GenericModel
    β0::Float64
    β1::Vector{Float64}
    optimized::Bool
end

#=
Build model function (general loss function version)

Calculates the convex hulls and probabilities associated with the given data and builds the PolieDRO model for the specified loss function.

# Arguments
- 'X::Matrix{Float64}': Matrix NxD of points in which the model is trained (N = number of points, D = dimension of points)
- 'y::Vector{Float64}': Dependent variable vector relative to the points in the matrix X (size N)
- 'loss_function::Function': A loss function to be used in the PolieDRO formulation
    - Has to be convex! (This is not checked)
    - This function defines the solver you will be able to use (for instance, if you use a nonlinear function you will need a nonlinear solver)
    - Function must have a method f(x::Vector{T}, y::T, β0::T, β1::Vector{T}) where T is Float64
- 'point_evaluator::Function': A function to evaluate a given point x and the optimized parameters β0, β1
    - Function must have a method f(x::Vector{T}, β0::T, β1::Vector{T}) where T is Float64
- 'significance_level::Float64': Used to define a confidence interval for the probabilities associated to the hulls (read more in the README.md)
    - Default value: 0.05

# Returns
- An unsolved PolieDROModel struct, that can be solved using the solve_model function
- An evaluator function, which takes the solved model and a matrix of points X and evaluates them

# Assertions
- X and Y must match in sizes (NxD and N)
- N must be larger than D (no high dimensional problems)
- No infinite or NaN values in either X or y
- No duplicate points in X
=#
function build_model(X::Matrix{T}, y::Vector{T}, loss_function::Function, point_evaluator::Function; significance_level::Float64=0.05) where T<:Float64
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

    # add loss function constraint
    @constraint(model, ct_loss[i in eachindex(Xhulls), j in Xhulls[i]], (loss_function(X[j,:], y[j], β0, β1) - sum([κ[l]-λ[l] for l=1:i]))<=0)

    # create model evaluator function
    function evaluator(model_struct::PolieDROModel, Xeval::Matrix{T}) where T<:Float64
        @assert model_struct.optimized "Model has not been optimized"
        return [point_evaluator(Xeval[i,:], model_struct.β0, model_struct.β1) for i=axes(Xeval,1)]
    end

    return PolieDROModel(model, -Inf64, [], false), evaluator
end

#=
Build model function (general epigraph version)

Calculates the convex hulls and probabilities associated with the given data and builds the PolieDRO model for the specified loss function.
The loss function in this case is a maximum of a group of functions, modeled as an epigraph. This is used, for instance, in the hinge loss function.

# Arguments
- 'X::Matrix{Float64}': Matrix NxD of points in which the model is trained (N = number of points, D = dimension of points)
- 'y::Vector{Float64}': Dependent variable vector relative to the points in the matrix X (size N)
- 'loss_function::Vector{Function}': A list of functions to be used in the PolieDRO formulation, the loss function will be an epigraph above all those
    - They have to be convex! (This is not checked)
    - These functions defines the solver you will be able to use (for instance, if you use a nonlinear function you will need a nonlinear solver)
    - Functions must have a method f(x::Vector{T}, y::T, β0::VariableRef, β1::Vector{VariableRef}) where T is Float64
    - This method allows you to use multiple linear functions instead of a piecewise linear one and use a linear solver
- 'point_evaluator::Function': A function to evaluate a given point x and the optimized parameters β0, β1
    - Function must have a method f(x::Vector{T}, β0::T, β1::Vector{T}) where T is Float64
- 'significance_level::Float64': Used to define a confidence interval for the probabilities associated to the hulls (read more in the README.md)
    - Default value: 0.05

# Returns
- An unsolved PolieDROModel struct, that can be solved using the solve_model function
- An evaluator function, which takes the solved model and a matrix of points X and evaluates them

# Assertions
- X and Y must match in sizes (NxD and N)
- N must be larger than D (no high dimensional problems)
- No infinite or NaN values in either X or y
- No duplicate points in X
=#
function build_model(X::Matrix{T}, y::Vector{T}, loss_function::Vector{Function}, point_evaluator::Function; significance_level::Float64=0.05) where T<:Float64
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

    # add loss function epigraph constraint
    vert_idxs = Iterators.flatten(Xhulls) # every vertex index
    @variable(model, η[j in vert_idxs].>=0) # η associated with each vertex (needs to be indexed like so for constraint ct2)

    # constraint applied for each hull
    @constraint(model, ct_loss[i in eachindex(Xhulls), j in Xhulls[i]], (η[j]-sum([κ[l]-λ[l] for l=1:i]))<=0)

    # epigraph above given functions
    @constraint(model, ct_epigraph[f in loss_function, i in eachindex(Xhulls), j in Xhulls[i]], η[j]>=f(X[j,:], y[j], β0, β1))

    # create model evaluator function
    function evaluator(model_struct::PolieDROModel, Xeval::Matrix{T}) where T<:Float64
        @assert model_struct.optimized "Model has not been optimized"
        return [point_evaluator(Xeval[i,:], model_struct.β0, model_struct.β1) for i=axes(Xeval,1)]
    end

    return PolieDROModel(model, -Inf64, [], false), evaluator
end

#=
Build model function

Calculates the convex hulls and probabilities associated with the given data and builds the PolieDRO model for a pre-implemented loss function.

# Arguments
- 'X::Matrix{Float64}': Matrix NxD of points in which the model is trained (N = number of points, D = dimension of points)
- 'y::Vector{Float64}': Dependent variable vector relative to the points in the matrix X (size N)
- 'loss_function::LossFunctions': One of the given loss functions implemented in the enumerator
    - Default value: hinge_loss (for the Hinge Loss classification model)
- 'significance_level::Float64': Used to define a confidence interval for the probabilities associated to the hulls (read more in the README.md)
    - Default value: 0.05

# Returns
- An unsolved PolieDROModel struct, that can be solved using the solve_model function
- An evaluator function, which takes the solved model and a matrix of points X and evaluates them

# Assertions
- X and Y must match in sizes (NxD and N)
- N must be larger than D (no high dimensional problems)
- No infinite or NaN values in either X or y
- No duplicate points in X
- For classification models (hinge and logistic loss) all values in y must be either 1 or -1
=#
function build_model(X::Matrix{T}, y::Vector{T}; loss_function::LossFunctions=hinge_loss, significance_level::Float64=0.05) where T<:Float64
    if (loss_function == hinge_loss)
        # classification problem: y values are all either 1 or -1
        @assert all([y[i] == 1 || y[i] == -1 for i in eachindex(y)]) "There is a value in y other than 1 or -1"

        # hinge loss epigraph define as above these two
        function hl_1(x::Vector{T}, y::T, β0::VariableRef, β1::Vector{VariableRef}) where T<:Float64
            return 1-y*(sum(β1[k]*x[k] for k in eachindex(β1))-β0)
        end
        function hl_2(x::Vector{T}, y::T, β0::VariableRef, β1::Vector{VariableRef}) where T<:Float64
            return 0
        end

        # hinge loss evaluator
        function hl_point_evaluator(x::Vector{T}, β0::T, β1::Vector{T}) where T<:Float64
            return β1'x - β0
        end

        return build_model(X, y, [hl_1, hl_2], hl_point_evaluator; significance_level=significance_level)

    elseif (loss_function == logistic_loss)
        # classification problem: y values are all either 1 or -1
        @assert all([y[i] == 1 || y[i] == -1 for i in eachindex(y)]) "There is a value in y other than 1 or -1"

        # logistic loss function
        function ll_function(x::Vector{T}, y::T, β0::VariableRef, β1::Vector{VariableRef}) where T<:Float64
            return log(1 + exp(-y*(β0+sum(β1[k]*x[k] for k in eachindex(β1)))))
        end

        # logistic loss evaluator
        function ll_point_evaluator(x::Vector{T}, β0::T, β1::Vector{T}) where T<:Float64
            return exp(β0 + β1'x)/(1+exp(β0 + β1'x))
        end

        return build_model(X, y, ll_function, ll_point_evaluator; significance_level=significance_level)

    elseif (loss_function == mse_loss)
        # mse loss function
        function mse_function(x::Vector{T}, y::T, β0::VariableRef, β1::Vector{VariableRef}) where T<:Float64
            return (y-(β0+sum(β1[k]*x[k] for k in eachindex(β1))))^2
        end

        # mse loss evaluator
        function mse_point_evaluator(x::Vector{T}, β0::T, β1::Vector{T}) where T<:Float64
            return β0 + β1'x
        end

        return build_model(X, y, mse_function, mse_point_evaluator; significance_level=significance_level)
    end

    # how did you get here
    error("Loss function not implemented")
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
About the model evaluator function returned by the build_model methods

Given an optimized PolieDRO model and a matrix of points, evaluate these points in the model and returns the associated dependent variable for each one in a vector.

# Arguments
- 'model::PolieDROModel': A PolieDRO model struct, already solved
- 'X::Matrix{Float64}': A matrix of points (each point a row) to be evaluated

# Returns
- The y vector of evaluated points associated with the X matrix
    - OBS: Hinge and Logistic Loss don't return vectors with -1 and 1 values but SVM and sigmoid values

# Assertions
- Model must be optimized
=#

end # module