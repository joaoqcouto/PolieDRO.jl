export build_model

"""
    build_model(X, y, loss_function, point_evaluator; significance_level=0.05)


Build model function (custom loss function version)

Calculates the convex hulls and probabilities associated with the given data and builds the PolieDRO model for the specified loss function.

# Arguments
- `X::Matrix{Float64}`: Matrix `N x D` of points in which the model is trained (`N` = number of points, `D` = dimension of points)
- `y::Vector{Float64}`: Dependent variable vector relative to the points in the matrix `X` (size `N`)
- `loss_function::Function`: A loss function to be used in the PolieDRO formulation
    - Has to be convex! (This is not checked)
    - This function defines the solver you will be able to use (for instance, if you use a nonlinear function you will need a nonlinear solver)
    - Function must have a method `f(x::Vector{T}, y::T, β0::T, β1::Vector{T})` where `T` is `Float64`
- `point_evaluator::Function`: A function to evaluate a given point x and the optimized parameters β0, β1
    - Function must have a method `f(x::Vector{T}, β0::T, β1::Vector{T})` where `T` is `Float64`
- `significance_level::Float64`: Used to define a confidence interval for the probabilities associated to the hulls
    - Default value: `0.05`

# Returns
- An unsolved PolieDROModel struct, that can be solved using the solve_model function
- An evaluator function, which takes the solved model and a matrix of points `X` and evaluates them

# Assertions
- `X` and `y` must match in sizes (`N x D` and `N`)
- `N` must be larger than `D` (no high dimensional problems)
- No `Infinite` or `NaN` values in either `X` or `y`
- No duplicate points in `X`
"""
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
    "$evaluator_docstring"
    function evaluator(model_struct::PolieDROModel, Xeval::Matrix{T}) where T<:Float64
        @assert model_struct.optimized "Model has not been optimized"
        return [point_evaluator(Xeval[i,:], model_struct.β0, model_struct.β1) for i=axes(Xeval,1)]
    end

    return PolieDROModel(model, -Inf64, [], false), evaluator
end





"""
    build_model(X, y, loss_functions, point_evaluator; significance_level=0.05)


Build model function (custom epigraph version)

Calculates the convex hulls and probabilities associated with the given data and builds the PolieDRO model for the specified loss function.
The loss function in this case is a maximum of a group of functions, modeled as an epigraph. This is used, for instance, in the hinge loss function.

# Arguments
- `X::Matrix{Float64}`: Matrix NxD of points in which the model is trained (`N` = number of points, `D` = dimension of points)
- `y::Vector{Float64}`: Dependent variable vector relative to the points in the matrix `X` (size `N`)
- `loss_function::Vector{Function}`: A list of functions to be used in the PolieDRO formulation, the loss function will be an epigraph above all those
    - They have to be convex! (This is not checked)
    - These functions defines the solver you will be able to use (for instance, if you use a nonlinear function you will need a nonlinear solver)
    - Functions must have a method `f(x::Vector{T}, y::T, β0::VariableRef, β1::Vector{VariableRef})` where `T` is `Float64`
    - This method allows you to use multiple linear functions instead of a piecewise linear one and use a linear solver
- `point_evaluator::Function`: A function to evaluate a given point `x` and the optimized parameters β0, β1
    - Function must have a method `f(x::Vector{T}, β0::T, β1::Vector{T})` where `T` is `Float64`
- `significance_level::Float64`: Used to define a confidence interval for the probabilities associated to the hulls
    - Default value: `0.05`

# Returns
- An unsolved PolieDROModel struct, that can be solved using the solve_model function
- An evaluator function, which takes the solved model and a matrix of points `X` and evaluates them

# Assertions
- `X` and `y` must match in sizes (`N x D` and `N`)
- `N` must be larger than `D` (no high dimensional problems)
- No `Infinite` or `NaN` values in either `X` or `y`
- No duplicate points in `X`
"""
function build_model(X::Matrix{T}, y::Vector{T}, loss_functions::Vector{Function}, point_evaluator::Function; significance_level::Float64=0.05) where T<:Float64
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
    @constraint(model, ct_epigraph[f in loss_functions, i in eachindex(Xhulls), j in Xhulls[i]], η[j]>=f(X[j,:], y[j], β0, β1))

    # create model evaluator function
    "$evaluator_docstring"
    function evaluator(model_struct::PolieDROModel, Xeval::Matrix{T}) where T<:Float64
        @assert model_struct.optimized "Model has not been optimized"
        return [point_evaluator(Xeval[i,:], model_struct.β0, model_struct.β1) for i=axes(Xeval,1)]
    end

    return PolieDROModel(model, -Inf64, [], false), evaluator
end
