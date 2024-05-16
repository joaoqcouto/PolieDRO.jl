module PolieDRO

using JuMP, LinearAlgebra
include("ConvexHulls.jl")

@enum LossFunctions hinge_loss logistic_loss msqe_loss

mutable struct PolieDROModel
    loss_function::LossFunctions
    model::GenericModel
    β0::Float64
    β1::Vector{Float64}
    optimized::Bool
end

#=
Build model function
- Receives a matrix of points, the loss function (within some options) and the significance level
- Calculates the convex hulls of the points and their associated probabilities
- Builds and returns a POlieDROModel struct, which wraps:
    - The loss function used
    - The JuMP model built
    - β0 and β1 coefficients
    - If it has been optimized or not
See PolieDRO paper, page 11 for formulation
=#
function build_model(X::Matrix{T}, y::Vector{T}, loss_function::LossFunctions=hinge_loss, significance_level::Float64=0.05) where T<:Float64
    N, D = size(X)
    Xhulls, not_vertices = ConvexHulls.convex_hulls(X)
    p = ConvexHulls.hulls_probabilities(Xhulls, significance_level)

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

        @variable(model, η[j=1:size(X,1)].>=0) # η associated with each vertex

        # constraints applied for each vertex in each hull
        @constraint(model, ct1[i in eachindex(Xhulls), j in Xhulls[i]],(η[j]-sum([κ[l]-λ[l] for l=1:i]))<=0)
        @constraint(model, ct2[i in eachindex(Xhulls), j in Xhulls[i]],η[j]>=1-y[j]*(β1⋅X[j,:]-β0))

    elseif (loss_function == logistic_loss)
        # classification problem: y values are all either 1 or -1
        @assert all([y[i] == 1 || y[i] == -1 for i in eachindex(y)]) "There is a value in y other than 1 or -1"

        # constraint applied for each vertex in each hull
        @constraint(model, ct[i in eachindex(Xhulls), j in Xhulls[i]], (log(1 + exp(y[j]*(β0+β1⋅X[j,:])))-sum([κ[l]-λ[l] for l=1:i]))<=0)
    elseif (loss_function == msqe_loss)
        # constraint applied for each vertex in each hull
        @constraint(model, ct[i in eachindex(Xhulls), j in Xhulls[i]], ((y[j]-(β0+β1⋅X[j,:]))^2-sum([κ[l]-λ[l] for l=1:i]))<=0)
    end

    return PolieDROModel(loss_function, model, -Inf64, [], false)
end

#=
Solve model function
- Receives the PolieDRO model struct and an optimizer
- Solves the JuMP model within the struct
- Updates the struct (adds in the coefficient values and switches 'optimized' bool to true)
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
- Receives the PolieDRO model struct (assumes it's solved) and a matrix of points to evaluate
- Uses the model coefficients to evaluate each point
- Returns the Y vector for the solved points
    !OBS: Hinge and Logistic Loss don't return -1 and 1 vectors but SVM and sigmoid values
=#
function evaluate_model(model::PolieDROModel, X::Matrix{T}) where T<:Float64
    @assert model.optimized "Model has not been optimized"

    if (model.loss_function == hinge_loss)
        # hinge loss evaluation
        return [model.β1'X[i,:] - model.β0 for i=axes(X,1)]

    elseif (model.loss_function == logistic_loss)
        # log loss evaluation
        return [(model.β0 + model.β1'X[i,:])/(1 + model.β0 + model.β1'X[i,:]) for i=axes(X,1)]

    elseif (model.loss_function == msqe_loss)
        # msqe loss evaluation
        return [model.β0 + model.β1'X[i,:] for i=axes(X,1)]

    end
end

end # module