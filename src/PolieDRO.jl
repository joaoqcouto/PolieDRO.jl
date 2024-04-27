module PolieDRO

using JuMP, LinearAlgebra
include("ConvexHulls.jl")

@enum LossFunctions hinge_loss logistic_loss msqe_loss

#=
Build model function
- Receives a matrix of points, the loss function (within some options) and the significance level
- Calculates the convex hulls of the points and their associated probabilities
- Builds and returns a JuMP model ready to be solved
See PolieDRO paper, page 11 for formulation
=#
function build_model(X::Matrix{T}, y::Vector{T}, loss_function::LossFunctions=hinge_loss, significance_level::Float64=0.05) where T<:Float64
    N, D = size(X)
    Xhulls = ConvexHulls.convex_hulls(X)
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

    # different loss functions
    if (loss_function == hinge_loss)
        @variable(model, η[j=1:size(X,1)].>=0) # η associated with each vertex

        # constraints applied for each vertex in each hull
        @constraint(model, ct1[i in eachindex(Xhulls), j in Xhulls[i]],(η[j]-sum([κ[l]-λ[l] for l=1:i]))<=0)
        @constraint(model, ct2[i in eachindex(Xhulls), j in Xhulls[i]],η[j]>=1-y[j]*(β1⋅X[j,:]-β0))

    elseif (loss_function == logistic_loss)
        # constraint applied for each vertex in each hull
        @constraint(model, ct[i in eachindex(Xhulls), j in Xhulls[i]],(log(exp(-y[j]*(β0+β1⋅X[j,:])))-sum([κ[l]-λ[l] for l=1:i]))<=0)
    elseif (loss_function == msqe_loss)
        # constraint applied for each vertex in each hull
        @constraint(model, ct[i in eachindex(Xhulls), j in Xhulls[i]], ((y[j]-(β0+β1⋅X[j,:]))^2-sum([κ[l]-λ[l] for l=1:i]))<=0)
    end

    return model
end

end
