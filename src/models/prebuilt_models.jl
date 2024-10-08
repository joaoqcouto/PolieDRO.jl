export build_model

"""
    build_model(X, y, loss_function, point_evaluator; significance_level=0.05)

Build model function (pre-implemented loss functions)

Calculates the convex hulls and probabilities associated with the given data and builds the PolieDRO model for a pre-implemented loss function.

# Arguments
- `X::Matrix{Float64}`: Matrix `N x D` of points in which the model is trained (`N` = number of points, `D` = dimension of points)
- `y::Vector{Float64}`: Dependent variable vector relative to the points in the matrix `X` (size `N`)
- `loss_function::LossFunctions`: One of the given loss functions implemented in the enumerator
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
- For classification models (hinge and logistic loss) all values in y must be either 1 or -1
"""
function build_model(X::Matrix{T}, y::Vector{T}, loss_function::LossFunctions; hulls::Union{HullsInfo,Nothing}=nothing, significance_level::Float64=0.05) where T<:Float64
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

        return build_model(X, y, [hl_1, hl_2], hl_point_evaluator; hulls=hulls, significance_level=significance_level)

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