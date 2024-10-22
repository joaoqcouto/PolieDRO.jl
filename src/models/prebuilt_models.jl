export build_model, LossFunctions

"""
Loss functions enumerator

# Values
- `hinge_loss`: Margin-based loss function for classification
- `logistic_loss`: Probability-based loss function for classification
- `mse_loss`: Distance-based loss function for regression
- `mae_loss`: Distance-based loss function for regression
"""
@enum LossFunctions hinge_loss logistic_loss mse_loss mae_loss

"""
    build_model(X, y, loss_function; hulls=nothing, significance_level=0.1, silent=true)

Build model function (pre-implemented loss functions)

Calculates the convex hulls and probabilities associated with the given data and builds the PolieDRO model for a pre-implemented loss function.

# Arguments
- `X::Matrix{Float64}`: Matrix `N x D` of points in which the model is trained (`N` = number of points, `D` = dimension of points)
- `y::Vector{Float64}`: Dependent variable vector relative to the points in the matrix `X` (size `N`)
- `loss_function::LossFunctions`: One of the given loss functions implemented in the enumerator
- `significance_level::Float64`: Used to define a confidence interval for the probabilities associated to the hulls
    - Default value: `0.1`
- `silent::Bool`: Sets the flag to build the hulls silently (without logs)
    - Default value: `true`

# Returns
- An unsolved PolieDROModel struct, that can be solved using the solve_model function
- A predictor function, which takes the solved model and a matrix of points `X` and predicts their `y`

# Assertions
- `X` and `y` must match in sizes (`N x D` and `N`)
- `N` must be larger than `D` (no high dimensional problems)
- No `Infinite` or `NaN` values in either `X` or `y`
- No duplicate points in `X`
- For classification models (hinge and logistic loss) all values in y must be either 1 or -1
"""
function build_model(X::Matrix{T}, y::Vector{T}, loss_function::LossFunctions; hulls::Union{HullsInfo,Nothing}=nothing, significance_level::Float64=0.1, silent::Bool=true) where T<:Float64
    if (loss_function == hinge_loss)
        # classification problem: y values are all either 1 or -1
        @assert all([y[i] == 1 || y[i] == -1 for i in eachindex(y)]) "There is a value in y other than 1 or -1"

        # hinge loss epigraph define as above these two
        function hl_1(x::Vector{T}, y::T, β0::VariableRef, β1::Vector{VariableRef}) where T<:Float64
            return 1 - y*(β1'x - β0)
        end
        function hl_2(x::Vector{T}, y::T, β0::VariableRef, β1::Vector{VariableRef}) where T<:Float64
            return 0
        end

        # hinge loss evaluator
        function hl_point_evaluator(x::Vector{T}, β0::T, β1::Vector{T}) where T<:Float64
            return β1'x - β0
        end

        return build_model(X, y, [hl_1, hl_2], hl_point_evaluator; hulls=hulls, significance_level=significance_level, silent=silent)

    elseif (loss_function == logistic_loss)
        # classification problem: y values are all either 1 or -1
        @assert all([y[i] == 1 || y[i] == -1 for i in eachindex(y)]) "There is a value in y other than 1 or -1"

        # logistic loss function
        function ll_function(x::Vector{T}, y::T, β0::VariableRef, β1::Vector{VariableRef}) where T<:Float64
            return log(1 + exp(-y*(β0 + β1'x)))
        end

        # logistic loss evaluator
        function ll_point_evaluator(x::Vector{T}, β0::T, β1::Vector{T}) where T<:Float64
            return exp(β0 + β1'x)/(1+exp(β0 + β1'x))
        end

        return build_model(X, y, ll_function, ll_point_evaluator; hulls=hulls, significance_level=significance_level, silent=silent)

    elseif (loss_function == mse_loss)
        # mse loss function
        function mse_function(x::Vector{T}, y::T, β0::VariableRef, β1::Vector{VariableRef}) where T<:Float64
            return (y-(β0 + β1'x))^2
        end

        # mse loss evaluator
        function mse_point_evaluator(x::Vector{T}, β0::T, β1::Vector{T}) where T<:Float64
            return β0 + β1'x
        end

        return build_model(X, y, mse_function, mse_point_evaluator; hulls=hulls, significance_level=significance_level, silent=silent)

    elseif (loss_function == mae_loss)
        # mae loss epigraph define as above these two
        function mae_1(x::Vector{T}, y::T, β0::VariableRef, β1::Vector{VariableRef}) where T<:Float64
            return y-(β0 + β1'x)
        end
        function mae_2(x::Vector{T}, y::T, β0::VariableRef, β1::Vector{VariableRef}) where T<:Float64
            return -y+(β0 + β1'x)
        end

        # mae evaluator
        function mae_point_evaluator(x::Vector{T}, β0::T, β1::Vector{T}) where T<:Float64
            return β0 + β1'x
        end

        return build_model(X, y, [mae_1, mae_2], mae_point_evaluator; hulls=hulls, significance_level=significance_level, silent=silent)

    end

    # how did you get here
    error("Loss function not implemented")
end