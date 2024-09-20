var documenterSearchIndex = {"docs":
[{"location":"reference/#Reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"reference/#Contents","page":"Reference","title":"Contents","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/#Index","page":"Reference","title":"Index","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [PolieDRO]","category":"page"},{"location":"reference/#PolieDRO.PolieDROModel","page":"Reference","title":"PolieDRO.PolieDROModel","text":"PolieDRO model structure\n\nFields\n\nmodel::GenericModel: JuMP model where the DRO problem is defined\nβ0::Float64: The intercept term of the model solution\nβ1::Vector{Float64}: The vector of coefficients of the model solution\noptimized::Bool: If the model has been solved or not\n\n\n\n\n\n","category":"type"},{"location":"reference/#PolieDRO.build_model-Union{Tuple{T}, Tuple{Matrix{T}, Vector{T}, Function, Function}} where T<:Float64","page":"Reference","title":"PolieDRO.build_model","text":"build_model(X, y, loss_function, point_evaluator; significance_level=0.05)\n\nBuild model function (custom loss function version)\n\nCalculates the convex hulls and probabilities associated with the given data and builds the PolieDRO model for the specified loss function.\n\nArguments\n\nX::Matrix{Float64}: Matrix NxD of points in which the model is trained (N = number of points, D = dimension of points)\ny::Vector{Float64}: Dependent variable vector relative to the points in the matrix X (size N)\nloss_function::Function: A loss function to be used in the PolieDRO formulation\nHas to be convex! (This is not checked)\nThis function defines the solver you will be able to use (for instance, if you use a nonlinear function you will need a nonlinear solver)\nFunction must have a method f(x::Vector{T}, y::T, β0::T, β1::Vector{T}) where T is Float64\npoint_evaluator::Function: A function to evaluate a given point x and the optimized parameters β0, β1\nFunction must have a method f(x::Vector{T}, β0::T, β1::Vector{T}) where T is Float64\nsignificance_level::Float64: Used to define a confidence interval for the probabilities associated to the hulls (read more in the README.md)\nDefault value: 0.05\n\nReturns\n\nAn unsolved PolieDROModel struct, that can be solved using the solve_model function\nAn evaluator function, which takes the solved model and a matrix of points X and evaluates them\n\nAssertions\n\nX and Y must match in sizes (NxD and N)\nN must be larger than D (no high dimensional problems)\nNo infinite or NaN values in either X or y\nNo duplicate points in X\n\n\n\n\n\n","category":"method"},{"location":"reference/#PolieDRO.build_model-Union{Tuple{T}, Tuple{Matrix{T}, Vector{T}, Vector{Function}, Function}} where T<:Float64","page":"Reference","title":"PolieDRO.build_model","text":"build_model(X, y, loss_functions, point_evaluator; significance_level=0.05)\n\nBuild model function (custom epigraph version)\n\nCalculates the convex hulls and probabilities associated with the given data and builds the PolieDRO model for the specified loss function. The loss function in this case is a maximum of a group of functions, modeled as an epigraph. This is used, for instance, in the hinge loss function.\n\nArguments\n\nX::Matrix{Float64}: Matrix NxD of points in which the model is trained (N = number of points, D = dimension of points)\ny::Vector{Float64}: Dependent variable vector relative to the points in the matrix X (size N)\nloss_function::Vector{Function}: A list of functions to be used in the PolieDRO formulation, the loss function will be an epigraph above all those\nThey have to be convex! (This is not checked)\nThese functions defines the solver you will be able to use (for instance, if you use a nonlinear function you will need a nonlinear solver)\nFunctions must have a method f(x::Vector{T}, y::T, β0::VariableRef, β1::Vector{VariableRef}) where T is Float64\nThis method allows you to use multiple linear functions instead of a piecewise linear one and use a linear solver\npoint_evaluator::Function: A function to evaluate a given point x and the optimized parameters β0, β1\nFunction must have a method f(x::Vector{T}, β0::T, β1::Vector{T}) where T is Float64\nsignificance_level::Float64: Used to define a confidence interval for the probabilities associated to the hulls (read more in the README.md)\nDefault value: 0.05\n\nReturns\n\nAn unsolved PolieDROModel struct, that can be solved using the solve_model function\nAn evaluator function, which takes the solved model and a matrix of points X and evaluates them\n\nAssertions\n\nX and Y must match in sizes (NxD and N)\nN must be larger than D (no high dimensional problems)\nNo infinite or NaN values in either X or y\nNo duplicate points in X\n\n\n\n\n\n","category":"method"},{"location":"reference/#PolieDRO.build_model-Union{Tuple{T}, Tuple{Matrix{T}, Vector{T}}} where T<:Float64","page":"Reference","title":"PolieDRO.build_model","text":"build_model(X, y, loss_function, point_evaluator; significance_level=0.05)\n\nBuild model function (pre-implemented loss functions)\n\nCalculates the convex hulls and probabilities associated with the given data and builds the PolieDRO model for a pre-implemented loss function.\n\nArguments\n\nX::Matrix{Float64}: Matrix NxD of points in which the model is trained (N = number of points, D = dimension of points)\ny::Vector{Float64}: Dependent variable vector relative to the points in the matrix X (size N)\nloss_function::LossFunctions: One of the given loss functions implemented in the enumerator\nDefault value: hinge_loss (for the Hinge Loss classification model)\nsignificance_level::Float64: Used to define a confidence interval for the probabilities associated to the hulls (read more in the README.md)\nDefault value: 0.05\n\nReturns\n\nAn unsolved PolieDROModel struct, that can be solved using the solve_model function\nAn evaluator function, which takes the solved model and a matrix of points X and evaluates them\n\nAssertions\n\nX and Y must match in sizes (NxD and N)\nN must be larger than D (no high dimensional problems)\nNo infinite or NaN values in either X or y\nNo duplicate points in X\nFor classification models (hinge and logistic loss) all values in y must be either 1 or -1\n\n\n\n\n\n","category":"method"},{"location":"reference/#PolieDRO.solve_model!-Tuple{PolieDRO.PolieDROModel, Any}","page":"Reference","title":"PolieDRO.solve_model!","text":"Solve model function\n\nUses the given solver to solve the PolieDRO model. Modifies the struct with the results and sets the 'optimized' bool in it to true\n\nArguments\n\nmodel::PolieDROModel: A PolieDRO model struct, as given by the build_model function, to be solved\noptimizer: An optimizer as the ones used to solve JuMP models\nNOTE: For the logistic and MSE models, a nonlinear solver is necessary\nsilent::Bool: Sets the flag to solve the model silently (without logs)\nDefault value: false\n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = PolieDRO","category":"page"},{"location":"#PolieDRO.jl","page":"Home","title":"PolieDRO.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for PolieDRO.","category":"page"}]
}
