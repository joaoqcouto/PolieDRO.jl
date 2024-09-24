var documenterSearchIndex = {"docs":
[{"location":"models/","page":"Models","title":"Models","text":"Pages = [\"models.md\"]","category":"page"},{"location":"models/#Introduction","page":"Models","title":"Introduction","text":"","category":"section"},{"location":"models/","page":"Models","title":"Models","text":"PolieDRO is a framework for classification and regression using distributionally robust optimization in a data-driven manner, avoiding the use of hyperparameters. From the input data, nested convex hulls are constructed and confidence intervals associated with each hull's coverage probability are calculated.","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"To calculate the confidence intervals for the hulls' associated probabilities, a significance level can be chosen by the user, with the default value being 5%.","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"(Image: Convex hulls)","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"With the information of the vertices' convex hulls and their associated probabilities, it is then possible to construct the DRO problem developed in the framework as seen below for a given convex loss function h(Wbeta):","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"min_beta lambdakappa sum_i in F (kappa_i overlinep_i - lambda_i underlinep_i)","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"textst quad h(Wbeta) - sum_l in A(i) (kappa_l - lambda_l) leq 0 quad forall j in V_i forall i in F","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"lambda_i geq 0 forall i in F","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"kappa_i geq 0 forall i in F","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"beta in B","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"Where F is the set of convex hulls of the observations, V_i is the set of vertices present in each convex hull i in F and underlinep_i, overlinep_i are the confidence intervals for each hull's coverage probability.","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"With this in hand, three loss functions used in common machine learning methods were applied to the framework.","category":"page"},{"location":"models/#Hinge-Loss","page":"Models","title":"Hinge Loss","text":"","category":"section"},{"location":"models/","page":"Models","title":"Models","text":"The hinge loss function is a margin-based loss function commonly used in classification tasks with the support vector machine (SVM). It linearly penalizes a misclassification of an observation. Below is the formulation of the PolieDRO problem with the use of the hinge loss function as h(Wbeta):","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"min_beta lambdakappa eta sum_i in F (kappa_i overlinep_i - lambda_i underlinep_i)","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"textst quad eta_j - sum_l in A(i) (kappa_l - lambda_l) leq 0 quad forall j in V_i forall i in F","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"eta_j geq 1 - y_j(beta_1^Tx_j - beta_0) forall j in V_i i in F","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"eta_j geq 0 forall j in V_i i in F","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"lambda_i geq 0 forall i in F","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"kappa_i geq 0 forall i in F","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"Having solved the problem, we have in hand the parameters beta. It is then possible to evaluate a given point x as below:","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"haty = beta_1^Tx - beta_0","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"This output is based on the hinge loss function, meaning a value above zero indicates the point is classified as the class '1', while a value below zero indicate a classification of '-1' (values between 0 and 1 are close to the boundary between classes).","category":"page"},{"location":"models/#Usage-in-PolieDRO.jl","page":"Models","title":"Usage in PolieDRO.jl","text":"","category":"section"},{"location":"models/","page":"Models","title":"Models","text":"To use the Hinge Loss PolieDRO model in a classification problem, a linear solver for JuMP is needed. The example below is using HiGHS, an open-source linear optimization solver.","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"To load an example dataset, the package UCIData.jl is used to load it directly in Julia as a Dataframe. This classification example uses thoracic surgery data.","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"using PolieDRO\nusing UCIData, HiGHS\n\ndf = UCIData.dataset(\"thoracic-surgery\")","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"To use this dataset in PolieDRO, some basic treatment is applied. The data is normalized, category columns are encoded, missing values are removed. The dataset is also split into a training and test set for an out-of-sample evaluation.","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"Since the models currently only take matrices, the dataframes are also converted.","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"# this function can be found as written within the test directory of this repository\nXtrain, Xtest, ytrain, ytest = treat_df(df; classification=true)\n\nXtrain_m = Matrix{Float64}(Xtrain)\nXtest_m = Matrix{Float64}(Xtest)","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"The model is then built using the training data. It is during this time that the convex hulls are calculated for the data. The loss function is also specified as hinge loss as a parameter to build the model. A custom significance level can be chosen, here the default value of 0.05 is used.","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"Besides the model, the function also returns an evaluator function. It can be used to evaluate points with the optimized model.","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"model, evaluator = PolieDRO.build_model(Xtrain_m, ytrain, PolieDRO.hinge_loss)","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"Now the model can be solved using a linear solver and the test set evaluated with the evaluator function:","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"PolieDRO.solve_model!(model, HiGHS.Optimizer; silent=true)\nytest_eval = evaluator(model, Xtest_m)","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"As said before, this outputs values relative to the hinge loss function. Below is an evaluation example where we take values above 0 as being classified as '1':","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"ytest_eval_abs = [yp >= 0.0 ? 1.0 : -1.0 for yp in ytest_eval]\nacc_poliedro = sum(ytest_eval_abs.==ytest)*100/length(ytest)","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"For this example, the PolieDRO Hinge Loss model achieves an accuracy of 85.1%.","category":"page"},{"location":"models/#Logistic-Loss","page":"Models","title":"Logistic Loss","text":"","category":"section"},{"location":"models/","page":"Models","title":"Models","text":"The logistic loss is used to estimate the probability of a data point being in a certain category. In a binary setting, data points classified as '1' are expected to have a probability evaluated near 1 while data points classified as '-1' are expected to have a probability near zero. Using this loss function as h(Wbeta) we arrive in the formulation below:","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"min_beta lambdakappa sum_i in F (kappa_i overlinep_i - lambda_i underlinep_i)","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"textst quad log(1+e^-y_j(beta_0 + beta_1^Tx_j)) - sum_l in A(i) (kappa_l - lambda_l) leq 0 quad forall j in V_i forall i in F","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"lambda_i geq 0 forall i in F","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"kappa_i geq 0 forall i in F","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"The parameters beta can be used to evaluate a given point x as below:","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"haty = frace^beta_0 + beta_1^Tx1+e^beta_0 + beta_1^Tx","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"As said above, this logistic loss output is near 1 when a point is classified as '1' and near 0 when '-1'. One could then choose something such as 0.5 to decide which class to assume.","category":"page"},{"location":"models/#Usage-in-PolieDRO.jl-2","page":"Models","title":"Usage in PolieDRO.jl","text":"","category":"section"},{"location":"models/","page":"Models","title":"Models","text":"To use the Logistic Loss PolieDRO model in a classification problem, a nonlinear solver for JuMP is needed. The example below is using Ipopt, an open-source nonlinear optimization solver.","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"To load an example dataset, the package UCIData.jl is used to load it directly in Julia as a Dataframe. This classification example uses thoracic surgery data.","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"using PolieDRO\nusing UCIData, Ipopt\n\ndf = UCIData.dataset(\"thoracic-surgery\")","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"To use this dataset in PolieDRO, some basic treatment is applied. The data is normalized, category columns are encoded, missing values are removed. The dataset is also split into a training and test set for an out-of-sample evaluation.","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"Since the models currently only take matrices, the dataframes are also converted.","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"# this function can be found as written within the test directory of this repository\nXtrain, Xtest, ytrain, ytest = treat_df(df; classification=true)\n\nXtrain_m = Matrix{Float64}(Xtrain)\nXtest_m = Matrix{Float64}(Xtest)","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"The model is then built using the training data. It is during this time that the convex hulls are calculated for the data. The loss function is also specified as logistic loss as a parameter to build the model. A custom significance level can be chosen, here the default value of 0.05 is used.","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"model, evaluator = PolieDRO.build_model(Xtrain_m, ytrain, PolieDRO.logistic_loss)","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"Now the model can be solved using a nonlinear solver and the test set evaluated:","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"PolieDRO.solve_model!(model, Ipopt.Optimizer; silent=true)\nytest_eval = evaluator(model, Xtest_m)","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"This outputs values relative to the logistic loss function, in other words the probability of a point being in the class '1'. Below is an evaluation example where we take values above 0.5 as being classified as '1':","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"ytest_eval_abs = [yp >= 0.5 ? 1.0 : -1.0 for yp in ytest_eval]\nacc_poliedro = sum(ytest_eval_abs.==ytest)*100/length(ytest)","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"For this example, the PolieDRO Logistic Loss model achieves an accuracy of 83.0%.","category":"page"},{"location":"models/#Mean-Squared-Error","page":"Models","title":"Mean Squared Error","text":"","category":"section"},{"location":"models/","page":"Models","title":"Models","text":"The mean squared error (MSE) is a distance-based error metric commonly seen in linear regression models, such as the LASSO regression. Using it in the PolieDRO framework, the formulation we arrive at is:","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"min_beta lambdakappa sum_i in F (kappa_i overlinep_i - lambda_i underlinep_i)","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"textst quad (y_j - (beta_0 + beta_1^Tx_j))^2 - sum_l in A(i) (kappa_l - lambda_l) leq 0 quad forall j in V_i forall i in F","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"lambda_i geq 0 forall i in F","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"kappa_i geq 0 forall i in F","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"A point x can then be evaluated as below:","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"haty = beta_0 + beta_1^Tx","category":"page"},{"location":"models/#Usage-in-PolieDRO.jl-3","page":"Models","title":"Usage in PolieDRO.jl","text":"","category":"section"},{"location":"models/","page":"Models","title":"Models","text":"To use the MSE PolieDRO model in a regression problem, a nonlinear solver for JuMP is needed. The example below is using Ipopt, an open-source nonlinear optimization solver.","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"To load an example dataset, the package UCIData.jl is used to load it directly in Julia as a Dataframe. This regression example uses automobile data.","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"using PolieDRO\nusing UCIData, Ipopt\n\ndf = UCIData.dataset(\"automobile\")","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"To use this dataset in PolieDRO, some basic treatment is applied. The data is normalized, category columns are encoded, missing values are removed. The dataset is also split into a training and test set for an out-of-sample evaluation.","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"Since the models currently only take matrices, the dataframes are also converted.","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"# this function can be found as written within the test directory of this repository\nXtrain, Xtest, ytrain, ytest = treat_df(df; classification=false)\n\nXtrain_m = Matrix{Float64}(Xtrain)\nXtest_m = Matrix{Float64}(Xtest)","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"The model is then built using the training data. It is during this time that the convex hulls are calculated for the data. The loss function is also specified as MSE as a parameter to build the model. A custom significance level can be chosen, here the default value of 0.05 is used.","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"model, evaluator = PolieDRO.build_model(Xtrain_m, ytrain, PolieDRO.mse_loss)","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"Now the model can be solved using a nonlinear solver and the test set evaluated:","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"PolieDRO.solve_model!(model, Ipopt.Optimizer; silent=true)\nytest_eval = evaluator(model, Xtest_m)","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"Since this is a regression problem, these values can then be directly used as evaluations. Below we calculate the mean squared error in the test set:","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"mse_poliedro = mean([(ytest_eval[i] - ytest[i])^2 for i in eachindex(ytest)])","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"For this example, the PolieDRO MSE model achieves a mean squared error of 0.394.","category":"page"},{"location":"models/#Implementing-a-custom-model","page":"Models","title":"Implementing a custom model","text":"","category":"section"},{"location":"models/","page":"Models","title":"Models","text":"To create a custom model, two functions need to be defined:","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"A convex loss function, which will be used in the general model formulation as h(Wbeta)\nIf the loss function is piecewise linear, it is also possible to use an array of multiple functions, in a way that the loss function will be defined as the maximum of all the given functions. This will avoid making the model nonlinear.\nA point evaluator function, which will take a given point x and the model's parameters beta and evaluate haty.","category":"page"},{"location":"models/#Example:-MAE-model","page":"Models","title":"Example: MAE model","text":"","category":"section"},{"location":"models/","page":"Models","title":"Models","text":"To exemplify the construction of a custom model, we will use the mean absolute error as a distance-based error metric to construct a regression model.","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"To use the MAE function in a linear model, it is possible to define two functions instead of one and pass them as an array to the model builder.","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"This makes use of the fact that absolute error could be seen as the maximum of positive and negative error, which are both linear. The use of two linear functions instead of a piecewise linear one allows for the use of linear solvers.","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"# Positive error\nfunction pos_error(x::Vector{T}, y::T, β0::VariableRef, β1::Vector{VariableRef}) where T<:Float64\n    return (y-(β0+sum(β1[k]*x[k] for k in eachindex(β1))))\nend\n# Negative error\nfunction neg_error(x::Vector{T}, y::T, β0::VariableRef, β1::Vector{VariableRef}) where T<:Float64\n    return -(y-(β0+sum(β1[k]*x[k] for k in eachindex(β1))))\nend","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"Besides that, we need an evaluator function that will use the optimized parameters to evaluate a point x.","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"function mae_point_evaluator(x::Vector{T}, β0::T, β1::Vector{T}) where T<:Float64\n    return β0 + β1'x\nend","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"The model is then built using the training data. Instead of using a predefined model, we pass the loss functions and the evaluator to the model builder function.","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"model, evaluator = PolieDRO.build_model(Xtrain_m, ytrain, [pos_error, neg_error], mae_point_evaluator)","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"Now the model can be solved using a linear solver and the test set evaluated:","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"PolieDRO.solve_model!(model, HiGHS.Optimizer; silent=true)\nytest_eval = evaluator(model, Xtest_m)","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"Since this is a regression problem, these values can then be directly used as evaluations. Below we calculate the mean squared error in the test set:","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"mse_poliedro = mean([(ytest_eval[i] - ytest[i])^2 for i in eachindex(ytest)])","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"In the automobile dataset, the custom PolieDRO MAE model achieves a mean squared error of 0.220.","category":"page"},{"location":"reference/#Reference","page":"API reference","title":"Reference","text":"","category":"section"},{"location":"reference/","page":"API reference","title":"API reference","text":"Full description of the functions implemented in PolieDRO.jl.","category":"page"},{"location":"reference/#Contents","page":"API reference","title":"Contents","text":"","category":"section"},{"location":"reference/","page":"API reference","title":"API reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/#Index","page":"API reference","title":"Index","text":"","category":"section"},{"location":"reference/","page":"API reference","title":"API reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"API reference","title":"API reference","text":"Modules = [PolieDRO]","category":"page"},{"location":"reference/#PolieDRO.PolieDROModel","page":"API reference","title":"PolieDRO.PolieDROModel","text":"PolieDRO model structure\n\nFields\n\nmodel::GenericModel: JuMP model where the DRO problem is defined\nβ0::Float64: The intercept term of the model solution\nβ1::Vector{Float64}: The vector of coefficients of the model solution\noptimized::Bool: If the model has been solved or not\n\n\n\n\n\n","category":"type"},{"location":"reference/#PolieDRO.build_model-Union{Tuple{T}, Tuple{Matrix{T}, Vector{T}, Function, Function}} where T<:Float64","page":"API reference","title":"PolieDRO.build_model","text":"build_model(X, y, loss_function, point_evaluator; significance_level=0.05)\n\nBuild model function (custom loss function version)\n\nCalculates the convex hulls and probabilities associated with the given data and builds the PolieDRO model for the specified loss function.\n\nArguments\n\nX::Matrix{Float64}: Matrix N x D of points in which the model is trained (N = number of points, D = dimension of points)\ny::Vector{Float64}: Dependent variable vector relative to the points in the matrix X (size N)\nloss_function::Function: A loss function to be used in the PolieDRO formulation\nHas to be convex! (This is not checked)\nThis function defines the solver you will be able to use (for instance, if you use a nonlinear function you will need a nonlinear solver)\nFunction must have a method f(x::Vector{T}, y::T, β0::T, β1::Vector{T}) where T is Float64\npoint_evaluator::Function: A function to evaluate a given point x and the optimized parameters β0, β1\nFunction must have a method f(x::Vector{T}, β0::T, β1::Vector{T}) where T is Float64\nsignificance_level::Float64: Used to define a confidence interval for the probabilities associated to the hulls\nDefault value: 0.05\n\nReturns\n\nAn unsolved PolieDROModel struct, that can be solved using the solve_model function\nAn evaluator function, which takes the solved model and a matrix of points X and evaluates them\n\nAssertions\n\nX and y must match in sizes (N x D and N)\nN must be larger than D (no high dimensional problems)\nNo Infinite or NaN values in either X or y\nNo duplicate points in X\n\n\n\n\n\n","category":"method"},{"location":"reference/#PolieDRO.build_model-Union{Tuple{T}, Tuple{Matrix{T}, Vector{T}, Vector{Function}, Function}} where T<:Float64","page":"API reference","title":"PolieDRO.build_model","text":"build_model(X, y, loss_functions, point_evaluator; significance_level=0.05)\n\nBuild model function (custom epigraph version)\n\nCalculates the convex hulls and probabilities associated with the given data and builds the PolieDRO model for the specified loss function. The loss function in this case is a maximum of a group of functions, modeled as an epigraph. This is used, for instance, in the hinge loss function.\n\nArguments\n\nX::Matrix{Float64}: Matrix NxD of points in which the model is trained (N = number of points, D = dimension of points)\ny::Vector{Float64}: Dependent variable vector relative to the points in the matrix X (size N)\nloss_function::Vector{Function}: A list of functions to be used in the PolieDRO formulation, the loss function will be an epigraph above all those\nThey have to be convex! (This is not checked)\nThese functions defines the solver you will be able to use (for instance, if you use a nonlinear function you will need a nonlinear solver)\nFunctions must have a method f(x::Vector{T}, y::T, β0::VariableRef, β1::Vector{VariableRef}) where T is Float64\nThis method allows you to use multiple linear functions instead of a piecewise linear one and use a linear solver\npoint_evaluator::Function: A function to evaluate a given point x and the optimized parameters β0, β1\nFunction must have a method f(x::Vector{T}, β0::T, β1::Vector{T}) where T is Float64\nsignificance_level::Float64: Used to define a confidence interval for the probabilities associated to the hulls\nDefault value: 0.05\n\nReturns\n\nAn unsolved PolieDROModel struct, that can be solved using the solve_model function\nAn evaluator function, which takes the solved model and a matrix of points X and evaluates them\n\nAssertions\n\nX and y must match in sizes (N x D and N)\nN must be larger than D (no high dimensional problems)\nNo Infinite or NaN values in either X or y\nNo duplicate points in X\n\n\n\n\n\n","category":"method"},{"location":"reference/#PolieDRO.build_model-Union{Tuple{T}, Tuple{Matrix{T}, Vector{T}}} where T<:Float64","page":"API reference","title":"PolieDRO.build_model","text":"build_model(X, y, loss_function, point_evaluator; significance_level=0.05)\n\nBuild model function (pre-implemented loss functions)\n\nCalculates the convex hulls and probabilities associated with the given data and builds the PolieDRO model for a pre-implemented loss function.\n\nArguments\n\nX::Matrix{Float64}: Matrix N x D of points in which the model is trained (N = number of points, D = dimension of points)\ny::Vector{Float64}: Dependent variable vector relative to the points in the matrix X (size N)\nloss_function::LossFunctions: One of the given loss functions implemented in the enumerator\nDefault value: hinge_loss (for the Hinge Loss classification model)\nsignificance_level::Float64: Used to define a confidence interval for the probabilities associated to the hulls\nDefault value: 0.05\n\nReturns\n\nAn unsolved PolieDROModel struct, that can be solved using the solve_model function\nAn evaluator function, which takes the solved model and a matrix of points X and evaluates them\n\nAssertions\n\nX and y must match in sizes (N x D and N)\nN must be larger than D (no high dimensional problems)\nNo Infinite or NaN values in either X or y\nNo duplicate points in X\nFor classification models (hinge and logistic loss) all values in y must be either 1 or -1\n\n\n\n\n\n","category":"method"},{"location":"reference/#PolieDRO.solve_model!-Tuple{PolieDRO.PolieDROModel, Any}","page":"API reference","title":"PolieDRO.solve_model!","text":"Solve model function\n\nUses the given solver to solve the PolieDRO model. Modifies the struct with the results and sets the optimized bool in it to true\n\nArguments\n\nmodel::PolieDROModel: A PolieDRO model struct, as given by the build_model function, to be solved\noptimizer: An optimizer as the ones used to solve JuMP models\nNOTE: For the logistic and MSE models, a nonlinear solver is necessary\nsilent::Bool: Sets the flag to solve the model silently (without logs)\nDefault value: false\nattributes::Union{Nothing, Dict{String}}: Sets optimizer attribute flags given a dictionary with entries attribute => value\nDefault value: nothing\n\n\n\n\n\n","category":"method"},{"location":"","page":"Introduction","title":"Introduction","text":"CurrentModule = PolieDRO","category":"page"},{"location":"#PolieDRO.jl","page":"Introduction","title":"PolieDRO.jl","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"PolieDRO is a novel analytics framework for classification and regression that harnesses the power and flexibility of data-driven distributionally robust optimization (DRO) to circumvent the need for regularization hyperparameters.","category":"page"},{"location":"#Features","page":"Introduction","title":"Features","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"Build, solve and evaluate models in the PolieDRO framework using 3 implemented loss functions\nHinge loss: classification model based on the support vector machine (SVM)\nLogistic loss: classification model based on the logistic regressor\nMean squared error: regression model based on the LASSO regressor","category":"page"},{"location":"#Quickstart","page":"Introduction","title":"Quickstart","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"import Pkg\nPkg.add(url = \"https://github.com/joaoqcouto/PolieDRO.jl\")\nusing PolieDRO\n\n# Ipopt as an example of nonlinear solver\n# HiGHS as example of linear solver\nusing Ipopt, HiGHS\n\n## split some dataset into train and test sets\n## one classification and one regression dataset as examples\n## for classification problems y values are all either 1 or -1\n\n# classification\nXtrain_class::Matrix{Float64}\nytrain_class::Vector{Float64}\nXtest_class::Matrix{Float64}\nytest_class::Vector{Float64}\n\n# regression\nXtrain_reg::Matrix{Float64}\nytrain_reg::Vector{Float64}\nXtest_reg::Matrix{Float64}\nytest_reg::Vector{Float64}\n\n# building the model\n## classification\nmodel_hl, evaluator_hl = PolieDRO.build_model(Xtrain_class, ytrain_class, PolieDRO.hinge_loss)\nmodel_ll, evaluator_ll = PolieDRO.build_model(Xtrain_class, ytrain_class, PolieDRO.logistic_loss)\n\n## regression\nmodel_mse, evaluator_mse = PolieDRO.build_model(Xtrain_reg, ytrain_reg, PolieDRO.mse_loss)\n\n# solving the models\nsolve_model!(model_hl, HiGHS.Optimizer)\nsolve_model!(model_ll, Ipopt.Optimizer)\nsolve_model!(model_mse, Ipopt.Optimizer)\n\n# evaluating the test sets\n# classification\ny_hl = evaluator_hl(model_hl, Xtest_class)\ny_ll = evaluator_ll(model_ll, Xtest_class)\n\n# regression\ny_mse = evaluator_mse(model_mse, Xtest_class)\n\n# predictions of the models could then be compared to their expected values in ytest_class and ytest_reg","category":"page"},{"location":"#References","page":"Introduction","title":"References","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"GUTIERREZ, T.; VALLADÃO, D. ; PAGNONCELLI, B.. PolieDRO: a novel classification and regression framework with non-parametric data-driven regularization. Machine Learning, 04 2024","category":"page"}]
}
