```@meta
CurrentModule = PolieDRO
```

# PolieDRO.jl

PolieDRO is a novel analytics framework for classification and regression that harnesses the power and flexibility of data-driven distributionally robust optimization (DRO) to circumvent the need for regularization hyperparameters.

## Features
* Build, solve and evaluate models in the PolieDRO framework using 3 implemented loss functions
  * Hinge loss: classification model based on the support vector machine (SVM)
  * Logistic loss: classification model based on the logistic regressor
  * Mean squared error: regression model based on the LASSO regressor

## Quickstart

```julia
import Pkg
Pkg.add(url = "https://github.com/joaoqcouto/PolieDRO.jl")
using PolieDRO

# Ipopt as an example of nonlinear solver
# HiGHS as example of linear solver
using Ipopt, HiGHS

## split some dataset into train and test sets
## one classification and one regression dataset as examples
## for classification problems y values are all either 1 or -1

# classification
Xtrain_class::Matrix{Float64}
ytrain_class::Vector{Float64}
Xtest_class::Matrix{Float64}
ytest_class::Vector{Float64}

# regression
Xtrain_reg::Matrix{Float64}
ytrain_reg::Vector{Float64}
Xtest_reg::Matrix{Float64}
ytest_reg::Vector{Float64}

# building the model
## classification
model_hl, evaluator_hl = PolieDRO.build_model(Xtrain_class, ytrain_class, PolieDRO.hinge_loss)
model_ll, evaluator_ll = PolieDRO.build_model(Xtrain_class, ytrain_class, PolieDRO.logistic_loss)

## regression
model_mse, evaluator_mse = PolieDRO.build_model(Xtrain_reg, ytrain_reg, PolieDRO.mse_loss)

# solving the models
solve_model!(model_hl, HiGHS.Optimizer)
solve_model!(model_ll, Ipopt.Optimizer)
solve_model!(model_mse, Ipopt.Optimizer)

# evaluating the test sets
# classification
y_hl = evaluator_hl(model_hl, Xtest_class)
y_ll = evaluator_ll(model_ll, Xtest_class)

# regression
y_mse = evaluator_mse(model_mse, Xtest_class)

# predictions of the models could then be compared to their expected values in ytest_class and ytest_reg
```

## References
-  GUTIERREZ, T.; VALLADÃO, D. ; PAGNONCELLI, B.. PolieDRO: a novel classification and regression framework with non-parametric data-driven regularization. Machine Learning, 04 2024