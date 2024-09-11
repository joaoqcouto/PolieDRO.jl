# PolieDRO.jl

[build-img]: https://github.com/joaoqcouto/PolieDRO.jl/actions/workflows/CI.yml/badge.svg?branch=main
[build-url]: https://github.com/joaoqcouto/PolieDRO.jl/actions/workflows/CI.yml?query=branch%3Amain

[codecov-img]: https://codecov.io/gh/joaoqcouto/PolieDRO.jl/graph/badge.svg?token=N0OWW35K7J
[codecov-url]: https://codecov.io/gh/joaoqcouto/PolieDRO.jl

| **Build Status** | **Coverage** |
|:-----------------:|:-----------------:|
| [![Build Status][build-img]][build-url] | [![codecov][codecov-img]][codecov-url]|

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
model_hl = PolieDRO.build_model(Xtrain_class, ytrain_class, PolieDRO.hinge_loss)
model_ll = PolieDRO.build_model(Xtrain_class, ytrain_class, PolieDRO.logistic_loss)

## for regression problems
model_mse = PolieDRO.build_model(Xtrain_reg, ytrain_reg, PolieDRO.mse_loss)

# solving the models
solve_model!(model_hl, HiGHS.Optimizer)
solve_model!(model_ll, Ipopt.Optimizer)
solve_model!(model_mse, Ipopt.Optimizer)

# evaluating the test sets
# classification
y_hl = evaluate_model(model_hl, Xtest_class)
y_ll = evaluate_model(model_ll, Xtest_class)

# regression
y_mse = evaluate_model(model_mse, Xtest_class)

# predictions of the models could then be compared to their expected values in ytest_class and ytest_reg
```

## Models

PolieDRO is a framework for classification and regression using distributionally robust optimization in a data-driven manner, avoiding the use of hyperparameters. From the input data, nested convex hulls are constructed and confidence intervals associated with each hull's coverage probability are calculated.

To calculate the confidence intervals for the hulls' associated probabilities, a significance level can be chosen by the user, with the default value being 5%.

![Convex hulls](https://github.com/user-attachments/assets/45cedbf1-fa43-42ec-bf4d-065e90da650b)

With the information of the vertices' convex hulls and their associated probabilities, it is then possible to construct the DRO problem developed in the framework as seen below for a given convex loss function $h(W;\beta)$:

$$\min_{\beta, \lambda,\kappa} \sum_{i \in F} (\kappa_i \overline{p_i} - \lambda_i \underline{p_i})$$

$$\text{s.t.} \quad h(W;\beta) - \sum_{l \in A(i)} (\kappa_l - \lambda_l) \leq 0, \quad \forall j \in V_i, \forall i \in F$$

$$\lambda_i \geq 0, \forall i \in F$$

$$\kappa_i \geq 0, \forall i \in F$$

$$\beta \in B$$

Where $F$ is the set of convex hulls of the observations, $V_i$ is the set of vertices present in each convex hull $i \in F$ and $\underline{p_i}$, $\overline{p_i}$ are the confidence intervals for each hull's coverage probability.

With this in hand, three loss functions used in common machine learning methods were applied to the framework.

### Hinge Loss

The hinge loss function is commonly used in classification tasks with the support vector machine (SVM). It linearly penalizes a misclassification of an observation. Below is the formulation of the DRO problem rewritten with the use of the hinge loss function as $h(W;\beta)$:

$$\min_{\beta, \lambda,\kappa, \eta} \sum_{i \in F} (\kappa_i \overline{p_i} - \lambda_i \underline{p_i})$$

$$\text{s.t.} \quad \eta_j - \sum_{l \in A(i)} (\kappa_l - \lambda_l) \leq 0, \quad \forall j \in V_i, \forall i \in F$$

$$\eta_j \geq 1 - y_j(\beta_1^Tx_j - \beta_0), \forall j \in V_i, i \in F$$

$$\eta_j \geq 0, \forall j \in V_i, i \in F$$

$$\lambda_i \geq 0, \forall i \in F$$

$$\kappa_i \geq 0, \forall i \in F$$

Having solved the problem, we have in hand the parameters $\beta$. It is then possible to evaluate a given point $x$ as below:

$$\hat{y} = \beta_1^Tx - \beta_0$$

This output is based on the hinge loss function, meaning a value below one indicates the point is classified as the class '1', while a value larger than one indicate a classification of '-1' (values between 0 and 1 are close to the boundary between classes).

#### Usage in PolieDRO

### Logistic Loss

The logistic loss is used to estimate the probability of a data point being in a certain category. In a binary setting, data points classified as '1' are expected to have a probability evaluated near 1 while data points classified as '-1' are expected to have a probability near zero. Using this loss function as $h(W;\beta)$ we arrive in the formulation below:

$$\min_{\beta, \lambda,\kappa} \sum_{i \in F} (\kappa_i \overline{p_i} - \lambda_i \underline{p_i})$$

$$\text{s.t.} \quad \log(1+e^{-y_j(\beta_0 + \beta_1^Tx_j)}) - \sum_{l \in A(i)} (\kappa_l - \lambda_l) \leq 0, \quad \forall j \in V_i, \forall i \in F$$

$$\lambda_i \geq 0, \forall i \in F$$

$$\kappa_i \geq 0, \forall i \in F$$

The parameters $\beta$ can be used to evaluate a given point $x$ as below:

$$\hat{y} = \frac{e^{\beta_0 + \beta_1^Tx}}{1+e^{\beta_0 + \beta_1^Tx}}$$

As said above, this logistic loss output is near 1 when a point is classified as '1' and near 0 when '-1'. One could then choose something such as 0.5 to decide which class to assume.

#### Usage in PolieDRO

### Mean Squared Error

The mean squared error (MSE) is commonly seen in linear regression models, such as the LASSO regression. Using it in the PolieDRO framework, the formulation we arrive at is:

$$\min_{\beta, \lambda,\kappa} \sum_{i \in F} (\kappa_i \overline{p_i} - \lambda_i \underline{p_i})$$

$$\text{s.t.} \quad (y_j - (\beta_0 + \beta_1^Tx_j))^2 - \sum_{l \in A(i)} (\kappa_l - \lambda_l) \leq 0, \quad \forall j \in V_i, \forall i \in F$$

$$\lambda_i \geq 0, \forall i \in F$$

$$\kappa_i \geq 0, \forall i \in F$$

A point $x$ can then be evaluated as below:

$$\hat{y} = \beta_0 + \beta_1^Tx$$

#### Usage in PolieDRO

## References 
-  GUTIERREZ, T.; VALLADÃO, D. ; PAGNONCELLI, B.. PolieDRO: a novel classification and regression framework with non-parametric data-driven regularization. Machine Learning, 04 2024

