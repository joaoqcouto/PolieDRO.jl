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

This output is based on the hinge loss function, meaning a value above zero indicates the point is classified as the class '1', while a value below zero indicate a classification of '-1' (values between 0 and 1 are close to the boundary between classes).

#### Usage in PolieDRO.jl

To use the Hinge Loss PolieDRO model in a classification problem, a linear solver for JuMP is needed. The example below is using [HiGHS](https://github.com/jump-dev/HiGHS.jl), an open-source linear optimization solver.

To load an example dataset, the package [UCIData.jl](https://github.com/JackDunnNZ/UCIData.jl) is used to load it directly in Julia as a Dataframe. This classification example uses [thoracic surgery data](https://archive.ics.uci.edu/dataset/277/thoracic+surgery+data).

```julia
using PolieDRO
using UCIData, HiGHS

df = UCIData.dataset("thoracic-surgery")
```

To use this dataset in PolieDRO, some basic treatment is applied. The data is normalized, category columns are encoded, missing values are removed. The dataset is also split into a training and test set for an out-of-sample evaluation.

Since the models currently only take matrices, the dataframes are also converted.

```julia
# this function can be found as written within the test directory of this repository
Xtrain, Xtest, ytrain, ytest = treat_df(df; classification=true)

Xtrain_m = Matrix{Float64}(Xtrain)
Xtest_m = Matrix{Float64}(Xtest)
```

The model is then built using the training data. It is during this time that the convex hulls are calculated for the data. The loss function is also specified as hinge loss as a parameter to build the model. A custom significance level can be chosen, here the default value of 0.05 is used.

```julia
model = PolieDRO.build_model(Xtrain_m, ytrain, PolieDRO.hinge_loss)
```

Now the model can be solved using a linear solver and the test set evaluated:

```julia
PolieDRO.solve_model!(model, HiGHS.Optimizer; silent=true)
ytest_eval = PolieDRO.evaluate_model(model, Xtest_m)
```

As said before, this outputs values relative to the hinge loss function. Below is an evaluation example where we take values above 0 as being classified as '1':

```julia
ytest_eval_abs = [yp >= 0.0 ? 1.0 : -1.0 for yp in ytest_eval]
acc_poliedro = sum(ytest_eval_abs.==ytest)*100/length(ytest)
```
For this example, the PolieDRO Hinge Loss model achieves an accuracy of 85.1%.

### Logistic Loss

The logistic loss is used to estimate the probability of a data point being in a certain category. In a binary setting, data points classified as '1' are expected to have a probability evaluated near 1 while data points classified as '-1' are expected to have a probability near zero. Using this loss function as $h(W;\beta)$ we arrive in the formulation below:

$$\min_{\beta, \lambda,\kappa} \sum_{i \in F} (\kappa_i \overline{p_i} - \lambda_i \underline{p_i})$$

$$\text{s.t.} \quad \log(1+e^{-y_j(\beta_0 + \beta_1^Tx_j)}) - \sum_{l \in A(i)} (\kappa_l - \lambda_l) \leq 0, \quad \forall j \in V_i, \forall i \in F$$

$$\lambda_i \geq 0, \forall i \in F$$

$$\kappa_i \geq 0, \forall i \in F$$

The parameters $\beta$ can be used to evaluate a given point $x$ as below:

$$\hat{y} = \frac{e^{\beta_0 + \beta_1^Tx}}{1+e^{\beta_0 + \beta_1^Tx}}$$

As said above, this logistic loss output is near 1 when a point is classified as '1' and near 0 when '-1'. One could then choose something such as 0.5 to decide which class to assume.

#### Usage in PolieDRO.jl

To use the Logistic Loss PolieDRO model in a classification problem, a nonlinear solver for JuMP is needed. The example below is using [Ipopt](https://github.com/jump-dev/Ipopt.jl), an open-source nonlinear optimization solver.

To load an example dataset, the package [UCIData.jl](https://github.com/JackDunnNZ/UCIData.jl) is used to load it directly in Julia as a Dataframe. This classification example uses [thoracic surgery data](https://archive.ics.uci.edu/dataset/277/thoracic+surgery+data).

```julia
using PolieDRO
using UCIData, Ipopt

df = UCIData.dataset("thoracic-surgery")
```

To use this dataset in PolieDRO, some basic treatment is applied. The data is normalized, category columns are encoded, missing values are removed. The dataset is also split into a training and test set for an out-of-sample evaluation.

Since the models currently only take matrices, the dataframes are also converted.

```julia
# this function can be found as written within the test directory of this repository
Xtrain, Xtest, ytrain, ytest = treat_df(df; classification=true)

Xtrain_m = Matrix{Float64}(Xtrain)
Xtest_m = Matrix{Float64}(Xtest)
```

The model is then built using the training data. It is during this time that the convex hulls are calculated for the data. The loss function is also specified as logistic loss as a parameter to build the model. A custom significance level can be chosen, here the default value of 0.05 is used.

```julia
model = PolieDRO.build_model(Xtrain_m, ytrain, PolieDRO.logistic_loss)
```

Now the model can be solved using a nonlinear solver and the test set evaluated:

```julia
PolieDRO.solve_model!(model, Ipopt.Optimizer; silent=true)
ytest_eval = PolieDRO.evaluate_model(model, Xtest_m)
```

This outputs values relative to the logistic loss function, in other words the probability of a point being in the class '1'. Below is an evaluation example where we take values above 0.5 as being classified as '1':

```julia
ytest_eval_abs = [yp >= 0.5 ? 1.0 : -1.0 for yp in ytest_eval]
acc_poliedro = sum(ytest_eval_abs.==ytest)*100/length(ytest)
```
For this example, the PolieDRO Logistic Loss model achieves an accuracy of 83.0%.

### Mean Squared Error

The mean squared error (MSE) is commonly seen in linear regression models, such as the LASSO regression. Using it in the PolieDRO framework, the formulation we arrive at is:

$$\min_{\beta, \lambda,\kappa} \sum_{i \in F} (\kappa_i \overline{p_i} - \lambda_i \underline{p_i})$$

$$\text{s.t.} \quad (y_j - (\beta_0 + \beta_1^Tx_j))^2 - \sum_{l \in A(i)} (\kappa_l - \lambda_l) \leq 0, \quad \forall j \in V_i, \forall i \in F$$

$$\lambda_i \geq 0, \forall i \in F$$

$$\kappa_i \geq 0, \forall i \in F$$

A point $x$ can then be evaluated as below:

$$\hat{y} = \beta_0 + \beta_1^Tx$$

#### Usage in PolieDRO.jl

To use the MSE PolieDRO model in a regression problem, a nonlinear solver for JuMP is needed. The example below is using [Ipopt](https://github.com/jump-dev/Ipopt.jl), an open-source nonlinear optimization solver.

To load an example dataset, the package [UCIData.jl](https://github.com/JackDunnNZ/UCIData.jl) is used to load it directly in Julia as a Dataframe. This regression example uses [automobile data](https://archive.ics.uci.edu/dataset/10/automobile).

```julia
using PolieDRO
using UCIData, Ipopt

df = UCIData.dataset("automobile")
```

To use this dataset in PolieDRO, some basic treatment is applied. The data is normalized, category columns are encoded, missing values are removed. The dataset is also split into a training and test set for an out-of-sample evaluation.

Since the models currently only take matrices, the dataframes are also converted.

```julia
# this function can be found as written within the test directory of this repository
Xtrain, Xtest, ytrain, ytest = treat_df(df; classification=false)

Xtrain_m = Matrix{Float64}(Xtrain)
Xtest_m = Matrix{Float64}(Xtest)
```

The model is then built using the training data. It is during this time that the convex hulls are calculated for the data. The loss function is also specified as MSE as a parameter to build the model. A custom significance level can be chosen, here the default value of 0.05 is used.

```julia
model = PolieDRO.build_model(Xtrain_m, ytrain, PolieDRO.mse_loss)
```

Now the model can be solved using a nonlinear solver and the test set evaluated:

```julia
PolieDRO.solve_model!(model, Ipopt.Optimizer; silent=true)
ytest_eval = PolieDRO.evaluate_model(model, Xtest_m)
```

Since this is a regression problem, these values can then be directly used as evaluations. Below we calculate the mean squared error in the test set:

```julia
mse_poliedro = mean([(ytest_eval[i] - ytest[i])^2 for i in eachindex(ytest)])
```
For this example, the PolieDRO MSE model achieves a mean squared error of 0.394.

## References 
-  GUTIERREZ, T.; VALLADÃO, D. ; PAGNONCELLI, B.. PolieDRO: a novel classification and regression framework with non-parametric data-driven regularization. Machine Learning, 04 2024

