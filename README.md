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
  * Mean squared error: regression model based on the linear regressor

## References 
-  GUTIERREZ, T.; VALLADÃO, D. ; PAGNONCELLI, B.. PolieDRO: a novel classification and regression framework with non-parametric data-driven regularization. Machine Learning, 04 2024
