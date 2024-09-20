using Test
using DataFrames
using Statistics, JuMP, UCIData
using MLJ, Ipopt, MLJLinearModels, MLJLIBSVMInterface
include("../src/ConvexHulls.jl")
include("../src/PolieDRO.jl")
include("../test/dataset_aux.jl")

# Function to build test matrices for the convex hulls algorithm
# Returns a matrix with points organized as N D-dimensional hypercubes of increasing size
# Each hypercube is expected to be a convex hull, the largest one being the outermost hull
function hypercubes_matrix(N, D)
    points = []
    for i in 1:N
        for p in 1:2^D
            point = []
            p_div = p
            for _ in 1:D
                result =  p_div÷2
                remainder = p_div%2
                append!(point, i * (-1)^(remainder))
                p_div = result
            end
            push!(points, point)
        end
    end
    matrix = Float64.(reduce(hcat,points)')
    return matrix
end

"""
Testing convex hulls creation on hypercubes
"""
@testset "Convex hulls tests" begin

    # test for 4 convex hulls made by hypercubes of dimensions 2 up to 10
    # one test without the origin point and one test with the origin point
    N = 4
    for D = 2:10
        @testset "Hypercube $(D)D tests" begin
            # hypercube tests
            X_1 = hypercubes_matrix(N, D)
            hulls_X_1, _ = ConvexHulls.convex_hulls(X_1)
            for n = 1:N
                hull = N-n+1
                @test issetequal(hulls_X_1[hull], [i for i in (n-1)*(2^D)+1:(n)*(2^D)])
            end
    
            # including origin point
            X_2 = vcat([0 for i = 1:D]', X_1)
            hulls_X_2, _ = ConvexHulls.convex_hulls(X_2)
            for n = 1:N
                hull = N-n+1
                if hull==N
                    @test issetequal(hulls_X_2[hull], [i for i in 1:(n)*(2^D)+1])
                else
                    @test issetequal(hulls_X_2[hull], [i for i in (n-1)*(2^D)+2:(n)*(2^D)+1])
                end
            end
        end
    end
end

"""
Testing convex hulls probability intervals calculation
"""
@testset "Probabilities test" begin

    # test for 2 -> 10 4 dimensional convex hulls
    # probability centers should be equally distributed since all hulls have the same number of points
    D = 4
    for N = 2:10
        @testset "$(N) hypercubes tests" begin
            # hypercube tests
            X = hypercubes_matrix(N, D)
            hulls_X, _ = ConvexHulls.convex_hulls(X)
            probabilities_X = ConvexHulls.hulls_probabilities(hulls_X, 0.05)

            expected_probability = 1.0
            for interval in probabilities_X
                avg_prob = Statistics.mean(interval)
                @test abs(avg_prob - expected_probability) < 0.00000001 # since it is float math there is an imprecision
                expected_probability -= 1.0/N
            end
        end
    end
end

"""
Testing on verified datasets
All have ~90% PolieDRO and SVC accuracy, if they fail it means something changed for the worse
"""
@testset "Hinge Loss classification tests" begin
    # loading model to test against
    @load SVC pkg=LIBSVM

    # testing on some working classification datasets
    some_datasets = ["balloons-a", "breast-cancer-wisconsin-diagnostic", "iris"]

    for dataset in some_datasets
        println("===================")
        println("$(dataset) dataset test")
        println("Fetching dataset...")
        df = nothing
        try
            df = UCIData.dataset(dataset)
        catch
            println("Failed to fetch $(dataset), skipping")
            continue
        end
        println("Treating dataset...")
        Xtrain, Xtest, ytrain, ytest = dataset_aux.treat_df(df; classification=true)

        Xtrain_m = Matrix{Float64}(Xtrain)
        Xtest_m = Matrix{Float64}(Xtest)

        # training PolieDRO
        model, hl_evaluator = PolieDRO.build_model(Xtrain_m, ytrain; loss_function=PolieDRO.hinge_loss)
        println("Solving PolieDRO model...")
        PolieDRO.solve_model!(model, Ipopt.Optimizer; silent=true)
        println("Evaluating PolieDRO model...")
        ypoliedro = hl_evaluator(model, Xtest_m)

        # comparing to MLJ SVM
        println("Fitting SVM...")
        mach = fit!(machine(SVC(), Xtrain, categorical(ytrain)))
        println("Evaluating SVM...")
        ysvm = Vector{Float64}(predict(mach, Xtest))

        println("Calculating error metrics...")
        ypoliedro_abs = [yp >= 0 ? 1.0 : -1.0 for yp in ypoliedro]
        acc_poliedro = sum(ypoliedro_abs.==ytest)*100/length(ytest)
        acc_svm = sum(ysvm.==ytest)*100/length(ytest)

        println("Accuracy % on $(dataset) dataset")
        println("PolieDRO = $(acc_poliedro)")
        println("SVM = $(acc_svm)")
        println("===================")

        # test against svm-25% performance
        # model should not be much worse than svm
        # PolieDRO accuracy was also already verified to be over 80%
        @test (acc_poliedro) >= (acc_svm)/1.25 && (acc_poliedro) > 0.8
    end
end

"""
Testing on verified datasets
All have ~90% PolieDRO and Logistic Loss accuracy, if they fail it means something changed for the worse
"""
@testset "Logistic loss classification tests" begin
    # testing on some classification datasets
    some_datasets = ["balloons-a", "connectionist-bench", "hayes-roth"]

    for dataset in some_datasets
        println("===================")
        println("$(dataset) dataset test")
        println("Fetching dataset...")
        df = nothing
        try
            df = UCIData.dataset(dataset)
        catch
            println("Failed to fetch $(dataset), skipping")
            continue
        end
        println("Treating dataset...")
        Xtrain, Xtest, ytrain, ytest = dataset_aux.treat_df(df; classification=true)

        Xtrain_m = Matrix{Float64}(Xtrain)
        Xtest_m = Matrix{Float64}(Xtest)

        # training PolieDRO
        model, ll_evaluator = PolieDRO.build_model(Xtrain_m, ytrain; loss_function=PolieDRO.logistic_loss)
        println("Solving PolieDRO model...")
        PolieDRO.solve_model!(model, Ipopt.Optimizer; silent=true)
        println("Evaluating PolieDRO model...")
        ypoliedro = ll_evaluator(model, Xtest_m)

        # comparing to MLJ Logistic Loss
        println("Fitting logistic classifier...")
        mach = fit!(machine(LogisticClassifier(), Xtrain, categorical(ytrain)))
        println("Evaluating logistic classifier...")
        ylogistic = Vector{Float64}([MLJ.mode(x) for x in predict(mach, Xtest)])

        println("Calculating error metrics...")
        ypoliedro_abs = [yp >= 0.5 ? 1.0 : -1.0 for yp in ypoliedro]
        ylogistic_abs = [yp >= 0.5 ? 1.0 : -1.0 for yp in ylogistic]
        acc_poliedro = sum(ypoliedro_abs.==ytest)*100/length(ytest)
        acc_logistic = sum(ylogistic.==ytest)*100/length(ytest)

        println("Accuracy % on $(dataset) dataset")
        println("PolieDRO = $(acc_poliedro)")
        println("Logistic loss = $(acc_logistic)")
        println("===================")

        # test against logistic-25% performance
        # model should not be much worse than regular logistic classification
        # PolieDRO accuracy was also already verified to be over 80%
        @test (acc_poliedro) >= (acc_logistic)/1.25 && (acc_poliedro) > 0.8
    end
end

"""
Testing on verified datasets
All have similar PolieDRO and Lasso error, if they fail it means something changed for the worse
"""
@testset "MSE Regression tests" begin
    # testing on some regression datasets
    some_datasets = ["auto-mpg", "housing", "yacht-hydrodynamics"]

    for dataset in some_datasets
        println("===================")
        println("$(dataset) dataset test")
        println("Fetching dataset...")
        df = nothing
        try
            df = UCIData.dataset(dataset)
        catch
            println("Failed to fetch $(dataset), skipping")
            continue
        end
        println("Treating dataset...")
        Xtrain, Xtest, ytrain, ytest = dataset_aux.treat_df(df)

        Xtrain_m = Matrix{Float64}(Xtrain)
        Xtest_m = Matrix{Float64}(Xtest)

        # training PolieDRO
        println("Building PolieDRO model...")
        model, mse_evaluator = PolieDRO.build_model(Xtrain_m, ytrain; loss_function=PolieDRO.mse_loss)
        println("Solving PolieDRO model...")
        PolieDRO.solve_model!(model, Ipopt.Optimizer; silent=true)
        println("Evaluating PolieDRO model...")
        ypoliedro = mse_evaluator(model, Xtest_m)

        # comparing to MLJ Lasso
        println("Fitting Lasso...")
        mach = fit!(machine(LassoRegressor(), Xtrain, ytrain))
        println("Evaluating Lasso...")
        ylasso = predict(mach, Xtest)

        println("Calculating error metrics...")
        mse_poliedro = mean([(ypoliedro[i] - ytest[i])^2 for i in eachindex(ytest)])
        mse_lasso = mean([(ylasso[i] - ytest[i])^2 for i in eachindex(ytest)])

        println("MSE on $(dataset) dataset")
        println("PolieDRO = $(mse_poliedro)")
        println("Lasso = $(mse_lasso)")
        println("===================")

        # test against lasso+25% performance
        # model should never be much worse than lasso
        @test mse_poliedro <= mse_lasso*1.25
    end
end