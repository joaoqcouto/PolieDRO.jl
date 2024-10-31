#=
Testing on verified datasets
All have similar PolieDRO and Lasso error, if they fail it means something changed for the worse
=#
@testset "Regression tests" begin
    # testing on some regression datasets
    some_datasets = ["concrete-slump-test-flow", "lpga-2009", "yacht-hydrodynamics"]

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
        Xsplits, ysplits = dataset_aux.treat_df(df; classification=false)
        test_idx = 1
        Xtest = Xsplits[test_idx]
        ytest = ysplits[test_idx]
        Xtrain = reduce(vcat, Xsplits[1:end .!= test_idx])
        ytrain = reduce(vcat, ysplits[1:end .!= test_idx])

        Xtrain_m = Matrix{Float64}(Xtrain)
        Xtest_m = Matrix{Float64}(Xtest)

        # training PolieDRO
        data_hulls = PolieDRO.calculate_convex_hulls(Xtrain_m)
        println("Building PolieDRO MSE model...")
        model, mse_evaluator = PolieDRO.build_model(Xtrain_m, ytrain, PolieDRO.mse_loss; hulls=data_hulls)
        println("Solving PolieDRO MSE model...")
        PolieDRO.solve_model!(model)
        println("Evaluating PolieDRO MSE model...")
        ypoliedro_mse = mse_evaluator(model, Xtest_m)

        println("Building PolieDRO MAE model...")
        model, mae_evaluator = PolieDRO.build_model(Xtrain_m, ytrain, PolieDRO.mae_loss; hulls=data_hulls)
        println("Solving PolieDRO MAE model...")
        PolieDRO.solve_model!(model)
        println("Evaluating PolieDRO MAE model...")
        ypoliedro_mae = mae_evaluator(model, Xtest_m)

        # comparing to MLJ Linear Regressor
        println("Fitting OLS...")
        mach = fit!(machine(LinearRegressor(), Xtrain, ytrain))
        println("Evaluating OLS...")
        yols = predict(mach, Xtest)

        println("Calculating error metrics...")
        mse_poliedro_mse = mean([(ypoliedro_mse[i] - ytest[i])^2 for i in eachindex(ytest)])
        mse_poliedro_mae = mean([(ypoliedro_mae[i] - ytest[i])^2 for i in eachindex(ytest)])
        mse_ols = mean([(yols[i] - ytest[i])^2 for i in eachindex(ytest)])

        println("MSE on $(dataset) dataset")
        println("PolieDRO MAE = $(mse_poliedro_mae)")
        println("PolieDRO MSE = $(mse_poliedro_mse)")
        println("OLS = $(mse_ols)")
        println("===================")

        # test against ols+25% performance
        # models should not be much worse than ols
        @test mse_poliedro_mse <= mse_ols*1.25
        @test mse_poliedro_mae <= mse_ols*1.25
    end
end