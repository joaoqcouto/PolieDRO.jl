#=
Testing on verified datasets
All have ~90% PolieDRO and SVC accuracy, if they fail it means something changed for the worse
=#
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
        Xsplits, ysplits = dataset_aux.treat_df(df; classification=true)
        test_idx = 1
        Xtest = Xsplits[test_idx]
        ytest = ysplits[test_idx]
        Xtrain = reduce(vcat, Xsplits[1:end .!= test_idx])
        ytrain = reduce(vcat, ysplits[1:end .!= test_idx])

        Xtrain_m = Matrix{Float64}(Xtrain)
        Xtest_m = Matrix{Float64}(Xtest)

        # training PolieDRO
        model, hl_evaluator = PolieDRO.build_model(Xtrain_m, ytrain, PolieDRO.hinge_loss)
        println("Solving PolieDRO model...")
        PolieDRO.solve_model!(model)
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

#=
Testing on verified datasets
All have ~90% PolieDRO and Logistic Loss accuracy, if they fail it means something changed for the worse
=#
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
        Xsplits, ysplits = dataset_aux.treat_df(df; classification=true)
        test_idx = 1
        Xtest = Xsplits[test_idx]
        ytest = ysplits[test_idx]
        Xtrain = reduce(vcat, Xsplits[1:end .!= test_idx])
        ytrain = reduce(vcat, ysplits[1:end .!= test_idx])

        Xtrain_m = Matrix{Float64}(Xtrain)
        Xtest_m = Matrix{Float64}(Xtest)

        # training PolieDRO
        model, ll_evaluator = PolieDRO.build_model(Xtrain_m, ytrain, PolieDRO.logistic_loss)
        println("Solving PolieDRO model...")
        PolieDRO.solve_model!(model)
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