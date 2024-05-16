using Test
using Statistics, JuMP, HiGHS, UCIData, MLJ
include("../src/ConvexHulls.jl")
include("../src/PolieDRO.jl")

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

@testset "Iris classification test" begin
    iris_df = UCIData.dataset("iris")
    X = iris_df[:,2:5]
    y = iris_df[:,end]

    (Xtrain, Xtest), (ytrain, ytest) = partition((X, y), 0.8, rng=12345, multi=true)

    Xtrain_m = Matrix{Float64}(Xtrain)
    Xtest_m = Matrix{Float64}(Xtest)

    # separating each flower
    flower_types = ["Iris-setosa", "Iris-virginica", "Iris-versicolour"]

    @testset "$(flower_type) HL classification test" for flower_type in flower_types
        ytrain_v = Vector{Float64}([flower==flower_type ? 1.0 : -1.0 for flower in ytrain])
        ytest_v = Vector{Float64}([flower==flower_type ? 1.0 : -1.0 for flower in ytest])

        # hinge loss
        model = PolieDRO.build_model(Xtrain_m, ytrain_v, PolieDRO.hinge_loss)
        PolieDRO.solve_model!(model, HiGHS.Optimizer, silent=true)
        ymodel = PolieDRO.evaluate_model(model, Xtest_m)
        ymodel_abs = [yp >= 0 ? 1.0 : -1.0 for yp in ymodel]
        errors = sum(ymodel_abs.!=ytest_v)

        println("Error % classifying $(flower_type) = $(errors*100/length(ytest_v))% ($(errors)/$(length(ytest_v)) errors)")
        @test errors/length(ytest_v) < 0.1 # test if accuracy is within 90%
    end

    @testset "$(flower_type) LL classification test" for flower_type in flower_types
        ytrain_v = Vector{Float64}([flower==flower_type ? 1.0 : -1.0 for flower in ytrain])
        ytest_v = Vector{Float64}([flower==flower_type ? 1.0 : -1.0 for flower in ytest])

        # logistic loss
        model = PolieDRO.build_model(Xtrain_m, ytrain_v, PolieDRO.hinge_loss)
        PolieDRO.solve_model!(model, HiGHS.Optimizer, silent=true)
        ymodel = PolieDRO.evaluate_model(model, Xtest_m)
        ymodel_abs = [yp >= 0.5 ? 1.0 : -1.0 for yp in ymodel]
        errors = sum(ymodel_abs.!=ytest_v)

        println("Error % classifying $(flower_type) = $(errors*100/length(ytest_v))% ($(errors)/$(length(ytest_v)) errors)")
        @test errors/length(ytest_v) < 0.1 # test if accuracy is within 90%
    end
end