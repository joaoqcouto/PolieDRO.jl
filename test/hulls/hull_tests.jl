#=
Testing convex hulls creation on hypercubes
=#
@testset "Convex hulls tests" begin

    # test for 4 convex hulls made by hypercubes of dimensions 2 up to 10
    # one test without the origin point and one test with the origin point
    N = 4
    for D = 2:10
        @testset "Hypercube $(D)D tests" begin
            println("Testing at $(D) dimensions")

            # hypercube tests
            X_1 = hypercubes_matrix(N, D)
            hulls_struct = PolieDRO.calculate_convex_hulls(X_1)
            for n = 1:N
                hull = N-n+1
                @test issetequal(hulls_struct.index_sets[hull], [i for i in (n-1)*(2^D)+1:(n)*(2^D)])
            end
    
            # including origin point
            X_2 = vcat([0 for i = 1:D]', X_1)
            hulls_struct = PolieDRO.calculate_convex_hulls(X_2)
            for n = 1:N
                hull = N-n+1
                if hull==N
                    @test issetequal(hulls_struct.index_sets[hull], [i for i in 1:(n)*(2^D)+1])
                else
                    @test issetequal(hulls_struct.index_sets[hull], [i for i in (n-1)*(2^D)+2:(n)*(2^D)+1])
                end
            end
        end
    end
end

#=
Testing convex hulls probability intervals calculation
=#
@testset "Probabilities test" begin

    # test for 2 -> 10 4 dimensional convex hulls
    # probability centers should be equally distributed since all hulls have the same number of points
    D = 4
    for N = 2:10
        @testset "$(N) hypercubes tests" begin
            # hypercube tests
            X = hypercubes_matrix(N, D)
            hulls_struct = PolieDRO.calculate_convex_hulls(X)
            PolieDRO.calculate_hulls_probabilities!(hulls_struct, 0.1)
            probabilities_X = hulls_struct.probabilities

            expected_probability = 1.0
            for i in eachindex(probabilities_X)
                avg_prob = Statistics.mean(probabilities_X[i])

                println("Hull $i probability interval")
                println("[ $(probabilities_X[i][1]) ; $(probabilities_X[i][2]) ]")
                println("Mean = $avg_prob")

                @test abs(avg_prob - expected_probability) < 1e-6 # since it is float math there is an imprecision
                expected_probability -= 1.0/N
            end
        end
    end
end