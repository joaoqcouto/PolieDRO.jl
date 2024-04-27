using Test, Statistics
include("../src/ConvexHulls.jl")

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
            hulls_X_1 = ConvexHulls.convex_hulls(X_1)
            for n = 1:N
                hull = N-n+1
                @test issetequal(hulls_X_1[hull], [i for i in (n-1)*(2^D)+1:(n)*(2^D)])
            end
    
            # including origin point
            X_2 = vcat([0 for i = 1:D]', X_1)
            hulls_X_2 = ConvexHulls.convex_hulls(X_2)
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
            hulls_X = ConvexHulls.convex_hulls(X)
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
