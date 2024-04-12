using Test
include("../src/QuickHull.jl")

# QuickHull tests
@testset "QuickHull tests" begin

    @testset "QuickHull 2D tests" begin
        # 2-Dimensional basic hulls test
        X_2D_1 = Float64[
            1    1  ;
            -1    1 ;
            1   -1  ;
            -1   -1 ;

            2    2  ;
            -2    2 ;
            2   -2  ;
            -2   -2 ;

            3    3  ;
            -3    3 ;
            3   -3  ;
            -3   -3 ;
        ]

        hulls_X_2D_1 = QuickHull.convex_hulls(X_2D_1)
        @test issetequal(hulls_X_2D_1[3], X_2D_1[1:4,:])
        @test issetequal(hulls_X_2D_1[2], X_2D_1[5:8,:])
        @test issetequal(hulls_X_2D_1[1], X_2D_1[9:12,:])

        X_2D_2 = Float64[
            0    0  ;

            1    1  ;
            -1    1 ;
            1   -1  ;
            -1   -1 ;

            2    2  ;
            -2    2 ;
            2   -2  ;
            -2   -2 ;

            3    3  ;
            -3    3 ;
            3   -3  ;
            -3   -3 ;
        ]

        hulls_X_2D_2 = QuickHull.convex_hulls(X_2D_2)
        @test issetequal(hulls_X_2D_2[3], X_2D_2[1:5,:])
        @test issetequal(hulls_X_2D_2[2], X_2D_2[6:9,:])
        @test issetequal(hulls_X_2D_2[1], X_2D_2[10:13,:])
    end

    @testset "QuickHull 3D tests" begin
        # 3-Dimensional basic hulls test
        X_3D_1 = Float64[
                1    1   1  ;
                1   -1   1  ;
                -1    1  1  ;
                -1   -1  1  ;
                1    1   -1 ;
                1   -1   -1 ;
                -1    1  -1 ;
                -1   -1  -1 ;

                2    2    2 ;
                2   -2    2 ;
                -2    2   2 ;
                -2   -2   2 ;
                2    2   -2 ;
                2   -2   -2 ;
                -2    2  -2 ;
                -2   -2  -2 ;

                3    3    3 ;
                3   -3    3 ;
                -3    3   3 ;
                -3   -3   3 ;
                3    3   -3 ;
                3   -3   -3 ;
                -3    3  -3 ;
                -3   -3  -3 ;
        ]

        hulls_X_3D_1 = QuickHull.convex_hulls(X_3D_1)
        @test issetequal(hulls_X_3D_1[3], X_3D_1[1:8,:])
        @test issetequal(hulls_X_3D_1[2], X_3D_1[9:16,:])
        @test issetequal(hulls_X_3D_1[1], X_3D_1[17:24,:])

        X_3D_2 = Float64[
                0    0   0  ;

                1    1   1  ;
                1   -1   1  ;
                -1    1  1  ;
                -1   -1  1  ;
                1    1   -1 ;
                1   -1   -1 ;
                -1    1  -1 ;
                -1   -1  -1 ;

                2    2    2 ;
                2   -2    2 ;
                -2    2   2 ;
                -2   -2   2 ;
                2    2   -2 ;
                2   -2   -2 ;
                -2    2  -2 ;
                -2   -2  -2 ;

                3    3    3 ;
                3   -3    3 ;
                -3    3   3 ;
                -3   -3   3 ;
                3    3   -3 ;
                3   -3   -3 ;
                -3    3  -3 ;
                -3   -3  -3 ;
        ]

        hulls_X_3D_2 = QuickHull.convex_hulls(X_3D_2)
        @test issetequal(hulls_X_3D_2[3], X_3D_2[1:9,:])
        @test issetequal(hulls_X_3D_2[2], X_3D_2[10:17,:])
        @test issetequal(hulls_X_3D_2[1], X_3D_2[18:25,:])
    end

end
