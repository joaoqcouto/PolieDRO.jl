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