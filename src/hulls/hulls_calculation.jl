export calculate_convex_hulls

"""
Convex hulls and probabilities structure

Stores information about the convex hulls that can be reused for separate PolieDRO models being trained in the same dataset
Not having to recalculate convex hulls drastically reduces model training time

# Fields
- `index_sets::Vector{Vector{Int64}}`: List of indices of each hull in the form:
```
    [
        [
            (index of point 1 in outermost hull),
            (index of point 2 in outermost hull),
            etc.
        ],
        [
            (index of point 1 in inner hull 1),
            (index of point 2 in inner hull 1),
            etc.
        ],
        (one entry per convex hull)
    ]
```
- `non_vertices::Vector{Int64}`: List of indices which are not vertices
    - The last hull includes its inner points, so some of the points inside it are not vertices of the hull
- `significance_level::Float64`: The significance level used to calculate the hulls' associated probabilities
- `probabilities::Vector{Vector{Int64}}`: The confidence interval associated with each convex hull in the form:
```
    [
        [
            (lower end of hull 1 interval),
            (upper end of hull 1 interval)
        ],
        [
            (lower end of hull 2 interval),
            (upper end of hull 2 interval)
        ],
        (one entry per convex hull)
    ]
```

"""
mutable struct HullsInfo
    index_sets::Vector{Vector{Int64}}
    non_vertices::Vector{Int64}
    significance_level::Float64
    probabilities::Vector{Vector{Float64}}
end

"""
    calculate_convex_hulls(X; silent=true)

Calculates the convex hulls associated with the given matrix of points

Calculates the outer hull, removes the points present in the hulls, calculate the inner hull, repeat to calculate all hulls
*Currently does this with a LP model where it tries to create each point with a convex combination of the rest
*Looking into faster ways of solving this problem, since it is by far the biggest bottleneck

# Arguments
- `X::Matrix{Float64}`: Matrix NxD of points to calculate the hulls (N = number of points, D = dimension of points)
- `silent::Bool`: Sets the flag to build the hulls silently (without logs)
    - Default value: `true`

# Returns
- Convex hulls structure that can be passed to a model builder function
"""
function calculate_convex_hulls(X::Matrix{T}; silent::Bool=true) where T<:Float64
    N, D = size(X)

    # building initial model
    model = JuMP.Model(GLPK.Optimizer)
    set_silent(model)

    # list of vector of indices for each hull (outer -> inner)
    hulls_idx_vector = Vector{Vector{Int64}}()
    free_points = ones(N) # indices free = 1; indices that are in a hull = 0

    # general convex combination variables
    @variable(model, α[i=1:N] .>= 0)
    @constraint(model, sum(α) == 1)

    Xtp = permutedims(X)
    @constraint(model, cc,  Xtp*α .== view(X,1,:))

    # until we can't make more hulls
    n_hulls = 1
    first_free_point = 1
    while true
        # going through points to make a hull
        hull_indices = Vector{Int64}()
        if (!silent) println("Hull $(n_hulls)...") end
        for i in first_free_point:N
            # skipping points already in a hull
            if (free_points[i] == 0) continue end

            # set point constraint
            set_normalized_rhs.(cc,view(X,i,:))

            @objective(model, Min, α[i])

            # solving point
            JuMP.optimize!(model)
            if (objective_value(model) > 0)
                # point is vertex
                push!(hull_indices, i)
                free_points[i] = 0
            end
        end

        # quit if there are too few points in this hull (no progress)
        if length(hull_indices) <= D
            if (!silent) println("No more hulls can be formed") end
            for i in hull_indices free_points[i] = 1 end
            break
        end

        # storing hull
        push!(hulls_idx_vector, hull_indices)
        n_hulls += 1

        # updating first free point
        first_free_point = findfirst(==(1), free_points)

        # locking points in hull to 0 (can't be used)
        for i in hull_indices
            fix(α[i], 0; force=true)
        end

        # quit if there are no points left (it's over)
        if isnothing(first_free_point)
            if (!silent) println("All point are in hulls") end
            break
        end
    end

    # create list of non vertices and add to last hull
    non_vertex_points = [i for i in 1:N if free_points[i] == 1]
    append!(hulls_idx_vector[end], non_vertex_points)

    # return HullsInfo struct without probabilities
    return HullsInfo(hulls_idx_vector, non_vertex_points, NaN, [[] for hull in hulls_idx_vector])
end