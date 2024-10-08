"""
    convex_hulls(X)

Calculates the convex hulls associated with the given matrix of points

Calculates the outer hull, removes the points present in the hulls, calculate the inner hull, repeat to calculate all hulls
*Currently does this with a LP model where it tries to create each point with a convex combination of the rest
*Looking into faster ways of solving this problem, since it is by far the biggest bottleneck

# Arguments
- `X::Matrix{T}`: Matrix NxD of points to calculate the hulls (N = number of points, D = dimension of points)

# Returns
- List of indices of each hull in the form:
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
        etc.
    ]
```
- List of indices which are not vertices
    - The last hull includes its inner points, so some of the points inside it are not vertices of the hull
"""
function convex_hulls(X::Matrix{T}) where T<:Float64
    N, D = size(X)

    # building initial model
    model = JuMP.Model(HiGHS.Optimizer)
    set_silent(model)

    # list of vector of indices for each hull (outer -> inner)
    hulls_idx_vector = Vector{Vector{Int64}}()
    free_points = ones(N) # indices free = 1; indices that are in a hull = 0

    # general convex combination variables
    @variable(model, α[i=1:N] .>= 0)
    @constraint(model, sum(α) == 1)

    # building problem for first point
    i = 1
    c = zeros(N)
    c[i] = 100
    Xv = view(X,i,:)
    Xtp = permutedims(X)
    @constraint(model, cc,  Xtp*α .== Xv)
    @objective(model, Min, sum(c[i]*α[i] for i in 1:N))

    # while there are more remaining points than D
    # at least D+1 points are necessary for the hulls
    n_hulls = 1
    first_free_point = 1
    while sum(free_points) > D
        println("Calculating hull $(n_hulls)...")

        hull_indices = Vector{Int64}()
        for i in first_free_point+1:N+1
            # skip point if it's already associated to a hull
            if (free_points[i-1] == 0) continue end

            # solving current problem
            JuMP.optimize!(model)
            if (objective_value(model) > 0)
                # vertex found, no way to build it as a combination of other points
                push!(hull_indices, i-1)
                free_points[i-1] = 0
            end
    
            if (i == N+1) break end # there is no next vertex
    
            # update objective
            c[i-1]=0
            c[i]=100
            @objective(model, Min, sum(c[i]*α[i] for i in 1:N))
    
            # update constraint
            Xv = view(X,i,:)
            set_normalized_rhs.(cc,Xv)
        end

        # store hulls
        push!(hulls_idx_vector, hull_indices)

        first_free_point = findfirst(==(1), free_points)

        # if there are no remaining free points, end
        if isnothing(first_free_point) break end

        # update objective
        # already associated points have high cost (they shouldn't be used in the convex combination)
        for i in 1:N
            c[i] = free_points[i]==1 ? 0 : 100
        end
        c[first_free_point] = 100
        @objective(model, Min, sum(c[i]*α[i] for i in 1:N))

        # update constraint
        Xv = view(X,first_free_point,:)
        set_normalized_rhs.(cc,Xv)

        n_hulls += 1
    end

    non_vertex_points = Int64[]
    for i in 1:N
        if free_points[i] == 1
            # add free points to list of non vertices and to last hull
            push!(non_vertex_points, i)
            push!(hulls_idx_vector[end], i)
        end
    end

    # return indices of each hull and indices which are not vertices (they go inside the last set)
    return hulls_idx_vector, non_vertex_points
end