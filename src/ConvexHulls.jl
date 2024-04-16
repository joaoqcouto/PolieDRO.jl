module ConvexHulls

using Polyhedra, LazySets

#=
Convex hulls function
- Receives a matrix of points NxD (points are lines, dimensions are columns)
- Returns a list of convex hulls, each one a matrix in the same structure
- First matrix is outermost hull, last matrix is innermost hull
    (+ innermost points if there aren't enough to form another hull)
=#
function convex_hulls(X::Matrix{T}) where T<:Float64
    N, D = size(X)
    
    Xpoints = [X[i,:] for i in 1:N]
    XhullSets = []
    n_hulls = 1

    # first convex hull and remaining inner points
    hull = LazySets.convex_hull(Xpoints)
    push!(XhullSets, hull)
    X_remaining = setdiff(Xpoints, hull)

    # while there are more remaining points than D
    # at least D+1 points are necessary for the hulls
    while size(X_remaining, 1) > D
        # create another hull
        hull = LazySets.convex_hull(X_remaining)

        # add it to the set of hulls
        push!(XhullSets, hull)

        # remove the points from the remaining set
        X_remaining = setdiff(X_remaining, hull)

        n_hulls += 1
    end

    # if there are any remaining points add them to the last set
    if size(X_remaining, 1) > 0
        XhullSets[end] = vcat(XhullSets[end], X_remaining)
    end

    # transform hull sets into matrices
    Xhulls = []
    for hull in XhullSets
        hullMatrix = reduce(hcat,hull)'
        push!(Xhulls, hullMatrix)
    end

    return Xhulls
end


end # module