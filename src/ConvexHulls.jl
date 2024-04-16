module ConvexHulls

using Polyhedra, LazySets, Distributions

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
    Xhulls = Vector{Matrix{Float64}}()
    for hull in XhullSets
        hullMatrix = reduce(hcat,hull)'
        push!(Xhulls, hullMatrix)
    end

    return Xhulls
end

#=
Convex hulls probabilities function
- Receives the output of the convex_hulls function, a list of H matrices NxD where each one is a convex hull and an error 0 ⋜ α ⋜ 1
    - As in the convex hulls output, it is assumed that the hulls are ordered outermost to innermost
    - The error α defines a 1-α confidence interval which will affect the size of the probability intervals
- Returns a list of tuples of size H
    - Each tuple represents the lower and upper limit of the probability interval assigned to the convex hull with the same index
    - For example, first probability tuple will always be (1,1) since we fix the outermost hull's probability to 1
=#
function hulls_probabilities(XHulls::Vector{Matrix{T}}, error::Float64) where T<:Float64
    @assert error > 0 "Choose a positive error"
    @assert error <= 1 "Choose an error <= 1"

    # TODO: Error must be such that confidence intervals don't overlap?

    # vector of probabilities (center of each interval)
    probabilities = []
    n_points = sum([size(XHulls[i],1) for i in eachindex(XHulls)]) # counting total points

    # start from smallest hull, adding points along the way
    # done this way since the outer hulls contain the inner ones
    set_points = 0
    for i in length(XHulls):-1:1
        set_points += size(XHulls[i],1)
        p = set_points/n_points
        pushfirst!(probabilities, p)
    end

    # calculating intervals
    intervals = []

    confidence_interval = 1 - error/2
    z = quantile(Normal(0.0, 1.0),confidence_interval)
    q = z/sqrt(n_points)

    for i in eachindex(probabilities)
        p = probabilities[i]
        p_lower = p - q*sqrt(p*(1 - p)/n_points)
        p_upper = p + q*sqrt(p*(1 - p)/n_points)
        push!(intervals, [p_lower, p_upper])
    end

    return intervals
end

end # module