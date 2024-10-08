"""
    hulls_probabilities(XHulls, error)

Calculates the convex hulls and probabilities associated with the given data and builds the PolieDRO model for the specified loss function.

# Arguments
- `XHulls::Vector{Vector{Int64}}`: Structure of convex hulls as returned by the convex_hulls function
- `error::Float64`: Used to define a confidence interval for the probabilities associated to the hulls (read more in the README.md)

# Returns
- List of tuples of the probability intervals associated with each convex hull in the form:
    ```
    [
        [(lower end of the interval), (upper end of the interval),],            <= interval for outermost hull
        [(lower end of the interval), (upper end of the interval),],            <= interval for inner hull 1
        etc.
    ]
    ```
    List is the size of the given list of hulls.
    First tuple is always (1,1) because first hull contains all points
    
# Assertions
- Error must be positive within 0 and 1
"""
function hulls_probabilities(XHulls::Vector{Vector{Int64}}, error::Float64)
    @assert error > 0 "Choose a positive error"
    @assert error <= 1 "Choose an error <= 1"

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
    q = quantile(Normal(0.0, 1.0),confidence_interval)

    for i in eachindex(probabilities)
        p = probabilities[i]
        p_lower = p - q*sqrt(p*(1 - p)/n_points)
        p_upper = p + q*sqrt(p*(1 - p)/n_points)
        push!(intervals, [p_lower, p_upper])
    end

    return intervals
end