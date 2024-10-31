"""
    calculate_hulls_probabilities!(hulls_struct, significance_level)

Calculates the convex hulls and probabilities associated with the given data and builds the PolieDRO model for the specified loss function.
Stores this data in the hulls struct.

# Arguments
- `hulls_struct::HullsInfo`: Structure of convex hulls as returned by the calculate_convex_hulls function
- `significance_level::Float64`: Used to define a confidence interval for the probabilities associated to the hulls (read more in the README.md)

# Data stored in the struct:
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
- Significance_level must be positive within 0 and 1
"""
function calculate_hulls_probabilities!(hulls_struct::HullsInfo, significance_level::Float64)
    @assert significance_level > 0 "Choose a positive significance_level"
    @assert significance_level <= 1 "Choose an significance_level <= 1"

    XHulls = hulls_struct.index_sets

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

    confidence_interval = 1 - significance_level/2
    q = quantile(Normal(0.0, 1.0),confidence_interval)

    for i in eachindex(probabilities)
        p = probabilities[i]
        p_lower = p - q*sqrt(p*(1 - p)/n_points)
        p_upper = p + q*sqrt(p*(1 - p)/n_points)
        push!(intervals, [p_lower, p_upper])
    end

    hulls_struct.significance_level = significance_level
    hulls_struct.probabilities = intervals

    return
end