module PolieDRO

# dependencies
using JuMP, Ipopt

# convex hulls module
include("ConvexHulls.jl")

# core utils
include("models/utils.jl")

# models
include("models/build_model.jl")
include("models/prebuilt_models.jl")

# solver
include("models/solve_model.jl")

end # module