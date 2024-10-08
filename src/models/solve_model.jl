export solve_model!

"""
    solve_model!(model; optimizer=Ipopt.Optimizer, silent=true, attributes=nothing)

Solve model function

Uses the given solver to solve the PolieDRO model.
Modifies the struct with the results and sets the `optimized` bool in it to true

# Arguments
- `model::PolieDROModel`: A PolieDRO model struct, as given by the build_model function, to be solved
- `optimizer`: An optimizer as the ones used to solve JuMP models
    - Default: Ipopt optimizer
    - NOTE: For the logistic and MSE models, a nonlinear solver is necessary
- `silent::Bool`: Sets the flag to solve the model silently (without logs)
    - Default value: `false`
- `attributes::Union{Nothing, Dict{String}}`: Sets optimizer attribute flags given a dictionary with entries `attribute => value`
    - Default value: `nothing`
"""
function solve_model!(model::PolieDROModel; optimizer=Ipopt.Optimizer, silent::Bool=true, attributes::Union{Nothing, Dict{String}}=nothing)
    set_optimizer(model.model, optimizer)
    if (silent)
        set_silent(model.model)
    end

    if !isnothing(attributes)
        for key in keys(attributes)
            set_attribute(model.model, key, attributes[key])
        end
    end
    
    optimize!(model.model)
    v = object_dictionary(model.model)

    model.β0 = value(v[:β0])
    model.β1 = value.(v[:β1])
    model.optimized = true

    return
end