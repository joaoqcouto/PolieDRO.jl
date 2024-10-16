module dataset_aux
using DataFrames, CategoricalArrays, MLJ, StatsBase

# one hot encoding function
function one_hot_encode(df::DataFrame)
    # calculate the threshold for 20% of the total row count
    # columns with too many categories for their row count will be dropped
    threshold = 0.2 * nrow(df)
    
    # collect the names of the categorical columns
    categorical_cols = [col for col in names(df) if eltype(df[!, col]) <: CategoricalValue]
    onehot_cols = []
    
    for col in categorical_cols
        # Get the levels of the categorical column
        categories = levels(df[!, col])
        
        # check if the number of categories does not exceed the threshold
        if length(categories) <= threshold
            for category in categories
                # create new column for each category
                new_col_name = Symbol("$(col)_$(string(category))")
                push!(onehot_cols, new_col_name)
                df[!, new_col_name] = ifelse.(df[!, col] .== category, 1.0, 0.0)
            end
        end
    end
    
    # Drop the original categorical columns
    df = select(df, Not(categorical_cols))
    
    return df, onehot_cols
end

# prepare df for usage
function treat_df(df; classification=false)
    # getting which columns have a high null count (>30%)
    threshold = 0.3 * nrow(df)
    col_names = names(df)

    # first column is ID column
    cols_to_drop = [col_names[1]]

    # not considering y column
    for col in col_names[2:end-1]
        if count(ismissing, df[!, col]) > threshold
            push!(cols_to_drop, col)
        end
    end
    treated_df = select(df, Not(cols_to_drop))
    
    # removing remaining rows with null values
    dropmissing!(treated_df)

    # splitting X and Y from dataframe
    X = treated_df[:,1:end-1]
    y = treated_df[:,end]

    # one-hot encoding categorical variables in X
    Xnew, onehot_cols = one_hot_encode(X)
    X = Xnew

    # drop duplicates
    mask = nonunique(X)
    remove_idxs = [i for i in eachindex(mask) if mask[i]]
    deleteat!(X, remove_idxs)
    deleteat!(y, remove_idxs)

    # if classification, assert y column is 1 and -1
    # 1 will be given to whatever is the first value of the y column
    # if regression, assure it's a float vector
    if classification
        y_values = unique(y)
        y_target = y_values[1]
        y = [y_class==y_target ? 1.0 : -1.0 for y_class in y]
    else
        y = Float64.(y)
    end

    # Convert Int columns to Float64
    for col in names(X)
        if eltype(X[!, col]) <: Integer
            X[!, col] = Float64.(X[!, col])
        end
    end

    # standardize X
    X = mapcols(zscore, X;cols=Not(onehot_cols))

    # removing NaN columns
    threshold = 0.3 * nrow(X)
    col_names = names(X)
    cols_to_drop = []
    for col in col_names
        if count(isnan, X[!, col]) > threshold
            push!(cols_to_drop, col)
        end
    end
    X = select(X, Not(cols_to_drop))

    # standardize y if regression
    if !classification
        y = zscore(y)
    end

    # split in 5 pieces and return
    Xsplits, Ysplits = partition((X, y), 0.2, 0.2, 0.2, 0.2, rng=123, multi=true)

    return Xsplits, Ysplits
end

end # module