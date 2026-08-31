export get_features

"""
    features(jlt::JLBoostTree)
    features(jlt::JLBoostTreeModel)

Return all the features used in the tree
"""
get_features(jlt::JLBoostTreeModel) = get_features(trees(jlt))

get_features(jlt::WeightedJLBoostTree) = get_features(jlt.tree)

get_features(jlt::AbstractJLBoostTree) = begin
    d = get_features!(jlt, Dict{Symbol, Bool}())
    keys(d) |> collect |> sort
end

get_features(jlt::AbstractVector{<:AbstractJLBoostTree}) = begin
    d = get_features!(jlt[1])
    for jlt1 in @view(jlt[2:end])
        get_features!(jlt1, d)
    end
    keys(d) |> collect |> sort
end

get_features!(jlt::WeightedJLBoostTree, d = Dict{Symbol, Bool}()) = get_features!(jlt.tree, d)

get_features!(jlt::AbstractJLBoostTree, d = Dict{Symbol, Bool}()) = begin
    if isequal(jlt.splitfeature, missing)
        return d
    else
        d[jlt.splitfeature] = true
    end

    for c in jlt.children
        get_features!(c, d)
    end

    d
end

# exported as `features` from JLBoost.jl
features(jlt) = get_features(jlt)
