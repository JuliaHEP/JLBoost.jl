# AbstractTrees.jl interface
# https://juliacollections.github.io/AbstractTrees.jl/stable/#The-Abstract-Tree-Interface

import AbstractTrees
using Tables: Tables

# Traits must be defined on the node *type*, not on instances.
AbstractTrees.ParentLinks(::Type{<:AbstractJLBoostTree}) = AbstractTrees.StoredParents()
AbstractTrees.SiblingLinks(::Type{<:AbstractJLBoostTree}) = AbstractTrees.ImplicitSiblings()
AbstractTrees.ChildIndexing(::Type{<:AbstractJLBoostTree}) = AbstractTrees.IndexedChildren()
AbstractTrees.NodeType(::Type{<:AbstractJLBoostTree}) = AbstractTrees.HasNodeType()
AbstractTrees.nodetype(::Type{<:AbstractJLBoostTree}) = AbstractJLBoostTree

AbstractTrees.parent(jlt::AbstractJLBoostTree) = jlt.parent
AbstractTrees.parent(jlt::WeightedJLBoostTree) = jlt.tree.parent

AbstractTrees.children(jlt::WeightedJLBoostTree) = AbstractTrees.children(jlt.tree)

"""
    FeatureSplitPredicate

Split decision: feature, threshold, and whether the left child is `x <= split_val`.
"""
struct FeatureSplitPredicate
    feature
    split_val
    inclusive::Bool
end

(f::FeatureSplitPredicate)(tbl) = begin
    col = Tables.getcolumn(tbl, f.feature)
    f.inclusive ? (col .<= f.split_val) : (col .< f.split_val)
end

"""
    AbstractTrees.nodevalue(jlt)

Named tuple of this node's fields: `weight` on leaves; `splitfeature`, `split`,
`gain`, and `weight` on split nodes.
"""
function AbstractTrees.nodevalue(jlt::AbstractJLBoostTree)
    if isempty(jlt.children) || ismissing(jlt.splitfeature)
        return (weight = jlt.weight,)
    else
        return (splitfeature = jlt.splitfeature,
                split = jlt.split,
                gain = jlt.gain,
                weight = jlt.weight)
    end
end

function AbstractTrees.nodevalue(jlt::WeightedJLBoostTree)
    merge((eta = jlt.eta,), AbstractTrees.nodevalue(jlt.tree))
end
