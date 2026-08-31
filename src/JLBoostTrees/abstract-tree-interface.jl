# AbstractTrees.jl interface
# https://juliacollections.github.io/AbstractTrees.jl/stable/#The-Abstract-Tree-Interface
#
# Mirrors XGBoost.jl's `Node` in src/introspection.jl: children, node type, and
# print_tree. JLBoost additionally stores parent pointers (`StoredParents`).

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

Value stored on split nodes: the feature, threshold, and whether the left child
is `x <= split_val` (XGBoost's JSON dump uses strict `<` as `split_condition`).
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

For leaves, the value is named `leaf` to match XGBoost's dump field. For split
nodes, `split` / `split_condition` / `gain` match `XGBoost.Node`.
"""
function AbstractTrees.nodevalue(jlt::AbstractJLBoostTree)
    if isempty(jlt.children) || ismissing(jlt.splitfeature)
        return (leaf = jlt.weight,)
    else
        return (split = jlt.splitfeature,
                split_condition = jlt.split,
                gain = jlt.gain,
                weight = jlt.weight)
    end
end

function AbstractTrees.nodevalue(jlt::WeightedJLBoostTree)
    merge((eta = jlt.eta,), AbstractTrees.nodevalue(jlt.tree))
end
