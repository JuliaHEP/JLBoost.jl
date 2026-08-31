# Trait helpers. Canonical AbstractTrees 0.4 traits are in
# abstract-tree-interface.jl (defined on types, not instances).

export parentlinks

parentlinks(::Type{<:AbstractJLBoostTree}) = AbstractTrees.StoredParents()
parentlinks(::AbstractJLBoostTree) = AbstractTrees.StoredParents()
