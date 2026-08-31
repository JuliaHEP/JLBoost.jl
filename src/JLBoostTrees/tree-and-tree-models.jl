using AbstractTrees: AbstractNode

abstract type AbstractJLBoostTree{T} <: AbstractNode{T} end

mutable struct JLBoostTreeModel
	jlt::AbstractVector{T} where {T <: AbstractJLBoostTree} # of JLBoostTree
	loss # this should be a function with deriv defined
	target::Symbol
end

"""
	trees(jlt::JLBoostTreeModel)

Return the trees from a tree-model
"""
trees(jlt::JLBoostTreeModel) = jlt.jlt


+(v1::JLBoostTreeModel, v2::JLBoostTreeModel) = begin
	v3 = deepcopy(v1)
	v3.jlt = vcat(trees(v1), trees(v2))
	v3
end

"""
    JLBoostTree

A binary regression-tree node. Field names map to `XGBoost.Node` as follows:

| JLBoostTree       | XGBoost.Node     | Role |
| ----------------- | ---------------- | ---- |
| `weight`          | `leaf`           | leaf score `-G / (H + λ)`; stored on every node, used at predict time only on leaves |
| `splitfeature`    | `split`          | feature used at this split |
| `split`           | `split_condition`| threshold; left child is `x <= split` (XGBoost uses `x < split_condition`) |
| `gain`            | `gain`           | split gain |
| `children`        | `children`       | `[left, right]`; XGBoost also stores child ids as `yes` / `no` |
| `parent`          | (none)           | parent pointer (`StoredParents`) |
| (none)            | `cover`          | sum of hessians; JLBoost reports this via `feature_importance` Coverage |
| (none)            | `nmissing`       | missing-value child; JLBoost currently sends missings left |
| (none)            | `id`, `depth`    | XGBoost stores these; JLBoost computes depth with `treedepth` |
"""
mutable struct JLBoostTree{T} <: AbstractJLBoostTree{T}
    weight
	parent::Union{JLBoostTree, Nothing}
    children::AbstractVector{AbstractJLBoostTree} # this is deliberate kept as an vector of AbstractJLBoostTree; because we can genuinely mix and match types in htere
    # TODO store the node value as FeatureSplitPredicate so you can generalise it to include missing
    splitfeature
    split
    gain
    JLBoostTree(weight; parent=nothing) = new{nothing}(weight, parent, JLBoostTree[], missing, missing, missing)
    JLBoostTree(args...; kwargs...) = new{nothing}(args...; kwargs...)
end

"""
    WeightedJLBoostTree

A tree with an `eta` shrinkage factor (XGBoost `eta` / `learning_rate`).
XGBoost bakes `eta` into each `leaf` at training time; JLBoost keeps it as a
wrapper so trees can be reweighted after fitting (`0.3 * tree`).
"""
mutable struct WeightedJLBoostTree{T} <: AbstractJLBoostTree{T}
	tree::JLBoostTree
	eta::Number
    WeightedJLBoostTree(tree, eta) = new{nothing}(tree, eta)
end

Base.getproperty(jlt::WeightedJLBoostTree, sym::Symbol) = begin
	if sym == :eta
		return getfield(jlt, :eta)
	elseif sym == :tree
		return getfield(jlt, :tree)
	else
		return getproperty(getfield(jlt, :tree), sym)
	end
end

*(jlt::JLBoostTree, eta::Number) = WeightedJLBoostTree(jlt, eta)

*(eta::Number, jlt::JLBoostTree) = WeightedJLBoostTree(jlt, eta)

*(jlt::WeightedJLBoostTree, eta::Number) = begin
	jlt.eta *= eta
	jlt
end

*(eta::Number, jlt::WeightedJLBoostTree) = begin
	jlt.eta *= eta
	jlt
end


"""
	show(io, jlt, ntabs; splitfeature="")

Show a JLBoostTree
"""
function show(io, jlt::WeightedJLBoostTree, ntabs::I; kwargs...) where {I <: Integer}
	println(io, "eta = $(jlt.eta) (tree weight)")
	show(io, jlt.tree, ntabs;  kwargs...)
end


function show(io, jlt::JLBoostTree, ntabs::I; splitfeature="") where {I <: Integer}
    if ntabs == 0
        tabs = ""
    elseif ntabs < 0
        @warn "ntabs < 0, setting to 0"
        tabs = 0
    else ntabs >= 1
        tabs = reduce(*, ["  " for i = 1:ntabs])
    end
    if splitfeature != ""
        println(io, "")
        print(io, "$tabs -- $splitfeature")
        # TODO check if ismissing is necessary
        #if ismissing(jlt.splitfeature)
        # if no children then must be an end node
        if length(jlt.children) == 0
            println(io, "")
            println(io, "  $tabs ---- weight = $(jlt.weight)")
        else
            #println(io, "  $tabs ---- $(jlt.splitfeature), weight = $(jlt.weight)")
            #println(io, "  $tabs ---- $(jlt.splitfeature)")
        end
    elseif ismissing(jlt.splitfeature)
        println(io, "$tabs weight = $(jlt.weight)")
    else
        #println("$tabs $(jlt.splitfeature), weight = $(jlt.weight)")
    end

    if length(jlt.children) == 2
        show(io, jlt.children[1], ntabs + 1; splitfeature = "$(jlt.splitfeature) <= $(jlt.split)")
        show(io, jlt.children[2], ntabs + 1; splitfeature = "$(jlt.splitfeature) > $(jlt.split)")
    end
end

"""
	show(io, jlt)

Show a JLBoostTree or a Vector{JLBoostTree}
"""
function show(io::IO, jlt::AbstractJLBoostTree)
    show(io, jlt, 0)
end

function show(io::IO, ::MIME"text/plain", jlt::AbstractVector{T}) where T <: AbstractJLBoostTree
	for (i, tree) in enumerate(jlt)
		println(io, "Tree $i")
    	show(io, tree, 0)
    	println(io, " ")
    end
end