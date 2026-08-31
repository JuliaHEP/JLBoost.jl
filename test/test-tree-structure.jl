using JLBoost
using Test
using DataFrames
using AbstractTrees

@testset "tree structure" begin
    df = DataFrame(x = Float64[1, 1, 1, 1, 0, 0, 0, 0],
                   y = [1, 1, 1, 1, 0, 0, 0, 0])
    m = jlboost(df, :y; nrounds = 1, max_depth = 1, min_child_weight = 0, eta = 1.0)
    wt = trees(m)[1]
    @test wt isa WeightedJLBoostTree
    @test wt.eta == 1.0

    node = wt.tree
    @test node isa JLBoostTree
    @test node.splitfeature === :x
    @test node.split == 0.0
    @test node.gain > 0
    @test length(node.children) == 2

    left, right = node.children
    @test left.parent === node
    @test right.parent === node
    @test JLBoost.is_left_child(left)
    @test JLBoost.is_right_child(right)
    @test left.weight < 0   # y = 0 rows (x <= 0)
    @test right.weight > 0  # y = 1 rows (x > 0)

    # leaf score formula -G/(H+λ) at warmstart 0 for logit loss
    # left: four y=0 → g=0.5 each, h=0.25 each → leaf = -2 / 1 = -2
    @test left.weight ≈ -2.0
    @test right.weight ≈ 2.0

    @testset "AbstractTrees" begin
        @test AbstractTrees.ParentLinks(node) isa AbstractTrees.StoredParents
        @test AbstractTrees.ChildIndexing(node) isa AbstractTrees.IndexedChildren
        @test AbstractTrees.parent(node) === nothing
        @test AbstractTrees.parent(left) === node
        @test AbstractTrees.children(node) == node.children
        @test AbstractTrees.isroot(node)
        @test !AbstractTrees.isroot(left)

        nv = AbstractTrees.nodevalue(left)
        @test haskey(nv, :weight)
        @test nv.weight == left.weight

        nv_split = AbstractTrees.nodevalue(node)
        @test nv_split.splitfeature === :x
        @test nv_split.split == node.split
        @test nv_split.gain == node.gain

        printed = sprint(print_tree, node)
        @test occursin("splitfeature", printed) || occursin("weight", printed)
    end

    @testset "get_features" begin
        @test get_features(m) == [:x]
        @test features(m) == [:x]
    end

    @testset "eta shrinkage is not baked into leaves" begin
        m2 = jlboost(df, :y; nrounds = 1, max_depth = 1, min_child_weight = 0, eta = 0.3)
        @test trees(m2)[1].eta ≈ 0.3
        @test trees(m2)[1].tree.children[1].weight ≈ left.weight
        @test predict(m2, df) ≈ 0.3 .* predict(m, df)
    end
end

@testset "stump leaf weight when no split" begin
    df = DataFrame(x = ones(8), y = ones(Int, 8))
    m = jlboost(df, :y; nrounds = 1, max_depth = 1, min_child_weight = 0)
    leaf = trees(m)[1].tree
    @test isempty(leaf.children) || ismissing(leaf.splitfeature) || length(leaf.children) == 0
    # all y=1, warmstart 0: G = 8*(-0.5) = -4, H = 8*0.25 = 2 → leaf = 2
    @test leaf.weight ≈ 2.0
    @test all(predict(m, df) .≈ 2.0)
end
