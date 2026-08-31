using JLBoost
using Test
using DataFrames

# Observation weights scale gradient and hessian: g_i ← w_i g_i, h_i ← w_i h_i.
# Uniform rescaling of all weights must leave leaf values unchanged.

@testset "observation weights" begin
    df = DataFrame(x = Float64[1, 1, 1, 1, 0, 0, 0, 0],
                   y = [1, 1, 1, 1, 0, 0, 0, 0])

    unweighted = jlboost(df, :y; nrounds = 1, max_depth = 1, min_child_weight = 0)
    doubled = jlboost(df, :y; nrounds = 1, max_depth = 1, min_child_weight = 0,
                      weights = fill(2.0, 8))

    t0 = trees(unweighted)[1].tree
    t2 = trees(doubled)[1].tree
    @test t0.splitfeature == t2.splitfeature
    @test t0.split == t2.split
    @test t0.children[1].weight ≈ t2.children[1].weight
    @test t0.children[2].weight ≈ t2.children[2].weight
    @test predict(unweighted, df) ≈ predict(doubled, df)

    # Zero-weight rows should not pull the split
    w = ones(8)
    w[1:4] .= 0  # ignore the y=1 / x=1 rows
    ignore_pos = jlboost(df, :y; nrounds = 1, max_depth = 1, min_child_weight = 0, weights = w)
    # remaining mass is four y=0 rows: stump leaf ≈ -2, or a split whose right child is unused
    pred = predict(ignore_pos, df)
    @test all(pred[5:8] .< 0)

    @test_throws AssertionError jlboost(df, :y; weights = [1.0, 2.0])
end
