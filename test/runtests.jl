using JLBoost
using Test
using DataFrames

@testset "smoke test" begin
    df = DataFrame(x = [1, 1, 1, 1, 0, 0, 0, 0], y = [1, 1, 1, 1, 0, 0, 0, 0])
    m = jlboost(df, :y; nrounds = 1, max_depth = 1, min_child_weight = 0, eta = 1.0)
    ŷ = predict(m, df)
    @test length(ŷ) == 8
    @test all(ŷ[1:4] .> 0)
    @test all(ŷ[5:8] .< 0)
    @test length(trees(m)) == 1
end

include("test-get_leaf_nodes.jl")
include("test-tree-structure.jl")
include("test-weights.jl")
include("test-xgboost-accuracy.jl")
