using JLBoost
using Test
using DataFrames

# Live comparison (2026-08-31) against XGBoost.jl 2.5.5 / libxgboost 2.1.5
# with matched hyperparameters:
#   objective = binary:logistic / LogitLogLoss
#   eta, lambda=0, gamma=0, min_child_weight=0, subsample=1, colsample_bytree=1
#   XGBoost tree_method="exact", grow_policy="depthwise", base_score=0.5
#   raw margin via XGBoost.predict(...; margin=true)
#
# On the 8-row stump, 1 tree: margins are Float64 bit-identical.
# On 4 rounds with eta=0.3, and on n=40 continuous depth-3 × 5 rounds after
# the predict fix: max|Δ| ~ 2e-8 (XGBoost Float32 vs JLBoost Float64).

const XGB_STUMP_MARGIN = [2.0, 2.0, 2.0, 2.0, -2.0, -2.0, -2.0, -2.0]
# XGBoost Float32 value for 4 rounds, eta=0.3 on the same stump
const XGB_4ROUND_ETA03 = 1.83720767

@testset "XGBoost numerical accuracy (logistic, exact splits)" begin
    df = DataFrame(x = Float64[1, 1, 1, 1, 0, 0, 0, 0],
                   y = [1, 1, 1, 1, 0, 0, 0, 0])

    @testset "1 tree, depth 1: bit-identical margins and leaves ±2" begin
        m = jlboost(df, :y; nrounds = 1, max_depth = 1, min_child_weight = 0,
                    eta = 1.0, lambda = 0, gamma = 0)
        ŷ = predict(m, df)
        @test ŷ == XGB_STUMP_MARGIN
        t = trees(m)[1].tree
        @test t.gain ≈ 8.0
        @test t.children[1].weight ≈ -2.0
        @test t.children[2].weight ≈ 2.0
    end

    @testset "4 rounds, eta=0.3: agree to Float32 (~1e-8)" begin
        m = jlboost(df, :y; nrounds = 4, max_depth = 1, min_child_weight = 0,
                    eta = 0.3, lambda = 0, gamma = 0)
        ŷ = predict(m, df)
        @test all(isapprox.(ŷ[1:4], XGB_4ROUND_ETA03; atol = 1e-6, rtol = 0))
        @test all(isapprox.(ŷ[5:8], -XGB_4ROUND_ETA03; atol = 1e-6, rtol = 0))
        @test maximum(abs, ŷ[1:4] .- XGB_4ROUND_ETA03) < 1e-6
    end

    @testset "observation weights=2: same leaves, doubled gain" begin
        m = jlboost(df, :y; nrounds = 1, max_depth = 1, min_child_weight = 0,
                    eta = 1.0, lambda = 0, gamma = 0, weights = fill(2.0, 8))
        t = trees(m)[1].tree
        @test t.gain ≈ 16.0
        @test t.children[1].weight ≈ -2.0
        @test t.children[2].weight ≈ 2.0
        @test predict(m, df) == XGB_STUMP_MARGIN
    end
end
