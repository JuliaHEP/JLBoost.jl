using JLBoost
using Test
using DataFrames
using LossFunctions: L2DistLoss
using Random

@testset "Warm starts persist across logistic rounds" begin
    df = DataFrame(x = zeros(8), y = Float64[0, 0, 0, 1, 0, 1, 1, 1])
    baseline = collect(range(-0.4, 1.0; length = 8))
    original = copy(baseline)
    weights = Float64[1, 2, 3, 4, 4, 3, 2, 1]
    eta, lambda = 0.3, 0.7
    for rounds in [1, 2, 4]
        # Independent Newton updates for a constant leaf, on the full score.
        correction = 0.0
        for _ in 1:rounds
            p = 1 ./ (1 .+ exp.(-(baseline .+ correction)))
            G = sum(weights .* (p .- df.y))
            H = sum(weights .* p .* (1 .- p))
            correction -= eta * G / (H + lambda)
        end
        model = jlboost(df, :y, [:x], baseline, LogitLogLoss();
                        nrounds = rounds, max_depth = 1, min_child_weight = 0,
                        weights, eta, lambda)
        # Predictions remain the correction; callers add the external baseline.
        @test predict(model, df) ≈ fill(correction, 8)
        @test baseline == original
    end
end

@testset "Warm-start regression equals residual-target regression" begin
    x = collect(range(-2.0, 2.0; length = 40))
    baseline = 0.7 .+ 0.2 .* x
    df = DataFrame(x = x, y = baseline .+ ifelse.(x .< 0, -1.0, 2.0))
    residual = DataFrame(x = x, y = df.y .- baseline)
    original = copy(baseline)
    for subsample in [1.0, 0.5], weighted in [false, true], rounds in [1, 2, 4]
        weights = weighted ? Float64.(1 .+ mod.(1:40, 3)) : nothing
        settings = (; nrounds = rounds, max_depth = 2, eta = 0.3,
                      lambda = 0.5, min_child_weight = 0, subsample, weights)
        # Same random rows in each pair, testing baseline/weight row alignment.
        Random.seed!(821)
        warm = jlboost(df, :y, [:x], baseline, L2DistLoss(); settings...)
        Random.seed!(821)
        cold = jlboost(residual, :y, [:x], zeros(40), L2DistLoss(); settings...)
        @test predict(warm, df) ≈ predict(cold, residual)
        @test baseline == original
    end
end
