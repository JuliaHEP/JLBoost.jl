using JLBoost
using Test
using DataFrames
using LossFunctions: L2DistLoss, PoissonLoss

@testset "LossFunctions prediction-first derivatives" begin
    # Poisson curvature depends on the prediction, so it catches reversed
    # Hessian arguments as well as reversed gradients.
    for (target, prediction) in [(3.0, 0.0), (1.0, 0.7)]
        @test JLBoost.g(L2DistLoss(), target, prediction) ≈ 2 * (prediction - target)
        @test JLBoost.h(L2DistLoss(), target, prediction) ≈ 2
        @test JLBoost.g(PoissonLoss(), target, prediction) ≈ exp(prediction) - target
        @test JLBoost.h(PoissonLoss(), target, prediction) ≈ exp(prediction)
    end
end

@testset "L2 regression learns toward the target" begin
    df = DataFrame(x = zeros(8), y = fill(3.0, 8))
    model = jlboost(df, :y, [:x], zeros(8), L2DistLoss();
                    nrounds = 1, eta = 1.0, max_depth = 1, min_child_weight = 0)
    @test predict(model, df) ≈ fill(3.0, 8)

    # Repeated shrunken updates must approach the target, not diverge from it.
    model = jlboost(df, :y, [:x], zeros(8), L2DistLoss();
                    nrounds = 4, eta = 0.25, max_depth = 1, min_child_weight = 0)
    @test predict(model, df) ≈ fill(3 * (1 - 0.75^4), 8)

    # Independent weighted Newton leaf values, including L2 regularization.
    df = DataFrame(x = repeat([0.0, 1.0], inner = 4),
                   y = [-2.0, -1, -3, -2, 2, 3, 4, 3])
    weights = Float64[1, 2, 3, 4, 4, 3, 2, 1]
    lambda, eta = 2.0, 0.3
    model = jlboost(df, :y, [:x], zeros(8), L2DistLoss();
                    nrounds = 1, max_depth = 1, min_child_weight = 0,
                    weights, lambda, eta)
    leaf(rows) = eta * 2 * sum(weights[rows] .* df.y[rows]) /
                 (2 * sum(weights[rows]) + lambda)
    @test predict(model, df) ≈ vcat(fill(leaf(1:4), 4), fill(leaf(5:8), 4))
end

@testset "Built-in logistic derivative compatibility" begin
    loss = LogitLogLoss()
    for target in [0.0, 1.0], prediction in [-1.2, 0.0, 0.7]
        p = 1 / (1 + exp(-prediction))
        # Preserve the existing public target-first LogitLogLoss methods.
        @test deriv(loss, target, prediction) ≈ p - target
        @test deriv2(loss, target, prediction) ≈ p * (1 - p)
        @test JLBoost.g(loss, target, prediction) ≈ p - target
        @test JLBoost.h(loss, target, prediction) ≈ p * (1 - p)
    end
end
