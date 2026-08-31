# JLBoost.jl

[![Test](https://github.com/JuliaHEP/JLBoost.jl/actions/workflows/Test.yml/badge.svg)](https://github.com/JuliaHEP/JLBoost.jl/actions/workflows/Test.yml)

A 100%-Julia implementation of gradient boosting regression trees (GBRT / GBDT).

This is the [JuliaHEP](https://github.com/JuliaHEP/JLBoost.jl) fork of [`xiaodaigh/JLBoost.jl`](https://github.com/xiaodaigh/JLBoost.jl). It requires **Julia 1.10+**.

Longer tutorials live in [`docs/`](docs/). A [Quarto](https://quarto.org) documentation site from that folder is planned.

## Install

```julia
using Pkg
Pkg.add(url="https://github.com/JuliaHEP/JLBoost.jl")
```

## Fit and predict

`jlboost` fits on any [Tables.jl](https://github.com/JuliaData/Tables.jl) table. The default loss is logistic (`LogitLogLoss`) for a binary target. Features default to every column except the target.

```julia
using JLBoost, DataFrames

df = DataFrame(
    x = Float64[1, 1, 1, 1, 0, 0, 0, 0],
    y = [1, 1, 1, 1, 0, 0, 0, 0],
)

model = jlboost(df, :y; nrounds = 2, max_depth = 2)
ŷ = predict(model, df)          # raw margin
ŷ = model(df)                   # same: the model is callable
```

A fitted `JLBoostTreeModel` holds the trees, the loss, and the target name:

```julia
trees(model)
model.loss      # LogitLogLoss()
model.target    # :y
```

Control boosting with keywords on `jlboost`:

```julia
jlboost(df, :y;
    nrounds = 10,
    max_depth = 3,          # depth-wise growth (set `0` if you use `max_leaves`)
    max_leaves = 0,         # leaf-wise / best-first growth when > 0
    eta = 0.3,              # shrinkage; kept on each tree, not baked into leaves
    lambda = 0,             # L2 on leaf scores
    gamma = 0,              # minimum gain to split
    subsample = 1,
    min_child_weight = 1,
    weights = nothing,      # optional per-row observation weights
)
```

Score several models by concatenating them:

```julia
predict(vcat(model, model), df)
```

## Observation weights

Pass a vector the same length as the table. Gradients and hessians are multiplied by `w_i`.

```julia
w = fill(2.0, nrow(df))
model_w = jlboost(df, :y; weights = w)
```

## Inspect and reweight trees

Leaf scores live in `tree.weight`. Shrinkage is a wrapper (`WeightedJLBoostTree.eta`), so you can rescale a tree after fitting:

```julia
t = trees(model)[1]
t.eta
0.3 * t                         # new tree with 30% of the original eta
feature_importance(model, df)   # Quality_Gain, Coverage, Frequency
```

## Regression

Swap in a [LossFunctions.jl](https://github.com/JuliaML/LossFunctions.jl) loss. Least squares is `L2DistLoss()`:

```julia
using LossFunctions: L2DistLoss

df = DataFrame(x = rand(100) .* 100)
df.y = 2 .* df.x .+ rand(100)

jlboost(df, :y, [:x], fill(0.0, nrow(df)), L2DistLoss(); max_depth = 2)
```

## Save and load

```julia
JLBoost.save(model, "model.jlb")
JLBoost.load("model.jlb")
```

## Tables.jl

Any column-accessible table works. For a custom type, define efficient `nrow`, `ncol`, and `view` if the generic methods are too slow.

## Documentation

| | |
| --- | --- |
| [docs/](docs/) | Tutorials (Quarto site planned) |
| [Classification](docs/classification.md) | Binary logistic example, AUC / gini, leaf-wise growth |
| [Out of core](docs/out-of-core.md) | Fit from a `JDF.JDFFile` without loading every column |
| [Give Me Some Credit](tutorial/give-me-some-credit/) | Larger example on disk |

## Fork ahead

* Julia 1.10+ (tested on 1.10, 1.11, and latest 1.x).
* Per-row observation `weights`: `jlboost(df, target; weights = w)` (XGBoost `DMatrix` `weight`).
* Training-set raw margins agree with XGBoost to ~10⁻⁸ under matched exact-split logistic settings. The 8-row stump is bit-identical.
* GitHub Actions tests: Julia 1.11 and latest 1.x on Linux, macOS, and Windows (push); Ubuntu + latest 1.x on pull requests.

## Limitations

* `Union{T, Missing}` features are not supported yet.
* Only a single target column (no multivariate models yet).
* Numeric and boolean features only (no categoricals yet).

## Related packages

* [EvoTrees.jl](https://github.com/Evovest/EvoTrees.jl)
* [JuML.jl](https://github.com/Statfactory/JuML.jl)

There is no MLJ interface in this fork. The old [`JLBoostMLJ.jl`](https://github.com/xiaodaigh/JLBoostMLJ.jl) wrapper is **archived** (last commit 2021) and depends on JLBoost 0.1, DataFrames 0.21, and MLJ 0.10–0.14, so it will not resolve against this package.
