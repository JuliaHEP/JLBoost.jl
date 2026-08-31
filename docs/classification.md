# Binary classification

Fit a logistic booster on a `DataFrame`. The default loss is `LogitLogLoss`. Features default to every column except the target.

```julia
using JLBoost, RDatasets

iris = dataset("datasets", "iris")
iris.is_setosa = iris.Species .== "setosa"
target = :is_setosa
features = setdiff(names(iris), ["Species", "is_setosa"])

model = jlboost(iris, target)   # one tree, default depth
```

The model stores trees, the loss, and the target name:

```julia
trees(model)
model.loss      # LogitLogLoss()
model.target    # :is_setosa
```

Grow more trees, or cap depth:

```julia
model2 = jlboost(iris, target; nrounds = 2, max_depth = 2)
```

Leaf-wise (best-first) growth uses `max_leaves`. Set `max_depth = 0` so depth is not also enforced:

```julia
model3 = jlboost(iris, target; nrounds = 2, max_leaves = 8, max_depth = 0)
```

## Predict

`predict` returns the raw margin (sum of `eta * leaf` along the path). The model is also callable:

```julia
ŷ = predict(model, iris)
ŷ = model(iris)
ŷ = predict(vcat(model, model2), iris)
```

`AUC` and `gini` take a score and a binary target. For this loss the ranking is often clearer with `-ŷ`:

```julia
AUC(-ŷ, iris.is_setosa)
gini(-ŷ, iris.is_setosa)
```

## Shrinkage after fitting

`eta` is stored on each `WeightedJLBoostTree`, not multiplied into leaf values at training time:

```julia
t = trees(model)[1]
new_tree = 0.3 * t
predict(new_tree, iris)   # 0.3 times the original tree's scores
```

## Feature importance

```julia
feature_importance(model2, iris)
```

Columns are `Quality_Gain`, `Coverage`, and `Frequency` (split count).
