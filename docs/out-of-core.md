# Out-of-core fit with JDF

When a table does not fit in RAM, write it to [JDF](https://github.com/xiaodaigh/JDF.jl) and pass a `JDFFile`. Columns load one at a time; the `jlboost` / `predict` interface is the same as for a `DataFrame`.

```julia
using JLBoost, RDatasets, JDF

iris = dataset("datasets", "iris")
iris.is_setosa = iris.Species .== "setosa"
target = :is_setosa
features = setdiff(Symbol.(names(iris)), [:Species, :is_setosa])

savejdf("iris.jdf", iris)
irisdisk = JDFFile("iris.jdf")

model = jlboost(irisdisk, target, features)
ŷ = predict(model, irisdisk)

AUC(-ŷ, irisdisk[:, :is_setosa])
gini(-ŷ, irisdisk[:, :is_setosa])

rm("iris.jdf"; force = true, recursive = true)
```
