# v0.2.0
* Require Julia 1.10+ so the package runs on current LTS and Julia 1.11.
* Allow LossFunctions 1.x and CategoricalArrays 1.x; fill in missing `[compat]` entries (`Statistics`, `Serialization`).
* Fix AbstractTrees 0.4 traits (type-based `StoredParents` / `IndexedChildren`) so `print_tree` works.
* Set stump leaf scores to `-G/(H+λ)` when no split is found; subsample now actually trains on the sampled rows.
* Support per-row observation `weights`: gradients and hessians are scaled by the weights.
* Fix `predict` for trees deeper than one split (`.`&` bound tighter than `.||`, so left-leaf rows also received a right-subtree increment). Training margins now match XGBoost to ~10⁻⁸ (Float32) under matched exact-split logistic settings.
* Add GitHub Actions CI (Julia 1.10, 1.11, latest 1.x; Linux, macOS, Windows).
# v0.1.12
Minor bug fix for jlboost

# v0.1.11
Support DataFrames.jl 0.21

# v0.1.7
Support Table v1

# v0.1.4
* Updated to MLJBase
* Getting ready for JLBoostmlj migration

## v0.1.3
Supporting DataFrames v0.20

## v0.1.2
Bug fixes and better MLJ support

## v0.1.1
* MLJ.jl support
* Tables.jl support JDFFile and IndexedTables.jl etc
