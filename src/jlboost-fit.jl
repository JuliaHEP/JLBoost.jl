export jlboost!, jlboost

using DataFrames: nrow, ncol
using Tables

using ..JLBoostTrees: JLBoostTree

"""
    jlboost(df, target, features = setdiff(names(df), (target, prev_w, new_weight)),
        warm_start = fill(0.0, nrow(df)); nrounds = 1, eta = 1.0, lambda = 0, gamma = 0,
        max_depth = 6, subsample = 1, colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1,
        verbose = false)

Fit a tree boosting model with a DataFrame, df, and target symbol and allowed features.

This is based on the xgboost interface, where possible the parameters have the same name as xgboost,
see https://xgboost.readthedocs.io/en/latest/parameter.html

* nrounds: Number of trees to fit
* warmstart: A vector of scores from which to start training. Defaults to 0. The warmstart may be
    different for every row. This is designed to allow the model to improve upon existing models.
    (XGBoost's analogue is `base_margin`.)
* eta: The learning rate. Also known as the weight of each tree in the final summation of trees.
    XGBoost bakes `eta` into leaf values; JLBoost stores it on `WeightedJLBoostTree`.
* lambda: XGBoost lambda hyperparameter
* gamma: XGBoost gamma hyperparameter
* max_depth: the maximum depth of each tree
* subsample: 0-1, the proportion of rows to subsample for each tree build
* verbose: Print more information
* colsample_bytree: (0-1] The proportion of feature column to sample for each tree.
* min_child_weight: The weight that needs to be in each child node before a split can occur. The
    weight is the hessian (2nd derivative) of the loss function, which happens to be 1 for squares
    loss. Same meaning as XGBoost `min_child_weight`.
* colsample_bylevel: Not yet implemented
* colsample_bynode: Not yet implemented
* monotone_contraints: Not yet implemented
* interaction_constraints: Not yet implemented
"""
function jlboost(df, target::Union{Symbol, String}; kwargs...)
    target = Symbol(target)
    warm_start = fill(0.0, nrow(df))
	jlboost(df, target, setdiff(Tables.columnnames(df), [target]), warm_start; kwargs...)
end

function jlboost(df, target::Union{Symbol, String}, warm_start::AbstractVector{T}; kwargs...) where T <: Number
    target = Symbol(target)
	jlboost(df, target, setdiff(names(df), [target]), warm_start; kwargs...)
end

function jlboost(df, target::Union{Symbol, String}, features::AbstractVector{T}; kwargs...) where T <: Union{String, Symbol}
    target = Symbol(target)
    features = Symbol.(features)
	jlboost(df, target, features, fill(0.0, nrow(df)); kwargs...)
end

function jlboost(df, target::Union{Symbol, String}, features::AbstractVector,
    warm_start::AbstractVector, loss = LogitLogLoss();
    subsample = 1, colsample_bytree = 1, max_depth = 6, max_leaves = 0, kwargs...)

    @assert 0 < subsample <= 1
    @assert 0 < colsample_bytree <= 1
    @assert Tables.istable(df)

    target = Symbol(target)
    features = Symbol.(features)

    # a sample of the rows: returns row indices
    row_sampling_bytree_strategy = select_row_sampling_strategy(subsample)

    # a function to sample the columns
    col_sampling_bytree_strategy = select_col_sampling_strategy(colsample_bytree)

    if max_leaves > 0
        if max_depth > 0
            @warn "You have set max_leaves=$max_leaves but max_depth > 0. The max_depth parameter is ignored."
        end
        tree_growth = lossguide
        stopping_criterion = max_leaves_stopping_criterion(max_leaves)
    else
        tree_growth = depth_wise
        stopping_criterion = max_depth_stopping_criterion(max_depth)
    end

    # TODO look at target column and provide a possible selection of loss
    # e.g. if the target is numeric then RSMELoss is more appropriate
    jlboost(df, target, features, warm_start, loss,
            row_sampling_bytree_strategy,
            col_sampling_bytree_strategy,
            tree_growth,
            stopping_criterion; kwargs...)
end

# the most canonical version of jlboost is here
function jlboost(df, target, features, warm_start::AbstractVector,
    loss,
    row_sampling_strategy::Function,
    col_sampling_bytree_strategy::Function,
    tree_growth::Function,
    stopping_criterion::Function;
	nrounds = 1, eta = 1.0, verbose = false, kwargs...)
    # eta = 1, lambda = 0, gamma = 0,  min_child_weight = 1, colsample_bylevel = 1, colsample_bynode = 1,
	#, ,  colsample_bynode = 1,

    @assert nrounds >= 1
	@assert Tables.istable(df)

    target = Symbol(target)
    features = Symbol.(features)

    n = nrow(df)
    if length(warm_start) != n
        throw(ArgumentError("warm_start length ($(length(warm_start))) must equal number of rows ($n)"))
    end

    # TODO get only the needed columns from the table
	# dfc = Tables.columns(df)
    dfc = df

    # res_jlt = result JLBoost trees
	res_jlt = AbstractJLBoostTree[]

    # fit the next round
	for nround in 1:nrounds
		if verbose
			println("Fitting tree #$(nround)")
		end

        # sample new columns
		features_sample = col_sampling_bytree_strategy(features, df, target, warm_start, loss;
                                                   nrounds=nrounds, eta=eta, kwargs...)
        idx = row_sampling_strategy(nrow(dfc))
        if idx == 1:nrow(dfc)
            dfs = dfc
            ws = nround == 1 ? warm_start : predict(res_jlt[1:nround-1], dfc)
        else
            dfs = dfc[idx, :]
            if nround == 1
                ws = warm_start[idx]
            else
                ws = predict(res_jlt[1:nround-1], dfs)
            end
        end

        new_jlt = _fit_tree!(loss, dfs, target, features_sample, ws, JLBoostTree(0.0),
                             tree_growth,
                             stopping_criterion; verbose=verbose, kwargs...);

        # added a new round of tree
        push!(res_jlt, eta*deepcopy(new_jlt))
	end
	res_jlt


    JLBoostTreeModel(res_jlt, loss, target)
end