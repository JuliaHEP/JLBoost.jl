using StatsBase: sample

"""
    select_row_sampling_strategy(subsample)

Return a function `(n) -> row_indices` used once per boosting round.
`subsample == 1` returns `1:n` (no copy). Otherwise a sample without replacement.
"""
function select_row_sampling_strategy(subsample)
    if 0 < subsample < 1
        return function (n, args...; kwargs...)
            sample(1:n, round(Int, n * subsample); replace = false)
        end
    elseif subsample == 1
        return function (n, args...; kwargs...)
            1:n
        end
    else
        error("`subsample` must be within (0, 1]")
    end
end
