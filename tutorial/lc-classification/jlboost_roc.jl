# Λc signal vs background with JLBoost
#
# Realistic HEP example: train a logistic gradient-boosted tree on all kinematic
# and PID features, and compare it to the usual first analysis step — a single
# cut on log10(Lc_PT). The script stops at the ROC curve.
#
# Data are LHCb-like Monte Carlo generated for the RUB Data Analysis Block Course
# (2025). They are not stored in this repository. See README.md in this folder:
#   https://indico.global/event/14058/#28-exercises-samples-fom-roc
#
# From the JLBoost.jl repo root:
#   julia --project=tutorial/lc-classification -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   julia --project=tutorial/lc-classification tutorial/lc-classification/jlboost_roc.jl

using Random
using UnROOT
using DataFrames
using JLBoost
using Plots

const DATADIR = joinpath(@__DIR__, "data")

# Expected yields in the (unlabeled) data sample, from the course notebook sliders.
# FoM operating points are chosen for this S/B, not for the raw MC event counts.
const NSIG = 32150
const NBKG = 215599

# Course notebook names (see solutions/2_samples_fom_roc.jl):
#   "gini"         = s / √(s+b)          — usual HEP significance
#   "significance" = 2 s b / (s+b)²      — Gini impurity of a binary mix
fom_gini(s, b) = s / sqrt(s + b)
fom_significance(s, b) = 2 * s * b / (s + b)^2

# Variables that span orders of magnitude. log10 makes tree splits more uniform.
# Applied only when both samples are strictly positive (log of ≤0 is undefined).
const LOG_TRANSFORM = [
    "K_OWNPVIPCHI2", "pi_PT", "Xi_PT", "Lc_OWNPVVDRHO", "Lc_OWNPVVDZ",
    "pi_OWNPVIPCHI2", "Lc_OWNPVFDCHI2", "Xi_OWNPVIP", "K_PT", "Lc_PT",
]

# Numbers printed by the course solution for the 1D Lc_PT scan (full samples).
const COURSE_BASELINE = (
    sig_eff_gini = 0.6027515047291487,
    bkg_rej_gini = 0.8870625,
    sig_eff_signif = 0.5212381771281169,
    bkg_rej_signif = 0.921625,
)

"""
    fom_for_cut(sdf, bdf, var, cut, fom; op= >, tf=identity, nsig=0, nbkg=0)

Figure of merit after `op(tf(x), cut)` on column `var`.

If `nsig` / `nbkg` are set, counts are scaled to those expected yields so the
FoM matches a data-sized sample rather than the MC statistics. Pass `nsig=1,
nbkg=1` to get raw efficiencies (fraction of events passing the cut).
"""
function fom_for_cut(sdf, bdf, var, cut, fom; op = >, tf = identity, nsig = 0, nbkg = 0)
    fsig = nsig > 0 ? nsig / nrow(sdf) : 1.0
    fbkg = nbkg > 0 ? nbkg / nrow(bdf) : 1.0
    s = fsig * count(x -> op(tf(x), cut), sdf[!, var])
    b = fbkg * count(x -> op(tf(x), cut), bdf[!, var])
    # Empty selection is 0/0; Julia `argmax` treats NaN as larger than any finite FoM.
    (s + b) <= 0 && return -Inf
    return fom(s, b)
end

drop_nan(df) = filter(row -> all(x -> !(x isa Number && isnan(x)), row), df)

function load_root(path)
    f = ROOTFile(path)
    return drop_nan(DataFrame(LazyTree(f, "t")))
end

function require_samples()
    sig = joinpath(DATADIR, "signal.root")
    bkg = joinpath(DATADIR, "background.root")
    if isfile(sig) && isfile(bkg)
        return sig, bkg
    end
    error("""
    LHCb-like ROOT samples are not in the repository.

    They were generated for the RUB Data Analysis Block Course (2025):
      https://indico.global/event/14058/#28-exercises-samples-fom-roc

    Download `signal.root` and `background.root` from that page (CERNBox links),
    or request them from the course authors, and place both files in:
      $DATADIR
    """)
end

"""Scan a cut grid. Efficiencies use raw fractions; FoMs use NSIG/NBKG scaling."""
function scan_roc(sdf, bdf, var; cuts, tf = identity, nsig = NSIG, nbkg = NBKG)
    ginis = [fom_for_cut(sdf, bdf, var, c, fom_gini; tf, nsig, nbkg) for c in cuts]
    significances = [fom_for_cut(sdf, bdf, var, c, fom_significance; tf, nsig, nbkg) for c in cuts]
    sig_effs = [fom_for_cut(sdf, bdf, var, c, (s, b) -> s; tf, nsig = 1, nbkg = 1) for c in cuts]
    bkg_effs = [fom_for_cut(sdf, bdf, var, c, (s, b) -> b; tf, nsig = 1, nbkg = 1) for c in cuts]
    ig = argmax(ginis)
    isig = argmax(significances)
    return (; cuts, ginis, significances, sig_effs, bkg_effs,
            best_index_gini = ig, best_index_significances = isig,
            best_cut_gini = cuts[ig], best_cut_significances = cuts[isig])
end

function print_operating_points(label, roc)
    println(label)
    println("  Best cut gini: $(roc.best_cut_gini)")
    println("  Best cut significance: $(roc.best_cut_significances)")
    println("  Signal efficiency gini: $(roc.sig_effs[roc.best_index_gini])")
    println("  Background rejection gini: $(1 - roc.bkg_effs[roc.best_index_gini])")
    println("  Signal efficiency significance: $(roc.sig_effs[roc.best_index_significances])")
    println("  Background rejection significance: $(1 - roc.bkg_effs[roc.best_index_significances])")
end

# JLBoost.predict returns the raw logistic margin. A sigmoid maps that to (0, 1)
# so a cut on the score is the same kind of object as an XGBoost probability.
sigmoid(x) = 1 / (1 + exp(-x))

function split_df(df, frac_train = 0.8)
    n = nrow(df)
    idx = randperm(n)
    n_train = round(Int, frac_train * n)
    return df[idx[1:n_train], :], df[idx[n_train+1:end], :]
end

"""
Log-transform heavy-tailed features and restrict each column to the overlap of
signal and background. Different MC preselections can leave a region that is
pure signal or pure background; a tree will split there and look perfect on
MC without learning anything that applies to data.
"""
function preprocess(dfsig, dfbkg)
    local_sig = copy(dfsig)
    local_bkg = copy(dfbkg)
    feature_names = names(dfsig)
    for var in feature_names
        col_s = local_sig[!, var]
        col_b = local_bkg[!, var]
        if var in LOG_TRANSFORM && minimum(col_s) > 0 && minimum(col_b) > 0
            transform!(local_sig, Symbol(var) => ByRow(log10) => Symbol(var))
            transform!(local_bkg, Symbol(var) => ByRow(log10) => Symbol(var))
        end
        support_min = max(minimum(local_sig[!, var]), minimum(local_bkg[!, var]))
        support_max = min(maximum(local_sig[!, var]), maximum(local_bkg[!, var]))
        local_sig = filter(row -> support_min < row[var] < support_max, local_sig)
        local_bkg = filter(row -> support_min < row[var] < support_max, local_bkg)
    end
    local_sig.label .= 1.0
    local_bkg.label .= 0.0
    return local_sig, local_bkg, Symbol.(feature_names)
end

function roc_auc(sig_effs, bkg_effs)
    perm = sortperm(sig_effs)
    xs, ys = sig_effs[perm], (1 .- bkg_effs)[perm]
    pushfirst!(xs, 0.0); pushfirst!(ys, 1.0)
    push!(xs, 1.0); push!(ys, 0.0)
    sum((ys[2:end] .+ ys[1:end-1]) .* (xs[2:end] .- xs[1:end-1]) ./ 2)
end

# --- 1. Load labeled MC -------------------------------------------------------

sigpath, bkgpath = require_samples()
println("Loading ROOT samples from $DATADIR")
dfsig = load_root(sigpath)
dfbkg = load_root(bkgpath)
println("  signal: $(nrow(dfsig))  background: $(nrow(dfbkg))")
println("  features: $(names(dfsig))")

# --- 2. Baseline: one cut on log10(Lc_PT), full samples -----------------------
# This is exercise 2 of the course: scan the cut, pick the FoM maxima, draw ROC.
# Evaluated on all MC (no train/val split) so the numbers match the solution.

cut_values_pt = range(3.2, 4.0; length = 120)
roc_pt = scan_roc(dfsig, dfbkg, "Lc_PT"; cuts = cut_values_pt, tf = log10)
print_operating_points("\nBaseline (cut on log10(Lc_PT), full samples)", roc_pt)
println("  (course solution: ε_s gini=$(COURSE_BASELINE.sig_eff_gini), ",
        "1-ε_b gini=$(COURSE_BASELINE.bkg_rej_gini))")

# --- 3. Preprocess all features and hold out a validation sample --------------
# The 1D scan above has no extra degrees of freedom, so using the full sample is
# fair. A boosted tree can overfit; ROC and FoM for JLBoost are quoted on val.

Random.seed!(1234)
dfsig_p, dfbkg_p, features = preprocess(dfsig, dfbkg)
tsig_train, tsig_val = split_df(dfsig_p, 0.8)
tbkg_train, tbkg_val = split_df(dfbkg_p, 0.8)
train = vcat(tsig_train, tbkg_train)
println("\nAfter preprocess + 80/20 split:")
println("  train: $(nrow(train))  val: $(nrow(tsig_val) + nrow(tbkg_val))  nfeatures: $(length(features))")

# --- 4. Fit a logistic GBDT ---------------------------------------------------
# Defaults that are enough to beat the 1D cut without a long hyperparameter hunt:
# shallow trees (max_depth=3), modest shrinkage (eta=0.3), L2 on leaves (lambda=1).

nrounds = 40
max_depth = 3
eta = 0.3
println("\nFitting JLBoost: nrounds=$nrounds max_depth=$max_depth eta=$eta")
@time model = jlboost(train, :label, features;
                      nrounds = nrounds, max_depth = max_depth, eta = eta,
                      lambda = 1.0, verbose = false)

# --- 5. Score the validation sample and scan the same FoMs --------------------

tsig_val = copy(tsig_val)
tbkg_val = copy(tbkg_val)
tsig_val.jlboost = sigmoid.(predict(model, tsig_val))
tbkg_val.jlboost = sigmoid.(predict(model, tbkg_val))
println("  score signal val extrema: $(extrema(tsig_val.jlboost))")
println("  score background val extrema: $(extrema(tbkg_val.jlboost))")

lo = min(minimum(tsig_val.jlboost), minimum(tbkg_val.jlboost))
hi = max(maximum(tsig_val.jlboost), maximum(tbkg_val.jlboost))
# Drop the last grid point: a cut at max(score) with `>` selects nobody.
cuts_bdt = range(lo, hi; length = 401)[1:end-1]
roc_bdt = scan_roc(tsig_val, tbkg_val, "jlboost"; cuts = cuts_bdt)
print_operating_points("\nJLBoost (validation)", roc_bdt)

# Same 1D Lc_PT scan on the validation events (Lc_PT is already log10 here).
roc_pt_val = scan_roc(tsig_val, tbkg_val, "Lc_PT"; cuts = range(3.2, 4.0; length = 120))
print_operating_points("\nBaseline log10(Lc_PT) on the same validation sample", roc_pt_val)

println("\nROC AUC (ε_s vs 1-ε_b):")
println("  Lc_PT full sample:  $(roc_auc(roc_pt.sig_effs, roc_pt.bkg_effs))")
println("  JLBoost validation: $(roc_auc(roc_bdt.sig_effs, roc_bdt.bkg_effs))")
println("  Lc_PT validation:   $(roc_auc(roc_pt_val.sig_effs, roc_pt_val.bkg_effs))")

# --- 6. ROC overlay -----------------------------------------------------------

theme(:default; grid = false, linewidth = 2,
      guidefontsize = 14, legendfontsize = 12, tickfontsize = 12,
      foreground_color_legend = :transparent)

plt = plot(roc_pt.sig_effs, 1 .- roc_pt.bkg_effs;
           label = "ROC Lc_PT (full sample)", lw = 2, lc = 2,
           xlabel = "ε(signal)", ylabel = "1-ε(background)",
           legend = :bottomleft, xlims = (0, 1.01), ylims = (0, 1.01))
plot!(plt, roc_bdt.sig_effs, 1 .- roc_bdt.bkg_effs; label = "ROC JLBoost (val)", lw = 2, lc = 3)
scatter!(plt, [roc_pt.sig_effs[roc_pt.best_index_gini]],
         [1 - roc_pt.bkg_effs[roc_pt.best_index_gini]];
         label = "Lc_PT best Gini", marker = :circle, color = 32, ms = 6)
scatter!(plt, [roc_pt.sig_effs[roc_pt.best_index_significances]],
         [1 - roc_pt.bkg_effs[roc_pt.best_index_significances]];
         label = "Lc_PT best Significance", marker = :square, color = 651, ms = 6)
scatter!(plt, [roc_bdt.sig_effs[roc_bdt.best_index_gini]],
         [1 - roc_bdt.bkg_effs[roc_bdt.best_index_gini]];
         label = "JLBoost best Gini", marker = :diamond, color = 3, ms = 6)
scatter!(plt, [roc_bdt.sig_effs[roc_bdt.best_index_significances]],
         [1 - roc_bdt.bkg_effs[roc_bdt.best_index_significances]];
         label = "JLBoost best Significance", marker = :utriangle, color = 4, ms = 6)

out = joinpath(@__DIR__, "roc_jlboost_vs_lcpt.png")
savefig(plt, out)
println("\nWrote $out")

# --- 7. Which variables did the trees actually use? ---------------------------

imp = feature_importance(model, train)
println("\nFeature importance (top 8):")
show(first(sort(imp, :Quality_Gain, rev = true), 8), allcols = true)
println()
