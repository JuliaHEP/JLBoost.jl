# Λc signal vs background

Train a logistic JLBoost model on LHCb-like Monte Carlo and compare it to a
single cut on `log10(Lc_PT)`. The example stops at the ROC curve.

## Data

These are **LHCb-like** samples generated for the
[RUB Data Analysis Block Course (2025)](https://indico.global/event/14058/#28-exercises-samples-fom-roc).
They are **not** stored in this repository.

Get `signal.root` and `background.root` from that Indico contribution (CERNBox
links on the page), or request them from the course authors. Place both files in
`tutorial/lc-classification/data/`.

The unlabeled test sample (`test.root`, with `Lc_M`) is not used here.

## Run

From the JLBoost.jl repository root:

```julia
using Pkg
Pkg.activate("tutorial/lc-classification")
Pkg.develop(path = ".")
Pkg.instantiate()
```

```bash
julia --project=tutorial/lc-classification tutorial/lc-classification/jlboost_roc.jl
```

A typical result is a validation ROC well above the 1D `Lc_PT` baseline
(AUC ~0.96 vs ~0.83). The script writes the overlay to `roc_jlboost_vs_lcpt.png`
(generated locally, not committed).
