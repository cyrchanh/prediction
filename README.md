# Predicting Task-Ordering Effects on Catastrophic Forgetting

## Overview

This repository investigates whether the effect of task ordering on catastrophic
forgetting in continual learning can be **predicted before training**, using a cheap,
Fisher-based signal, rather than only observed after the fact.

It builds on and extends two lines of work: (1) empirical demonstrations that task
order affects forgetting (Bell & Lawrence, 2022; and a related unpublished MSc
dissertation draft used here for framing, Chan 2026), and (2) recent diagnostic and
corrective work on Fisher-based importance estimation in Elastic Weight
Consolidation (Kirkpatrick et al., 2017; "EWC-DR", Liu & Chang; "IEWC", Wiest).

The core hypothesis under test:

> **H1:** A cheap Fisher-based conflict signal — computed by evaluating one task's
> gradient at another task's already-converged weights, weighted by that task's
> diagonal Fisher importance — correctly predicts which of two task orderings
> produces less forgetting, at a rate meaningfully better than the 50% chance
> baseline.

This is a deliberately narrow, tractable slice of a larger question (can *any*
n-task ordering be ranked cheaply before training?). The pairwise case is tested
first because it is the smallest unit on which the underlying signal can be
validated at all; a signal that cannot predict pairwise asymmetry has no
foundation for predicting full-sequence orderings.

## Status

| Stage | What it tests | Status |
|---|---|---|
| exp0 | Does the third-party `iewc` package install and run correctly? | ✅ Done, passed |
| exp1 | Does the conflict signal work at all, on one task pair? | ✅ Done — worked, but only after fixing a methodological bug (see below) |
| exp2 | Does the signal work *reliably*, across many task pairs? | ⬜ Not yet run — this is the actual test of H1 |

**No claim of general predictive validity has been established yet.** exp1's
result (n=1) is a proof-of-concept and a justification for running exp2, not
evidence in itself.

## Repository structure

\`\`\`
.
├── src/
│   ├── models.py        # MLP architecture (shared across all experiments)
│   ├── data.py           # Split-MNIST binary digit-pair task loading
│   ├── training.py        # train(), accuracy() helpers
│   └── signals.py         # the Fisher-based conflict signal (core mechanism)
├── experiments/
│   ├── exp0_iewc_toy_sanity_check.py
│   ├── exp1_single_pair.py
│   └── exp2_multi_pair_validation.py
├── results/
│   └── exp2_hit_rate.csv   # produced by exp2
└── data/                     # gitignored — MNIST download cache
\`\`\`

## Setup

\`\`\`bash
python -m venv .venv
source .venv/bin/activate
pip install -e path/to/iewc      # third-party Fisher-estimator package (see Dependencies)
pip install torch torchvision scipy
\`\`\`

## The three experiments

### exp0 — Sanity check

Confirms the `iewc` package's three importance estimators (`ef`, `ewc_dr`,
`ief_diag`) run correctly on a tiny 4-sample toy dataset, before trusting the
package on any real data.

\`\`\`bash
python experiments/exp0_iewc_toy_sanity_check.py
\`\`\`

**Expected output:** `ewc_dr` total importance should be roughly an order of
magnitude larger than `ef`'s (confirms the logit-reversal correction from
EWC-DR is active), and `ief_diag` should sit at a different, intermediate
magnitude.

### exp1 — Single-pair proof of concept

Trains two real MNIST binary-digit tasks (0 vs 1, and 2 vs 3), computes the
Fisher-based conflict signal in both directions, predicts which ordering is
safer, then actually trains both orderings and checks whether the prediction
was correct.

\`\`\`bash
python experiments/exp1_single_pair.py
\`\`\`

**Important note on this script's history:** the first version of this
experiment computed each task's Fisher matrix from an **independently
initialised** probe model, and compared the two matrices coordinate-wise. This
produced two near-identical, uninformative scores (0.0004 vs 0.0004) with no
predictive value — because two independently initialised networks organise
their hidden units under different, arbitrary bases, so comparing "parameter
47" across them is not meaningful (the same problem Centred Kernel Alignment,
Kornblith et al. 2019, exists to solve for representation comparison more
generally).

The corrected version evaluates **both directions from a single shared
trajectory per probe** — i.e., both the Fisher matrix and the alternate
task's gradient are computed at the *same* converged weights, guaranteeing a
consistent coordinate system. This produced a well-separated, meaningful
signal (0.155 vs 0.013) whose predicted direction matched the actual
observed forgetting outcome on this task pair.

**This shared-trajectory requirement is treated as a methodological finding
in its own right**, not just a bug fix — see the discussion in the
accompanying thesis draft, Section 5.3.

### exp2 — Multi-pair validation (the actual test of H1)

Repeats exp1's procedure across multiple task pairs (drawn from combinations
of Split-MNIST binary digit tasks) and reports the aggregate **hit rate**:
the proportion of pairs where the predicted safer ordering matched the
actual safer ordering, plus a binomial significance test against the 50%
chance baseline.

\`\`\`bash
python experiments/exp2_multi_pair_validation.py
\`\`\`

**Output:** `results/exp2_hit_rate.csv` (per-pair breakdown) and a printed
summary (`hit_rate`, `p-value`).

**How to interpret the result:**

| Result | Interpretation |
|---|---|
| Hit rate ≈ 0.5, p ≥ 0.05 | No evidence the signal predicts better than chance. A valid, reportable negative result. |
| Hit rate meaningfully > 0.5, p < 0.05 | Evidence supporting H1. |
| Hit rate = 1.0 on a small sample | Interesting but not yet conclusive — report the confidence interval, not just the point estimate, given the small number of pairs tested. |

## What this repository does *not* yet do

To keep scope realistic, the following are explicitly out of scope for the
current stage and are noted as future work rather than partially-built
features:

- Prediction of **full n-task orderings** (only pairwise asymmetry is tested)
- Comparison against the other two signals proposed in the wider thesis
  framing (Centred Kernel Alignment representational overlap; transfer
  asymmetry) — only the Fisher-conflict signal is implemented here
- Testing whether the ordering-prediction benefit survives once a mitigation
  method (EWC, replay) is applied during actual training — the present
  experiments deliberately measure naive, unregularised forgetting only
- Any architecture other than the single shared-backbone MLP described in
  `src/models.py`
- Multi-head / task-incremental output architectures — all experiments here
  use a single shared output head, which prior work (Chan, 2026) shows is
  the regime in which ordering effects are strongest, but also most
  entangled with output-layer/recency bias rather than feature-level
  forgetting

## Dependencies

- [`iewc`](https://github.com/Axym-Labs/iewc) — reference implementation of
  the empirical Fisher, EWC-DR logit-reversal correction, and improved
  empirical Fisher (IEF) importance estimators. Used here as an installed,
  unmodified dependency; not vendored into this repository. See its own
  citation block for the accompanying paper.
- PyTorch, torchvision, scipy

## Key references

- Kirkpatrick et al. (2017). Overcoming catastrophic forgetting in neural
  networks. *PNAS*.
- Liu & Chang. Elastic Weight Consolidation Done Right for Continual
  Learning ("EWC-DR").
- Wiest. Improved Elastic Weight Consolidation as an Optimization Constraint
  for Continual Learning ("IEWC").
- Bell, S. J. and Lawrence, N. D. (2022). The effect of task ordering in
  continual learning. *arXiv:2205.13323*.
- Kornblith, S., Norouzi, M., Lee, H., and Hinton, G. (2019). Similarity of
  neural network representations revisited. *ICML*.
- Chan, H. (2026). Task Ordering as a Predictive Tool in Continual Learning.
  MSc dissertation, University College London. [Used here for framing and
  experimental-design conventions, not as a source of results.]
