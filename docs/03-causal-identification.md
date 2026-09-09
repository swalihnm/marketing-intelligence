# Causal identification

## Can the MMM's coefficients be read causally?

**Not on their own, and not by default.** `mmm/model.py`'s `beta_c`
coefficients are fit by a likelihood-based regression against observed
spend and revenue — they answer "what spend/revenue relationship best
explains this data under these priors," not "what would revenue have been
had spend been different." Those are the same question only under specific
assumptions, listed below, most of which are not verified by the model
itself.

### Confounders

- **Shared trend and seasonality driving both spend and revenue.**
  `mmm/model.py` explicitly models trend and two-harmonic seasonality as
  separate terms specifically to control for this — but only to the extent
  those terms are correctly specified. Any trend/seasonal shape the model's
  Fourier terms can't capture leaks into the channel coefficients instead.
- **Reverse causality: budget set as a percentage of forecast revenue.**
  This is the single biggest identification threat named in the spec, and
  it is **not addressed anywhere in this codebase**. If a business sets next
  month's marketing budget as a function of *expected* revenue (a common
  real practice), then spend and revenue are jointly determined, and no
  amount of trend/seasonality control fixes that — it needs either an
  instrument (a spend driver uncorrelated with revenue expectations, e.g. a
  platform CPM shock) or a natural experiment (a launch/pause not chosen in
  response to forecast revenu, which is exactly what `causal/did.py` uses).
  The simulator (`simulate/dgp.py`) does not generate spend this way — spend
  is generated independently of revenue shocks — so even the MMM's own
  parameter recovery validation (`docs/04`) does not test robustness to
  this threat. This is a real, unresolved gap, stated plainly rather than
  glossed over.
- **What's controlled for**: trend, seasonality (2 harmonics), price. What
  isn't: reverse causality (above), any confounder not simulated into the
  DGP in the first place (real data would have more).

### Quasi-experimental method implemented: difference-in-differences

Per the spec's own cut order ("drop IV analysis, keep DiD"), only DiD is
implemented; IV is not. `causal/did.py` simulates a channel launching in
one region at a known week, with a matched control region that never gets
the channel — and, critically, both regions share an identical underlying
trend (a genuine confounder: a rising regional trend correlated with the
timing of the "launch" decision, e.g. a market chosen for launch because
it was already growing). This tests whether DiD actually does what it's
supposed to do, the same validate-before-trusting move as the MMM recovery
study, rather than just implementing the estimator and assuming it works.

**Result:**

| | naive (before/after, launch region only) | DiD (launch region vs. control) |
|---|---|---|
| Estimated effect | £172,110 | £103,481 |
| True effect | £100,270 | £100,270 |
| % error | **71.6%** | **3.2%** |

`outputs/figures/fig_did_channel_launch.png` shows why: both regions trend
upward together before the launch (the shared confounder), then region A
jumps sharply above that shared trend at the launch week while the control
keeps trending as before. The naive estimate mistakes a large share of the
ordinary shared trend for a channel effect; DiD nets it out using the
control region and lands close to the true, known value.

(An earlier parameterization of this simulation, with a milder trend
relative to a symmetric seasonal cycle, produced naive ≈ DiD by
coincidence — a linear trend's pre/post difference and a sine-shaped
season's pre/post difference partially cancelled over that specific 26/26
week split. Documented in `causal/did.py`'s inline comment: that's a
property of that specific parameterization, not evidence DiD and naive are
interchangeable in general, so the trend was strengthened to remove the
coincidental cancellation and give an unambiguous illustration.)

### MMM estimate vs. quasi-experimental estimate for the same effect — not yet done

The spec's suggested senior-level move — "here is my MMM estimate, here is
a quasi-experimental estimate of the same effect, here is why they differ"
— is **not implemented**. `causal/did.py`'s simulation and
`simulate/dgp.py`'s MMM recovery simulation are separate, independently
parameterized scenarios; no single simulated dataset currently feeds both
an MMM fit and a DiD estimate for literally the same channel effect. Doing
that properly would mean extending the DGP to include a regional
launch/pause structure for one channel while keeping the others as normal
continuous spend — a real but bounded next step, not done here for time.

## Bottom line

Treat every `beta_c` in this repo's MMM output as descriptive, not causal,
until it's cross-checked the way `causal/did.py` cross-checks its own
scenario. DiD is implemented and validated against a known effect; MMM
coefficients and DiD estimates have not yet been reconciled on the same
underlying effect; reverse causality from budget-as-%-of-forecast remains
a stated, unresolved gap.
