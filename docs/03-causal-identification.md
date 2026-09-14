# Causal identification

## Can the MMM's coefficients be read causally?

**Not on their own, and not by default.** `mmm/model.py`'s `beta_c`
coefficients are fit by a likelihood-based regression against observed
spend and revenue — they answer "what spend/revenue relationship best
explains this data under these priors," not "what would revenue have been
had spend been different." Those are the same question only under specific
assumptions. Three are addressed below with their own known-ground-truth
validations (same discipline as the MMM parameter recovery study); one
remains a stated, unaddressed gap.

### Confounders

- **Shared trend and seasonality driving both spend and revenue.**
  `mmm/model.py` explicitly models trend and two-harmonic seasonality as
  separate terms specifically to control for this — but only to the extent
  those terms are correctly specified. Any trend/seasonal shape the model's
  Fourier terms can't capture leaks into the channel coefficients instead.
- **Reverse causality: budget set as a percentage of forecast revenue** —
  now addressed. See below.
- **What's controlled for**: trend, seasonality (2 harmonics), price,
  reverse causality (via IV, below). What isn't: any confounder not
  simulated into a DGP in the first place (real data would have more).

### Quasi-experimental method implemented: difference-in-differences

Per the spec's own cut order ("drop IV analysis, keep DiD"), DiD was the
first method implemented; IV was added afterward anyway (below) because it
was the natural way to address reverse causality, which has no
regional-launch structure to exploit with DiD. `causal/did.py` simulates a
channel launching in one region at a known week, with a matched control
region that never gets the channel — and, critically, both regions share
an identical underlying trend (a genuine confounder: a rising regional
trend correlated with the timing of the "launch" decision, e.g. a market
chosen for launch because it was already growing). This tests whether DiD
actually does what it's supposed to do, rather than just implementing the
estimator and assuming it works.

**Result:**

| | naive (before/after, launch region only) | DiD (launch region vs. control) |
|---|---|---|
| Estimated effect | £173,237 | £104,608 |
| True effect | £101,396 | £101,396 |
| % error | **70.9%** | **3.2%** |

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

### Reverse causality: budget set as a function of forecast revenue

The spec names this the single biggest identification threat, and for most
of this project's build it was an explicitly documented, unaddressed gap.
`causal/reverse_causality.py` closes it with a known-ground-truth
simulation: budget is set partly in response to a noisy forecast of an
unobserved "market condition" process that *also* directly drives revenue
— the textbook simultaneity problem. An exogenous cost-shifter
(`cpm`, the spec's own suggested instrument: "a platform CPM shock") shifts
realized spend for reasons unrelated to that confound, and is used as an
instrument in two-stage least squares (2SLS).

**Result** (260 simulated weeks):

| | naive OLS (spend treated as exogenous) | IV / 2SLS (cpm as instrument) |
|---|---|---|
| Estimated beta | 4.017 | 3.495 |
| True beta | 3.5 | 3.5 |
| % error | **14.8%** | **0.14%** |

Instrument diagnostics: relevance (correlation between `cpm` and `spend`)
= −0.84, strongly relevant; exclusion check (correlation between `cpm` and
the unobserved confound, which should be ~0 since they're constructed
independently) = 0.012, negligible. `outputs/figures/fig_reverse_causality_iv.png`.

**Scope note, stated plainly**: this uses a linear spend→revenue
specification, not the full adstock+Hill-saturation MMM — 2SLS is a linear
method and doesn't directly apply to a nonlinear model without a
control-function or nonlinear-IV extension, which is out of scope here.
This demonstrates the identification mechanics (why reverse causality
biases naive estimates, and how a cost-shifter instrument corrects for it),
not a drop-in fix for `mmm/model.py` itself, which still does not account
for reverse causality.

### MMM estimate vs. quasi-experimental estimate for the same effect

`causal/mmm_vs_did.py` does what the spec calls a senior-level move: fits
a single-channel MMM (same model class as `mmm/model.py`) on **region A's
data alone** — the way an MMM is actually run in practice, no control
region — for the exact same channel-launch effect `causal/did.py` already
estimates with DiD. Same simulated dataset, same true effect, two
different methods.

**Result:**

| method | estimate | % error | r-hat max | divergences |
|---|---|---|---|---|
| True effect | £101,396 | — | — | — |
| MMM (region A alone) | £103,905 | 2.5% | **3.43** | **1,537 / 2,400 (64%)** |
| DiD (region B as control) | £104,608 | 3.2% | 1.0×–1.01× (clean) | 0 |

**Read the MMM row carefully — this is the actual finding, not the % error
column.** The MMM's point estimate looks fine, even slightly better than
DiD's. Its posterior is not trustworthy: r-hat of 3.43 (target: <1.01) and
divergences in 64% of draws mean the sampler could not properly explore
the posterior — individual parameters (`beta`, `alpha`, `s`, `k`) trade off
against each other in ways the likelihood can't resolve. `outputs/figures/fig_mmm_vs_did.png`
carries this warning directly on the chart, not just in this text, because
the bar chart alone looks like a clean agreement between methods and would
mislead a reader who didn't check diagnostics.

**Why the MMM is poorly identified here but DiD isn't** — this is the
actual "why they differ" answer the spec is asking for, and it's a
different one than "different confounder-handling":

- The MMM needs to observe how revenue responds to *spend varying across a
  meaningfully wide range* to separately identify decay (`alpha`) and
  saturation shape (`s`, `k`). Region A's channel is brand new: spend is
  exactly zero for the first 26 weeks, then fluctuates only mildly (±15%,
  after this was deliberately widened from a perfectly flat step function
  — see `causal/did.py`'s inline comment — which made things *even worse*,
  1,764 divergences) around one level for the remaining 26. That's not
  enough spend-level variation to trace out a saturation curve; the main
  4-channel recovery study (`docs/04`) that recovers these same parameters
  cleanly has 156 weeks of genuinely wide, randomly-varying spend per
  channel by construction.
- DiD needs none of that. It doesn't estimate a decay rate or a saturation
  curve at all — it only needs one comparable control unit and a
  before/after contrast. A brand-new channel with a short, narrow-range
  spend history is exactly the situation where **DiD is the more
  appropriate tool**, not a fallback for when MMM "isn't done yet."

That's the actual, defensible answer to "why might these two methods give
different numbers, and which should you trust more here": not because one
has a confounder and the other doesn't (both estimate the same true
effect reasonably well by point estimate), but because **MMM's data
requirements for a new channel are steeper than DiD's**, and checking
diagnostics — not just comparing point estimates — is what reveals that.

## Bottom line

Treat every `beta_c` in this repo's MMM output as descriptive, not causal,
unless cross-checked. DiD is implemented and validated against a known
effect. Reverse causality now has an IV-based check (on a simplified
linear specification). MMM and DiD have been reconciled on literally the
same simulated effect, with a substantive, diagnostics-driven answer for
why they can disagree. What's still not done: extending the IV correction
to the full nonlinear adstock+Hill model, and (necessarily, since it
requires real deployed data) confirming any of this transfers to a real
business's channel history.
