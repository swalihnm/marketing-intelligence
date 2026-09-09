# Causal identification

**Status: not started.** This file is a placeholder describing what will be
written, not a finished analysis — do not read anything below as a claim
about the current model's causal validity. Until this section exists with a
supporting quasi-experimental estimate, `mmm/model.py`'s coefficients should
be treated as descriptive, not causal.

## What will go here (per spec Section 6.3)

- An explicit account of confounders: seasonality driving both spend and
  revenue; budget set as a percentage of forecast revenue (reverse
  causality — the single biggest identification threat in MMM); what the
  current model controls for (trend, seasonality, price) and what it
  doesn't.
- At least one quasi-experimental estimate on top of the MMM: planned as
  difference-in-differences (`causal/did.py`) on a channel that
  launched/paused in one region and not another, contingent on the Tier 2/3
  ingestion landing with region-level granularity. IV is the spec's first
  cut if time runs short — see `progress.md`'s cut order.
- A stated comparison: MMM estimate vs quasi-experimental estimate for the
  same channel/effect, and a documented account of why they differ if they
  do.
