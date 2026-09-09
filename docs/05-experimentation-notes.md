# Experimentation notes

Geo-lift design: `experiment/geo_simulate.py` (40 simulated DMA-style
markets, known ground truth — see its docstring for why simulated rather
than the real UCI retail data), `experiment/geo_matching.py` (matched
control selection), `experiment/power.py` (MDE / required-duration
calculations), `experiment/analyze.py` (the actual DiD test). Pre-
registration: `docs/08-geo-lift-preregistration.md`.

## Validation result: does the design actually detect a known effect?

Same discipline as the MMM parameter recovery study
(`docs/04-parameter-recovery-validation.md`): don't just design an
experiment and assume it works — inject a known effect and check whether
the design recovers it before trusting the design on a real question.

A true **12%** lift was injected into the treatment market's post-period
revenue. The design (26-week pre-period matching + 8-week post-period DiD)
detected it: **estimated lift 17.1%** (z=9.4, p<0.001, significant at
alpha=0.05). The point estimate overshoots the true 12% — expected
estimation noise from a single 8-week draw against a tracking-error
standard error of ~277 (absolute revenue) per week, combined with an
imperfect match (pre-period correlation 0.69, not 1.0) — not a sign the
design is broken. Full numbers: `outputs/reports/geo_experiment_analysis.csv`.

The power analysis itself checked out too: at the matched pair's observed
tracking error, 8 weeks gives an MDE of ~5.1% — comfortably below the 12%
true effect, so **this design has more power than the plan assumed it
needed** (the pre-registration targeted a 10% minimum effect; theoretical
duration needed for the true 12% effect was ~1.5 weeks, not 8). That's a
useful finding on its own: either a shorter test would have sufficed, or
the extra duration is a reasonable safety margin for a real (noisier, more
adversarial) rollout — a real design would weigh the cost of the shorter
window against the value of a wider margin, not just take the theoretical
minimum.

## Why peeking inflates false positives

Checking a p-value partway through a fixed-duration test and stopping the
moment it crosses 0.05 doesn't behave like a single test at alpha=0.05 — it
behaves like several correlated tests, each with its own chance of a false
positive, and stopping at the first "significant" look inflates the
effective false-positive rate well above the nominal 5% (the more looks,
the worse). This is exactly why `experiment/analyze.py` is written to run
once, at the end of the fixed post-period, and why the pre-registration
doc states the stopping rule in advance rather than leaving it to be
decided in the moment a p-value looks favorable.

## Why sample size is fixed in advance

Choosing to extend a test because the current result "almost" reached
significance is a second form of the same problem — it's an undisclosed
extra look, chosen because of what the data already showed. The 8-week
duration in `docs/08-geo-lift-preregistration.md` was set from the power
analysis before the test was analyzed, and that duration is what
`experiment/analyze.py` actually uses — the commit history is the record
that the number wasn't adjusted after seeing the outcome.

## Multiple comparisons across channels

Running this design separately across several channels (or several
treatment markets) and flagging "significant" wherever alpha=0.05 is
crossed inflates the overall false-positive rate across the whole set of
tests, the same way repeated peeking does within one test. Not yet an issue
for this project (one treatment market, one channel) but the honest
answer for scaling it up: control the family-wise error rate (Bonferroni:
divide alpha by the number of simultaneous tests) or the false discovery
rate (Benjamini-Hochberg, less conservative, usually the better choice when
testing many channels/markets at once) rather than applying alpha=0.05
independently to each one.

## Statistical vs. practical significance

A detected effect can be statistically significant (unlikely to be pure
noise) while still being too small to justify acting on — reallocating
budget has a cost (organizational effort, risk of being wrong for reasons
outside the test), so the bar for "worth acting on" should be a minimum
practical effect size decided in advance, not "any p<0.05 result."
`docs/08-geo-lift-preregistration.md` states that minimum (10%) before
running the test, precisely so a small-but-significant result doesn't get
reframed as a win after the fact.
