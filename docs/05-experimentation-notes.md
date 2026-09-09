**Status: not started.** Planned content per spec Section 6.5: why peeking
inflates false positives, why sample size is fixed in advance (no early
stopping on significance), how multiple comparisons across channels would
be handled (e.g. Bonferroni/Benjamini-Hochberg over the channel set), and
the difference between statistical and practical significance (an MDE that
is "detectable" at n but too small to matter for a reallocation decision).

Depends on `experiment/power.py` and `experiment/geo_matching.py`
(geo-lift design, matched market pairs, power analysis) — neither is built
yet. See `progress.md`.
