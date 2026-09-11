# Correlation-Sensitivity Study

How parameter recovery degrades as cross-channel spend correlation rises from 0.3 to 0.9 — the multicollinearity problem that makes MMM hard, and the reason `mmm/priors.py` uses informative priors rather than flat ones.

| spend correlation | headline coverage | mean % error (beta) | mean % error (alpha) | r_hat max | divergences |
|---|---|---|---|---|---|
| 0.3 | 100.0% | 28.1% | 21.4% | 1.0048 | 0 |
| 0.5 | 100.0% | 31.3% | 21.6% | 1.0060 | 0 |
| 0.7 | 100.0% | 35.3% | 24.8% | 1.0050 | 0 |
| 0.9 | 100.0% | 38.8% | 31.3% | 1.0037 | 0 |

Full numbers: `outputs/reports/correlation_sensitivity.csv`. Figure: `outputs/figures/fig_correlation_sensitivity.png`.