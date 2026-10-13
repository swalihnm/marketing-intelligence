import numpy as np

from causal.nonlinear_iv import first_stage_diagnostics, first_stage_residual, simulate


def test_simulate_produces_known_true_contribution():
    df, true_avg = simulate()
    assert true_avg > 0
    assert len(df) == 156
    assert {"revenue", "spend_x", "cpm", "price", "latent_shock"} <= set(df.columns)


def test_spend_has_wide_variation_not_a_step_function():
    df, _ = simulate()
    # unlike causal/mmm_vs_did.py's step-function spend, this should vary
    # continuously across a wide multiplicative range
    assert (df["spend_x"] > 0).all()
    assert df["spend_x"].max() / df["spend_x"].min() > 2.0


def test_first_stage_residual_is_uncorrelated_with_instrument():
    df, _ = simulate()
    v_hat = first_stage_residual(df)
    # OLS residuals are orthogonal to the regressor (cpm) by construction
    corr = np.corrcoef(df["cpm"], v_hat)[0, 1]
    assert abs(corr) < 1e-6


def test_first_stage_diagnostics_shape():
    df, _ = simulate()
    v_hat = first_stage_residual(df)
    diag = first_stage_diagnostics(df, v_hat)
    assert 0 <= diag["first_stage_r2"] <= 1
    assert -1 <= diag["residual_spend_corr"] <= 1
