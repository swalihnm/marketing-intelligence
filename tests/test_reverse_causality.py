from causal.reverse_causality import (
    TRUE_BETA, iv_2sls_estimate, naive_ols_estimate, simulate_endogenous_budget,
)


def test_naive_ols_is_biased():
    df = simulate_endogenous_budget()
    naive = naive_ols_estimate(df)
    # spend is set partly in response to the same confound driving revenue,
    # so naive OLS should be measurably off true beta.
    assert abs(naive - TRUE_BETA) / TRUE_BETA > 0.05


def test_iv_closer_to_truth_than_naive():
    df = simulate_endogenous_budget()
    naive = naive_ols_estimate(df)
    iv = iv_2sls_estimate(df)
    assert abs(iv - TRUE_BETA) < abs(naive - TRUE_BETA)


def test_instrument_is_relevant():
    df = simulate_endogenous_budget()
    corr = df["cpm"].corr(df["spend"])
    # cpm must meaningfully predict spend for 2SLS to be valid
    assert abs(corr) > 0.3


def test_instrument_plausibly_exogenous():
    df = simulate_endogenous_budget()
    corr = df["cpm"].corr(df["latent_shock"])
    # constructed independently of the confound; should be small
    assert abs(corr) < 0.2
