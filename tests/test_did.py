from causal.did import estimate_did, simulate_two_region_launch


def test_did_closer_to_truth_than_naive():
    df, true_effect = simulate_two_region_launch()
    est = estimate_did(df)

    naive_error = abs(est["naive_estimate"] - true_effect)
    did_error = abs(est["did_estimate"] - true_effect)

    assert did_error < naive_error


def test_did_within_10_percent_of_truth():
    df, true_effect = simulate_two_region_launch()
    est = estimate_did(df)

    pct_error = abs(est["did_estimate"] - true_effect) / true_effect
    assert pct_error < 0.10


def test_control_region_shows_no_launch_effect():
    df, _ = simulate_two_region_launch()
    means = df.groupby(["region", "period"])["revenue"].mean()
    # region B's post-pre change should be much smaller than region A's,
    # since B never gets the channel launch — it only reflects the shared
    # trend/season confound.
    b_change = means["B_control", "post"] - means["B_control", "pre"]
    a_change = means["A_launch", "post"] - means["A_launch", "pre"]
    assert b_change < a_change
