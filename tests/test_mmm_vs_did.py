from causal.did import simulate_two_region_launch


def test_region_a_panel_has_mmm_required_columns():
    df, _ = simulate_two_region_launch()
    region_a = df[df["region"] == "A_launch"]
    for col in ["t", "revenue", "price", "spend"]:
        assert col in region_a.columns
    assert region_a["price"].std() > 0  # not a degenerate constant column


def test_post_launch_spend_has_realistic_variation():
    df, _ = simulate_two_region_launch(launch_week=26)
    region_a = df[df["region"] == "A_launch"]
    post = region_a[region_a["week"] >= 26]["spend"]
    pre = region_a[region_a["week"] < 26]["spend"]
    assert (pre == 0).all()
    assert post.std() > 0  # week-to-week variation, not a flat step function
    assert post.min() > 0
