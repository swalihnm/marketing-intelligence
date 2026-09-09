from customer.channel_synth import CHANNEL_CONFIG, assign_synthetic_channels


def test_probabilities_sum_to_one():
    total = sum(c["prob"] for c in CHANNEL_CONFIG.values())
    assert abs(total - 1.0) < 1e-9


def test_assigns_every_customer_a_known_channel():
    customer_ids = list(range(1000, 1050))
    df = assign_synthetic_channels(customer_ids)
    assert len(df) == len(customer_ids)
    assert set(df["channel"]) <= set(CHANNEL_CONFIG.keys())
    assert df["cac"].notna().all()


def test_deterministic_given_seed():
    customer_ids = list(range(1, 100))
    df1 = assign_synthetic_channels(customer_ids, seed=7)
    df2 = assign_synthetic_channels(customer_ids, seed=7)
    assert (df1["channel"] == df2["channel"]).all()
