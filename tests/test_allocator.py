import numpy as np

from allocate.optimizer import expected_response, optimize_allocation, response_per_draw


def _toy_draws(n_draws=50, n_channels=2, seed=0):
    rng = np.random.default_rng(seed)
    return dict(
        beta=rng.uniform(1e5, 2e5, size=(n_draws, n_channels)),
        alpha=rng.uniform(0.2, 0.5, size=(n_draws, n_channels)),
        s=rng.uniform(1.5, 2.5, size=(n_draws, n_channels)),
        k=rng.uniform(2e4, 4e4, size=(n_draws, n_channels)),
        channels=["a", "b"],
    )


def test_response_increases_with_more_spend():
    draws = _toy_draws()
    low = expected_response(np.array([1_000.0, 1_000.0]), draws)
    high = expected_response(np.array([50_000.0, 50_000.0]), draws)
    assert high > low


def test_response_per_draw_shape():
    draws = _toy_draws(n_draws=50, n_channels=2)
    out = response_per_draw(np.array([10_000.0, 10_000.0]), draws)
    assert out.shape == (50,)
    assert (out >= 0).all()


def test_optimizer_respects_budget_constraint():
    draws = _toy_draws()
    current_spend = {"a": 20_000.0, "b": 10_000.0}
    total_budget = 30_000.0
    result = optimize_allocation(draws, current_spend, total_budget, max_shift_frac=1.0)
    spend_sum = sum(result["optimal_spend"].values())
    assert spend_sum <= total_budget + 1.0  # small numerical tolerance


def test_optimizer_respects_max_shift_bounds():
    draws = _toy_draws()
    current_spend = {"a": 20_000.0, "b": 10_000.0}
    total_budget = 60_000.0  # generous budget so max_shift is the binding constraint
    result = optimize_allocation(draws, current_spend, total_budget, max_shift_frac=0.2)
    for c, spend in result["optimal_spend"].items():
        lo = current_spend[c] * 0.8
        hi = current_spend[c] * 1.2
        assert lo - 1.0 <= spend <= hi + 1.0


def test_optimizer_improves_on_or_matches_current_allocation():
    draws = _toy_draws()
    current_spend = {"a": 5_000.0, "b": 45_000.0}  # deliberately lopsided
    total_budget = sum(current_spend.values())
    result = optimize_allocation(draws, current_spend, total_budget, max_shift_frac=1.0)
    assert result["expected_response_optimal"] >= result["expected_response_current"] - 1e-6
