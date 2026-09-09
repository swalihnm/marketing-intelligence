import numpy as np

from transforms.adstock import geometric_adstock_numpy
from transforms.saturation import hill


def test_geometric_adstock_zero_decay_is_identity():
    spend = np.array([10.0, 20.0, 30.0, 0.0, 5.0])
    out = geometric_adstock_numpy(spend, alpha=0.0, max_lag=8)
    assert np.allclose(out, spend)


def test_geometric_adstock_known_value():
    spend = np.array([100.0, 0.0, 0.0, 0.0])
    out = geometric_adstock_numpy(spend, alpha=0.5, max_lag=8)
    # adstock_t = sum_l alpha^l * spend_{t-l}
    expected = np.array([100.0, 50.0, 25.0, 12.5])
    assert np.allclose(out, expected)


def test_geometric_adstock_nonnegative_and_bounded():
    rng = np.random.default_rng(0)
    spend = rng.uniform(0, 1000, size=50)
    out = geometric_adstock_numpy(spend, alpha=0.7, max_lag=8)
    assert (out >= 0).all()
    # bounded by the theoretical max: spend_max / (1 - alpha)
    assert out.max() <= spend.max() / (1 - 0.7) + 1e-6


def test_hill_at_half_saturation_point():
    # by construction, f(k) = k^s / (k^s + k^s) = 0.5 for any s
    assert np.isclose(hill(x=10.0, s=2.0, k=10.0), 0.5)
    assert np.isclose(hill(x=10.0, s=1.5, k=10.0), 0.5)


def test_hill_is_increasing_and_bounded_zero_one():
    xs = np.linspace(0.01, 100, 50)
    ys = hill(xs, s=2.0, k=20.0)
    assert (ys >= 0).all() and (ys <= 1).all()
    assert np.all(np.diff(ys) >= 0)
