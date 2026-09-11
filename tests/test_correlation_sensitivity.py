from simulate.correlation_sensitivity import CORRELATIONS


def test_correlation_levels_span_expected_range():
    assert CORRELATIONS[0] == 0.3
    assert CORRELATIONS[-1] == 0.9
    assert CORRELATIONS == sorted(CORRELATIONS)
    assert all(0 <= c <= 1 for c in CORRELATIONS)
