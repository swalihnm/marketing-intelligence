import numpy as np

from experiment.power import minimum_detectable_effect, required_duration_weeks


def test_mde_decreases_with_more_weeks():
    mde_2 = minimum_detectable_effect(tracking_error_std=100, n_weeks=2)
    mde_20 = minimum_detectable_effect(tracking_error_std=100, n_weeks=20)
    assert mde_20 < mde_2


def test_mde_scales_with_tracking_error():
    mde_low_noise = minimum_detectable_effect(tracking_error_std=50, n_weeks=8)
    mde_high_noise = minimum_detectable_effect(tracking_error_std=500, n_weeks=8)
    assert mde_high_noise > mde_low_noise


def test_required_duration_is_inverse_of_mde():
    tracking_error_std = 200.0
    n_weeks = 10
    mde = minimum_detectable_effect(tracking_error_std, n_weeks)
    recovered_weeks = required_duration_weeks(mde, tracking_error_std)
    assert np.isclose(recovered_weeks, n_weeks, rtol=1e-6)


def test_higher_power_requires_more_weeks():
    weeks_80 = required_duration_weeks(target_mde_abs=100, tracking_error_std=200, power=0.8)
    weeks_95 = required_duration_weeks(target_mde_abs=100, tracking_error_std=200, power=0.95)
    assert weeks_95 > weeks_80
