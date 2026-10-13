import numpy as np
import pandas as pd

from mmm.model import build_mmm


def _toy_df(n=52, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "t": np.arange(n),
        "revenue": rng.uniform(1e5, 2e5, n),
        "price": 100.0 + rng.normal(0, 1, n),
        "spend_x": rng.uniform(1e3, 1e4, n),
    })


def test_build_mmm_without_extra_covariates_is_unchanged():
    df = _toy_df()
    model = build_mmm(df, channels=["x"], max_lag=4)
    names = {rv.name for rv in model.free_RVs}
    assert not any(n.startswith("coef_") for n in names)


def test_build_mmm_with_extra_covariates_adds_coefficients():
    df = _toy_df()
    extra = {"sentiment": np.random.default_rng(1).normal(0, 1, len(df))}
    model = build_mmm(df, channels=["x"], max_lag=4, extra_covariates=extra)
    names = {rv.name for rv in model.free_RVs}
    assert "coef_sentiment" in names


def test_build_mmm_with_multiple_extra_covariates():
    df = _toy_df()
    rng = np.random.default_rng(2)
    extra = {
        "sentiment": rng.normal(0, 1, len(df)),
        "temperature": rng.normal(15, 5, len(df)),
    }
    model = build_mmm(df, channels=["x"], max_lag=4, extra_covariates=extra)
    names = {rv.name for rv in model.free_RVs}
    assert "coef_sentiment" in names
    assert "coef_temperature" in names
