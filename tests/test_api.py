import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_allocate_returns_valid_allocation():
    resp = client.post("/allocate", json={"budget": 150000, "max_shift": 0.5})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["optimal_spend"].keys()) == set(body["current_spend"].keys())
    assert sum(body["optimal_spend"].values()) <= 150000 + 1.0
    assert body["expected_gain_hdi90_lower"] <= body["expected_gain_mean"] <= body["expected_gain_hdi90_upper"]


def test_allocate_rejects_nonpositive_budget():
    resp = client.post("/allocate", json={"budget": -100})
    assert resp.status_code == 422


def test_response_curve_known_channel():
    resp = client.get("/response-curve/tv", params={"n_points": 10})
    assert resp.status_code == 200
    points = resp.json()
    assert len(points) == 10
    # response should be monotonically non-decreasing in spend (Hill saturation is increasing)
    means = [p["mean"] for p in points]
    assert all(b >= a - 1e-6 for a, b in zip(means, means[1:]))


def test_response_curve_unknown_channel_404():
    resp = client.get("/response-curve/not_a_real_channel")
    assert resp.status_code == 404


def test_contribution_returns_series():
    resp = client.get("/contribution", params={"start": "2021-01-01", "end": "2021-06-01"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["date"]) == len(body["revenue"])
    assert "contribution_tv" in body
    assert len(body["contribution_tv"]) == len(body["date"])


def test_contribution_empty_range_400():
    resp = client.get("/contribution", params={"start": "2099-01-01", "end": "2099-06-01"})
    assert resp.status_code == 400


def test_experiment_power_mde():
    resp = client.post("/experiment/power", json={
        "tracking_error_std": 800, "baseline_weekly_revenue": 15000, "n_weeks": 8,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["mde_abs"] > 0
    assert body["mde_pct"] == pytest.approx(body["mde_abs"] / 15000)


def test_experiment_power_duration():
    resp = client.post("/experiment/power", json={
        "tracking_error_std": 800, "baseline_weekly_revenue": 15000, "target_mde_pct": 0.1,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["required_duration_weeks"] > 0


def test_experiment_power_requires_one_of_n_weeks_or_target():
    resp = client.post("/experiment/power", json={"tracking_error_std": 800, "baseline_weekly_revenue": 15000})
    assert resp.status_code == 400
