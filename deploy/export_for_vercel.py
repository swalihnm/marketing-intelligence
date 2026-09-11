"""Exports the minimal data the trimmed Vercel API needs — posterior draws,
current spend, and the historical simulated series — as plain JSON, so the
deployed functions depend on numpy/scipy only (no PyMC/PyTensor/numba/arviz
at runtime; see deploy/vercel-api/README.md for why).

Run locally (not deployed): python -m deploy.export_for_vercel
"""
from __future__ import annotations

import json
from pathlib import Path

from allocate.optimizer import IDATA_PATH, load_posterior_draws
from simulate.dgp import sample_true_params, simulate as dgp_simulate

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "deploy" / "vercel-api" / "data"


def _round(nested_list, ndigits):
    if isinstance(nested_list, list):
        return [_round(x, ndigits) for x in nested_list]
    return round(nested_list, ndigits)


def export(n_draws: int = 80, round_digits: int = 2):
    """n_draws is deliberately much smaller than the 300 used for local
    analysis (allocate/optimizer.py) — this data is bundled directly into
    the deployed function's source, and needs to stay small enough to pass
    through as inline file content when deploying (not read from a
    filesystem path at deploy time). 80 draws is still enough to produce
    sensible 90% intervals for a demo; round_digits=2 further shrinks the
    file without materially changing any result at this scale."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    true = sample_true_params(n_weeks=156, spend_correlation=0.5, seed=42)
    df = dgp_simulate(true, seed=42 + 1000)
    channels = list(true.channels.keys())
    current_spend = {c: round(float(df[f"spend_{c}"].mean()), round_digits) for c in channels}

    draws = load_posterior_draws(IDATA_PATH, channels, n_draws=n_draws)
    posterior_payload = {
        "channels": channels,
        "beta": _round(draws["beta"].tolist(), round_digits),
        "alpha": _round(draws["alpha"].tolist(), 4),  # alpha in [0,1); needs more precision
        "s": _round(draws["s"].tolist(), 4),
        "k": _round(draws["k"].tolist(), round_digits),
        "current_spend": current_spend,
    }
    (OUT_DIR / "posterior_draws.json").write_text(json.dumps(posterior_payload, separators=(",", ":")), encoding="utf-8")

    history_payload = {
        "date": df["date"].dt.strftime("%Y-%m-%d").tolist(),
        "revenue": _round(df["revenue"].tolist(), round_digits),
        **{f"spend_{c}": _round(df[f"spend_{c}"].tolist(), round_digits) for c in channels},
    }
    (OUT_DIR / "history.json").write_text(json.dumps(history_payload, separators=(",", ":")), encoding="utf-8")

    print(f"wrote posterior_draws.json ({(OUT_DIR / 'posterior_draws.json').stat().st_size} bytes)")
    print(f"wrote history.json ({(OUT_DIR / 'history.json').stat().st_size} bytes)")


if __name__ == "__main__":
    export()
