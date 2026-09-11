import numpy as np

from _shared import JSONHandler, hill, load_posterior


class handler(JSONHandler):
    def do_GET(self):
        params = self._query_params()
        channel = params.get("channel")
        if not channel:
            self._error(422, "query param 'channel' is required")
            return

        draws = load_posterior()
        if channel not in draws["channels"]:
            self._error(404, f"unknown channel '{channel}'. Known: {draws['channels']}")
            return

        n_points = int(params.get("n_points", 50))
        max_multiplier = float(params.get("max_multiplier", 3.0))

        idx = draws["channels"].index(channel)
        max_x = draws["current_spend"][channel] * max_multiplier
        spend_range = np.linspace(1.0, max_x, n_points)

        alpha, s, k, beta = draws["alpha"][:, idx], draws["s"][:, idx], draws["k"][:, idx], draws["beta"][:, idx]
        adstock_ss = spend_range[None, :] / (1 - alpha[:, None])
        curves = beta[:, None] * hill(adstock_ss, s[:, None], k[:, None])

        mean = curves.mean(axis=0)
        lo = np.percentile(curves, 5, axis=0)
        hi = np.percentile(curves, 95, axis=0)

        points = [
            {"spend": float(sp), "mean": float(m), "hdi_lower": float(l), "hdi_upper": float(h)}
            for sp, m, l, h in zip(spend_range, mean, lo, hi)
        ]
        self._json(200, points)
