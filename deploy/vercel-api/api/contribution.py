import numpy as np

from _shared import JSONHandler, geometric_adstock, hill, load_history, load_posterior


class handler(JSONHandler):
    def do_GET(self):
        params = self._query_params()
        start, end = params.get("start"), params.get("end")

        history = load_history()
        dates = history["date"]

        mask = [True] * len(dates)
        if start:
            mask = [m and d >= start for m, d in zip(mask, dates)]
        if end:
            mask = [m and d <= end for m, d in zip(mask, dates)]
        idx = [i for i, m in enumerate(mask) if m]

        if not idx:
            self._error(400, "no data in the requested date range")
            return

        draws = load_posterior()
        result = {
            "date": [dates[i] for i in idx],
            "revenue": [history["revenue"][i] for i in idx],
        }
        for i, c in enumerate(draws["channels"]):
            alpha_mean = float(draws["alpha"][:, i].mean())
            s_mean = float(draws["s"][:, i].mean())
            k_mean = float(draws["k"][:, i].mean())
            beta_mean = float(draws["beta"][:, i].mean())
            spend = np.array([history[f"spend_{c}"][j] for j in idx])
            adstocked = geometric_adstock(spend, alpha_mean, max_lag=8)
            result[f"contribution_{c}"] = (beta_mean * hill(adstocked, s_mean, k_mean)).tolist()

        self._json(200, result)
