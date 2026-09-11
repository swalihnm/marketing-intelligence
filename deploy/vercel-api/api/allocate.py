from _shared import JSONHandler, load_posterior, optimize_allocation


class handler(JSONHandler):
    def do_POST(self):
        try:
            body = self._read_json_body()
        except Exception:
            self._error(400, "invalid JSON body")
            return

        budget = body.get("budget")
        if budget is None or not isinstance(budget, (int, float)) or budget <= 0:
            self._error(422, "budget must be a positive number")
            return

        max_shift = body.get("max_shift", 0.5)
        channel_caps = body.get("channel_caps")

        try:
            draws = load_posterior()
            result = optimize_allocation(draws, float(budget), max_shift_frac=float(max_shift), channel_caps=channel_caps)
        except Exception as e:
            self._error(500, f"allocation failed: {e}")
            return

        self._json(200, result)
