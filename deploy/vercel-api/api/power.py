from _shared import JSONHandler, minimum_detectable_effect, required_duration_weeks


class handler(JSONHandler):
    def do_POST(self):
        try:
            body = self._read_json_body()
        except Exception:
            self._error(400, "invalid JSON body")
            return

        tracking_error_std = body.get("tracking_error_std")
        baseline = body.get("baseline_weekly_revenue")
        if not tracking_error_std or tracking_error_std <= 0:
            self._error(422, "tracking_error_std must be a positive number")
            return
        if not baseline or baseline <= 0:
            self._error(422, "baseline_weekly_revenue must be a positive number")
            return

        n_weeks = body.get("n_weeks")
        target_mde_pct = body.get("target_mde_pct")
        alpha = body.get("alpha", 0.05)
        power = body.get("power", 0.8)

        if n_weeks is None and target_mde_pct is None:
            self._error(400, "provide n_weeks and/or target_mde_pct")
            return

        result = {}
        if n_weeks is not None:
            mde_abs = minimum_detectable_effect(tracking_error_std, n_weeks, alpha, power)
            result["mde_abs"] = mde_abs
            result["mde_pct"] = mde_abs / baseline
        if target_mde_pct is not None:
            target_abs = target_mde_pct * baseline
            result["required_duration_weeks"] = required_duration_weeks(target_abs, tracking_error_std, alpha, power)

        self._json(200, result)
