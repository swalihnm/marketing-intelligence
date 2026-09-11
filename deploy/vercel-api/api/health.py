from _shared import JSONHandler


class handler(JSONHandler):
    def do_GET(self):
        self._json(200, {"status": "ok"})
