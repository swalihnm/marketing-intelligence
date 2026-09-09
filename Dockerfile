# Serves the FastAPI app (app/main.py). NOT built or run in this
# environment — Docker isn't available here (confirmed: `docker` not on
# PATH) — so this is written to standard patterns but unverified beyond a
# manual read-through. See docs/06-assumptions-and-limitations.md.
#
# Build:  docker build -t mmm-api .
# Run:    docker run -p 8000:8000 mmm-api
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# the API loads outputs/reports/idata_recovery.nc at first request (see
# app/main.py::_load_state) — it must exist in the image. Generate it
# before building (`python -m simulate.recovery_study`) or mount it as a
# volume; it is gitignored (~11MB) so a fresh checkout won't have it.

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
