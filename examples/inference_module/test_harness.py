"""Test harness: mount the drop-in compression proxy router and serve it."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from fastapi import FastAPI  # noqa: E402

from api.compression_proxy import router  # noqa: E402

app = FastAPI(title="inference-module compression proxy (test harness)")
app.include_router(router)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
