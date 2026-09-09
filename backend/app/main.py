from pathlib import Path

import sentry_sdk
from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.middleware.cors import CORSMiddleware

from app.api.main import api_router
from app.core.config import settings

FRONTEND_DIR = Path(__file__).parent / "frontend"


def custom_generate_unique_id(route: APIRoute) -> str:
    return f"{route.tags[0]}-{route.name}"


if settings.SENTRY_DSN and settings.FASTAPI_ENV != "development":
    sentry_sdk.init(dsn=str(settings.SENTRY_DSN), enable_tracing=True)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    generate_unique_id_function=custom_generate_unique_id,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.FRONTEND_HOST,
        # The FastAPI-served frontend build can be reached via either hostname
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8001",
        "http://127.0.0.1:8001",
        # Vite dev server (http://localhost:5173) calling the API cross-origin
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)

# SPA serving: static assets + index.html fallback (works on any Host/IP).
# API routes above are matched first; unmatched /api/* keeps returning JSON 404.
if (FRONTEND_DIR / "index.html").is_file():
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    assets_dir = FRONTEND_DIR / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    _frontend_root = FRONTEND_DIR.resolve()

    @app.get("/{path:path}", tags=["spa"], include_in_schema=False)
    async def _spa(path: str) -> FileResponse:
        if path.startswith(settings.API_V1_STR.lstrip("/")):
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="Not Found")
        candidate = (FRONTEND_DIR / path).resolve()
        try:
            candidate.relative_to(_frontend_root)
        except ValueError:
            candidate = FRONTEND_DIR
        if candidate.is_file() and candidate != FRONTEND_DIR:
            return FileResponse(candidate)
        # index.html must never be cached long: assets are content-hashed and
        # immutable, so only the HTML needs revalidation on every visit.
        return FileResponse(
            FRONTEND_DIR / "index.html",
            headers={"Cache-Control": "no-cache"},
        )
