from fastapi import APIRouter

from app.api.routes import (
    compressions,
    datasets,
    inference,
    items,
    login,
    models,
    private,
    users,
    utils,
)
from app.core.config import settings

api_router = APIRouter()
api_router.include_router(login.router)
api_router.include_router(users.router)
api_router.include_router(utils.router)
api_router.include_router(items.router)
api_router.include_router(models.router)
api_router.include_router(datasets.router)
api_router.include_router(compressions.router)
api_router.include_router(inference.router)


if settings.FASTAPI_ENV == "development":
    api_router.include_router(private.router)
