import warnings
from typing import Literal, Self

from pydantic import (
    EmailStr,
    HttpUrl,
    PostgresDsn,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Use top level .env file (one level above ./backend/)
        env_file="../.env",
        env_ignore_empty=True,
        extra="ignore",
    )
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str
    # 60 minutes * 24 hours * 8 days = 8 days
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
    FRONTEND_HOST: str = "http://localhost:5173"
    FASTAPI_ENV: Literal["development"] | None = None

    PROJECT_NAME: str
    SENTRY_DSN: HttpUrl | None = None
    DATABASE_URL: PostgresDsn

    # -------------------------------------------------------------------------
    # Compute-backend configuration (GPU migration prep).
    #
    #   COMPUTE_BACKEND=cpu (default) | gpu
    #   GPU_DEVICE_IDS=0,1              # devices this deployment may schedule on
    #   CUDA_VISIBLE_DEVICES=...        # set by the orchestrator/Docker Compose;
    #                                   # this process only *reads* the value.
    # -------------------------------------------------------------------------
    REDIS_URL: str = "redis://localhost:6379/0"
    COMPUTE_BACKEND: str = "cpu"
    GPU_DEVICE_IDS: str = "0"
    # CUDA_VISIBLE_DEVICES is handled by the runtime environment; read for
    # logging/introspection only (see app.services.base.resolve_devices()).
    CUDA_VISIBLE_DEVICES: str | None = None

    MODEL_STORAGE_PATH: str = "./models-storage"
    # Maximum size (in bytes) allowed for a single downloaded/pulled model file
    MAX_MODEL_DOWNLOAD_BYTES: int = 10 * 1024 * 1024 * 1024  # 10 GB
    # Per-user storage quota (models + datasets). Set to 0 to disable.
    USER_STORAGE_QUOTA_BYTES: int = 50 * 1024 * 1024 * 1024  # 50 GB
    # Read-only roots allowed for "import from server-local directory".
    # Multiple paths separated by ';'. Containers mount host model folders here
    # (compose.cpu.yml: /home/zhangxuefei/Model -> /models-local).
    LOCAL_MODEL_DIRS: str = "/models-local"

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def _use_psycopg_driver(cls, value: str | PostgresDsn) -> str:
        database_url = str(value)
        for scheme in ("postgres://", "postgresql://"):
            if database_url.startswith(scheme):
                return database_url.replace(scheme, "postgresql+psycopg://", 1)
        return database_url

    SMTP_TLS: bool = True
    SMTP_SSL: bool = False
    SMTP_PORT: int = 587
    SMTP_HOST: str | None = None
    SMTP_USER: str | None = None
    SMTP_PASSWORD: str | None = None
    EMAILS_FROM_EMAIL: EmailStr | None = None
    EMAILS_FROM_NAME: str | None = None

    @model_validator(mode="after")
    def _set_default_emails_from(self) -> Self:
        if not self.EMAILS_FROM_NAME:
            self.EMAILS_FROM_NAME = self.PROJECT_NAME
        return self

    EMAIL_RESET_TOKEN_EXPIRE_HOURS: int = 48

    @computed_field  # type: ignore[prop-decorator]
    @property
    def emails_enabled(self) -> bool:
        return bool(self.SMTP_HOST and self.EMAILS_FROM_EMAIL)

    EMAIL_TEST_USER: EmailStr = "test@example.com"
    FIRST_SUPERUSER: EmailStr
    FIRST_SUPERUSER_PASSWORD: str

    def _check_default_secret(self, var_name: str, value: str | None) -> None:
        if value == "changethis":
            message = (
                f'The value of {var_name} is "changethis", '
                "for security, please change it, at least for deployments."
            )
            if self.FASTAPI_ENV == "development":
                warnings.warn(message, stacklevel=1)
            else:
                raise ValueError(message)

    @model_validator(mode="after")
    def _enforce_non_default_secrets(self) -> Self:
        self._check_default_secret("SECRET_KEY", self.SECRET_KEY)
        for host in self.DATABASE_URL.hosts():
            self._check_default_secret("DATABASE_URL password", host["password"])
        self._check_default_secret(
            "FIRST_SUPERUSER_PASSWORD", self.FIRST_SUPERUSER_PASSWORD
        )

        return self


settings = Settings()  # type: ignore
