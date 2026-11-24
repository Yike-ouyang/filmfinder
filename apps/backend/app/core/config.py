from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL

BACKEND_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: str = "development"
    filmfinder_artifacts_dir: str = (
        "../../packages/recommendation-engine/artifacts"
    )
    tmdb_metadata_path: str = (
        "../../packages/recommendation-engine/artifacts/"
        "tmdb_metadata.json"
    )
    cors_origins: str = "http://localhost:3000"
    max_upload_size_mb: int = 20
    max_uncompressed_size_mb: int = 100
    max_zip_entries: int = 64
    max_concurrent_recommendations: int = 1
    tmdb_api_key: str = ""
    tmdb_request_timeout: float = 20.0
    filmfinder_db_path: str = "var/filmfinder.db"
    sqlite_busy_timeout_ms: int = 5000
    app_base_url: str = "http://localhost:3000"
    email_backend: str = "console"
    email_from: str = "noreply@filmfinder.local"
    email_from_name: str = "FilmFinder"
    brevo_api_key: str = ""
    brevo_api_url: str = "https://api.brevo.com/v3/smtp/email"
    email_request_timeout: float = 10.0
    session_ttl_days: int = 30
    session_cookie_name: str = "ff_session"
    csrf_cookie_name: str = "ff_csrf"
    cookie_secure: bool = False
    require_email_verification: bool = False
    email_verification_ttl_hours: int = 24
    password_reset_ttl_minutes: int = 30
    password_min_length: int = 10
    password_max_length: int = 128
    argon2_memory_cost: int = 19456
    argon2_time_cost: int = 2
    argon2_parallelism: int = 1
    login_rate_limit_attempts: int = 5
    login_rate_limit_window_seconds: int = 900
    signup_rate_limit_attempts: int = 5
    signup_rate_limit_window_seconds: int = 3600
    email_rate_limit_attempts: int = 3
    email_rate_limit_window_seconds: int = 900
    taste_profile_path: str = "app/data/quick_taste_movies.json"
    taste_profile_rate_limit_attempts: int = 20
    taste_profile_rate_limit_window_seconds: int = 600
    analytics_retention_days: int = 365

    @property
    def artifacts_path(self) -> Path:
        return _resolve_path(self.filmfinder_artifacts_dir)

    @property
    def taste_profile_file(self) -> Path:
        return _resolve_path(self.taste_profile_path)

    @property
    def database_path(self) -> Path:
        return _resolve_path(self.filmfinder_db_path)

    @property
    def database_url(self) -> str:
        return URL.create(
            "sqlite+pysqlite", database=str(self.database_path)
        ).render_as_string(hide_password=False)

    @property
    def tmdb_metadata_file(self) -> Path:
        return _resolve_path(self.tmdb_metadata_path)

    @property
    def cors_origin_list(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_size_mb) * 1024 * 1024

    @property
    def max_uncompressed_bytes(self) -> int:
        return int(self.max_uncompressed_size_mb) * 1024 * 1024

def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = BACKEND_ROOT / path
    return path.resolve()

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
