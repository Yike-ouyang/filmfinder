from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

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

    @property
    def artifacts_path(self) -> Path:
        return _resolve_path(self.filmfinder_artifacts_dir)

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
