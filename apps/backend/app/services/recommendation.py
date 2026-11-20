from __future__ import annotations

from functools import lru_cache

from app.core.config import artifacts_dir


@lru_cache(maxsize=1)
def get_engine():
    from filmfinder import FilmFinderEngine

    return FilmFinderEngine.load(artifacts_dir())
