from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from filmfinder import FilmFinderEngine

from app.api.router import api_router
from app.core.config import Settings, get_settings
from app.core.exceptions import APIError
from app.core.rate_limit import RateLimiter
from app.db.base import get_session_factory
from app.services.analytics_service import AnalyticsService
from app.services.email_service import build_email_sender
from app.services.metadata_service import TMDBMetadataService
from app.services.profile_service import ProfileService
from app.services.recommendation_batch_service import (
    RecommendationBatchService,
)
from app.services.recommendation_service import RecommendationService
from app.services.swipe_service import SwipeService
from app.services.taste_profile_service import TasteProfileService


logger = logging.getLogger(__name__)


def create_app(
    app_settings: Optional[Settings] = None,
    *,
    load_engine: bool = True,
    session_factory=None,
) -> FastAPI:
    settings = app_settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.engine = None
        app.state.engine_error = None
        app.state.recommendation_service = None
        app.state.profile_service = None
        app.state.swipe_service = None
        app.state.batch_service = None
        app.state.taste_profile_service = None
        app.state.taste_profile_error = None
        app.state.email_sender = build_email_sender(settings)
        app.state.rate_limiter = RateLimiter()
        app.state.analytics_service = AnalyticsService()
        if load_engine:
            app.state.metadata = TMDBMetadataService.load(
                settings.tmdb_metadata_file
            )
            try:
                engine = FilmFinderEngine.load(settings.artifacts_path)
                app.state.engine = engine
                app.state.recommendation_service = RecommendationService(
                    engine,
                    app.state.metadata,
                    settings,
                )
                app.state.profile_service = ProfileService(engine)
                app.state.swipe_service = SwipeService(
                    engine,
                    app.state.metadata,
                )
                app.state.batch_service = RecommendationBatchService(
                    session_factory or get_session_factory(),
                    app.state.recommendation_service,
                    app.state.profile_service,
                    app.state.swipe_service,
                )
            except Exception as error:
                app.state.engine_error = str(error)
                logger.exception("Recommendation engine could not be loaded.")
            if app.state.engine is not None:
                try:
                    app.state.taste_profile_service = (
                        TasteProfileService.load(
                            app.state.engine,
                            app.state.metadata,
                            settings.taste_profile_file,
                        )
                    )
                except Exception as error:
                    app.state.taste_profile_error = str(error)
                    logger.exception(
                        "Quick Taste Profile could not be loaded."
                    )
        else:
            app.state.metadata = TMDBMetadataService({})
        try:
            yield
        finally:
            close_batch_service = getattr(
                app.state.batch_service, "close", None
            )
            if close_batch_service is not None:
                close_batch_service()

    app = FastAPI(
        title="FilmFinder API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ):
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Invalid request parameters.",
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(
        request: Request, exc: StarletteHTTPException
    ):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": "http_error",
                    "message": str(exc.detail),
                }
            },
        )

    app.include_router(api_router)
    return app


app = create_app()
