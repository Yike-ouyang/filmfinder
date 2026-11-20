from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from filmfinder import FilmFinderEngine

from app.api.router import api_router
from app.core.config import get_settings
from app.core.exceptions import APIError
from app.services.metadata_service import TMDBMetadataService
from app.services.recommendation_service import RecommendationService


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.engine = None
    app.state.engine_error = None
    app.state.recommendation_service = None
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
    except Exception as error:
        app.state.engine_error = str(error)
        logger.exception("Recommendation engine could not be loaded.")
    yield


app = FastAPI(
    title="FilmFinder API",
    version="0.1.0",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
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
async def http_error_handler(request: Request, exc: StarletteHTTPException):
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
