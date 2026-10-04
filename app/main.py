"""FastAPI application entrypoint.

Phase 0 slice: app wiring, RFC 7807 error format (B5) and the liveness and
dependency check at GET /health -- the one public endpoint in the API inventory.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import text
from starlette.exceptions import HTTPException as StarletteHTTPException
from app.api.accumulator import router as accumulator_router
from app.api.fixtures import router as fixtures_router
from app.api.notifications import router as notifications_router
from app.api.performance import router as performance_router
from app.api.picks import router as picks_router
from app import __version__
from app.config import settings
from app.db import SessionLocal, dispose_engine
from app.security import API_KEY_HEADER, RateLimited, Unauthorised, rate_limit, require_api_key

logger = logging.getLogger(__name__)

PROBLEM_JSON = "application/problem+json"


class DependencyStatus(BaseModel):
    name: str
    status: Literal["ok", "error"]
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    dependencies: list[DependencyStatus]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("starting prediction service")
    yield
    await dispose_engine()
    logger.info("stopped prediction service")


app = FastAPI(
    title="Football Micro-Event Prediction Platform API",
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# /health stays open so a monitor can reach it; everything else needs the key.
guarded = [Depends(require_api_key), Depends(rate_limit)]

app.include_router(fixtures_router, prefix="/api/v1", dependencies=guarded)
app.include_router(picks_router, prefix="/api/v1", dependencies=guarded)
app.include_router(performance_router, prefix="/api/v1", dependencies=guarded)
app.include_router(accumulator_router, prefix="/api/v1", dependencies=guarded)
app.include_router(notifications_router, prefix="/api/v1", dependencies=guarded)


def problem(
    status_code: int,
    title: str,
    detail: str,
    instance: str | None = None,
    **extra: Any,
) -> JSONResponse:
    """RFC 7807 problem details response (B5)."""
    body: dict[str, Any] = {
        "type": "about:blank",
        "title": title,
        "status": status_code,
        "detail": detail,
    }
    if instance:
        body["instance"] = instance
    body.update(extra)
    return JSONResponse(status_code=status_code, content=body, media_type=PROBLEM_JSON)


@app.exception_handler(Unauthorised)
async def unauthorised_handler(request: Request, exc: Unauthorised) -> JSONResponse:
    response = problem(
        status_code=401,
        title="Unauthorised",
        detail=exc.detail,
        instance=str(request.url.path),
    )
    response.headers["WWW-Authenticate"] = API_KEY_HEADER
    return response


@app.exception_handler(RateLimited)
async def rate_limited_handler(request: Request, exc: RateLimited) -> JSONResponse:
    response = problem(
        status_code=429,
        title="Too many requests",
        detail=f"Rate limit reached; retry in {exc.retry_after} seconds.",
        instance=str(request.url.path),
    )
    response.headers["Retry-After"] = str(exc.retry_after)
    return response


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    return problem(
        status_code=exc.status_code,
        title=str(exc.detail),
        detail=str(exc.detail),
        instance=str(request.url.path),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return problem(
        status_code=422,
        title="Validation error",
        detail="The request payload failed validation.",
        instance=str(request.url.path),
        errors=exc.errors(),
    )


async def check_database() -> DependencyStatus:
    try:
        async with SessionLocal() as session:
            await session.execute(text("SELECT 1"))
    except Exception as exc:
        logger.warning("database health check failed: %s", exc)
        return DependencyStatus(name="database", status="error", detail=str(exc))
    return DependencyStatus(name="database", status="ok")


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> JSONResponse:
    dependencies = [await check_database()]
    healthy = all(dep.status == "ok" for dep in dependencies)
    payload = HealthResponse(
        status="ok" if healthy else "degraded",
        version=__version__,
        environment=settings.environment,
        dependencies=dependencies,
    )
    return JSONResponse(
        status_code=200 if healthy else 503,
        content=payload.model_dump(),
    )
