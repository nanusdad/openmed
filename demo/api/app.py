"""FastAPI surface for the local NER and de-identification demo.

Paste requests are exactly three SDK wrappers. Catalog routes never accept
clinical text. Nothing on this app logs request or response bodies.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.routing import Route

from demo.api.config import DemoConfig, assert_loopback_bind
from demo.api.errors import DemoError
from demo.api.session import InferenceSession

LOGGER = logging.getLogger("openmed.demo")

INFERENCE_ROUTES = ("/analyze", "/pii/extract", "/pii/deidentify")
CATALOG_ROUTES = ("/models", "/models/pull", "/models/active")


class AnalyzeRequest(BaseModel):
    """Clinical NER request. Extra fields are rejected."""

    model_config = ConfigDict(extra="forbid")

    text: str
    model_name: str | None = None
    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class ExtractRequest(BaseModel):
    """PII extraction request. Extra fields are rejected."""

    model_config = ConfigDict(extra="forbid")

    text: str
    model_name: str | None = None
    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    lang: str = "en"


class DeidentifyRequest(BaseModel):
    """De-identification request. Only mask and replace are exposed."""

    model_config = ConfigDict(extra="forbid")

    text: str
    method: Literal["mask", "replace"]
    model_name: str | None = None
    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    lang: str = "en"


class PullRequest(BaseModel):
    """Local cache download. Clinical text is not accepted."""

    model_config = ConfigDict(extra="forbid")

    model_name: str


class ActiveModelsRequest(BaseModel):
    """Active NER and PII model selection. Clinical text is not accepted."""

    model_config = ConfigDict(extra="forbid")

    ner_model: str | None = None
    pii_model: str | None = None


def create_app(config: DemoConfig, session: InferenceSession) -> FastAPI:
    """Build the loopback demo app around an existing session."""

    assert_loopback_bind(config.api_host)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.session.start()
        try:
            yield
        finally:
            app.state.session.close()

    app = FastAPI(
        title="OpenMed local demo",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        debug=False,
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.session = session
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[config.cors_origin],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
        max_age=600,
    )

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            LOGGER.warning(
                "demo request failed method=%s path=%s",
                request.method,
                request.url.path,
            )
            raise
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        LOGGER.info(
            "demo request method=%s path=%s status=%s elapsed_ms=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.exception_handler(DemoError)
    async def demo_error(_request: Request, exc: DemoError) -> JSONResponse:
        LOGGER.info("demo error code=%s", exc.code)
        return JSONResponse(status_code=exc.status_code, content={"error": exc.code})

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        _request: Request, _exc: RequestValidationError
    ) -> JSONResponse:
        LOGGER.info("demo error code=invalid_request")
        return JSONResponse(status_code=422, content={"error": "invalid_request"})

    @app.exception_handler(Exception)
    async def unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
        LOGGER.warning(
            "demo error code=inference_failed type=%s", exc.__class__.__name__
        )
        return JSONResponse(status_code=500, content={"error": "inference_failed"})

    @app.post("/analyze")
    def analyze(body: AnalyzeRequest) -> JSONResponse:
        payload = session.analyze(
            body.text,
            model_name=body.model_name,
            confidence_threshold=body.confidence_threshold,
        )
        return JSONResponse(payload)

    @app.post("/pii/extract")
    def extract_pii(body: ExtractRequest) -> JSONResponse:
        payload = session.extract_pii(
            body.text,
            model_name=body.model_name,
            confidence_threshold=body.confidence_threshold,
            lang=body.lang,
        )
        return JSONResponse(payload)

    @app.post("/pii/deidentify")
    def deidentify(body: DeidentifyRequest) -> JSONResponse:
        payload = session.deidentify(
            body.text,
            method=body.method,
            model_name=body.model_name,
            confidence_threshold=body.confidence_threshold,
            lang=body.lang,
        )
        return JSONResponse(payload)

    @app.get("/models")
    def models() -> JSONResponse:
        return JSONResponse(session.catalog())

    @app.post("/models/pull")
    def pull(body: PullRequest) -> JSONResponse:
        return JSONResponse(session.pull(body.model_name))

    @app.post("/models/active")
    def active(body: ActiveModelsRequest) -> JSONResponse:
        return JSONResponse(
            session.select(ner_model=body.ner_model, pii_model=body.pii_model)
        )

    return app


def route_paths(app: FastAPI) -> list[str]:
    """Return the HTTP paths registered on the demo app."""

    return sorted(route.path for route in app.routes if isinstance(route, Route))
