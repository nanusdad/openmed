"""Process entrypoint for the loopback demo API."""

from __future__ import annotations

import logging
from typing import Any, Callable

from demo.api.app import create_app
from demo.api.config import (
    apply_process_environment,
    assert_loopback_bind,
    load_demo_config,
)
from demo.api.sdk import RealSdk
from demo.api.session import InferenceSession

LOGGER = logging.getLogger("openmed.demo")


def build_runtime() -> tuple[Any, Any]:
    """Load env config, pin caches, and construct the shared session."""

    config = load_demo_config()
    assert_loopback_bind(config.api_host)
    apply_process_environment(config)
    if config.cache_dir != config.hf_hub_cache:
        LOGGER.info(
            "demo cache directories differ; model pull and inference use "
            "OPENMED_DEMO_CACHE_DIR"
        )
    session = InferenceSession(config, RealSdk())
    return config, session


def serve(runner: Callable[..., Any] | None = None) -> None:
    """Bind the demo API. Non-loopback hosts fail before the socket opens."""

    config, session = build_runtime()
    app = create_app(config, session)
    if runner is None:
        import uvicorn

        runner = uvicorn.run
    runner(
        app,
        host=config.api_host,
        port=config.api_port,
        access_log=False,
        log_level="info",
        server_header=False,
        date_header=False,
    )


def main() -> None:
    """Start the local demo API."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    serve()
