"""Safe demo errors.

Error codes are stable tokens. They never include request text.
"""

from __future__ import annotations


class ConfigError(Exception):
    """Raised when demo configuration is missing or not loopback-safe."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class DemoError(Exception):
    """Raised for a request the demo can refuse without echoing PHI."""

    def __init__(self, code: str, status_code: int = 400) -> None:
        super().__init__(code)
        self.code = code
        self.status_code = status_code
