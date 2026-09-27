"""Environment-only configuration for the local demo.

Ports, bind host, CORS origin, model ids, cache directories, and Hub offline
flags are read from the process environment and ``demo/.env``. The demo does
not substitute hardcoded defaults for those values.
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

from demo.api.errors import ConfigError

DEMO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_ENV_NAMES = (
    "OPENMED_DEMO_API_HOST",
    "OPENMED_DEMO_API_PORT",
    "OPENMED_DEMO_WEB_HOST",
    "OPENMED_DEMO_WEB_PORT",
    "OPENMED_DEMO_CORS_ORIGIN",
    "VITE_OPENMED_DEMO_API_ORIGIN",
    "OPENMED_DEMO_NER_MODEL",
    "OPENMED_DEMO_PII_MODEL",
    "OPENMED_DEMO_CACHE_DIR",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "OPENMED_OFFLINE",
    "OPENMED_DEMO_BACKEND",
)

_ALLOWED_BACKENDS = frozenset({"auto", "hf", "mlx"})
_TRUE_FLAGS = frozenset({"1", "true", "yes", "on"})
_FALSE_FLAGS = frozenset({"0", "false", "no", "off"})


@dataclass(frozen=True)
class DemoConfig:
    """Resolved local-demo settings."""

    api_host: str
    api_port: int
    web_host: str
    web_port: int
    cors_origin: str
    api_origin: str
    ner_model: str
    pii_model: str
    cache_dir: str
    hf_home: str
    hf_hub_cache: str
    hf_hub_offline: bool
    transformers_offline: bool
    hf_datasets_offline: bool
    openmed_offline: bool
    backend: str

    @property
    def requests_offline(self) -> bool:
        """Return whether the operator asked the process to stay cache-only."""

        return (
            self.hf_hub_offline
            or self.transformers_offline
            or self.hf_datasets_offline
            or self.openmed_offline
        )


def parse_env_file(text: str) -> dict[str, str]:
    """Parse a dotenv file into names and values.

    Existing process environment wins over the file. Quotes around a value are
    removed. Comments and blank lines are ignored.
    """

    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def is_loopback_host(host: str) -> bool:
    """Return whether ``host`` is a loopback name or address."""

    candidate = host.strip().lower()
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    if "%" in candidate:
        candidate = candidate.split("%", 1)[0]
    if candidate == "localhost":
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def assert_loopback_bind(host: str) -> None:
    """Fail closed unless the API bind host is loopback."""

    if not is_loopback_host(host):
        raise ConfigError("bind_host_not_loopback")


def load_demo_config(
    *,
    environ: Mapping[str, str] | None = None,
    env_file: Path | None = None,
    read_default_env_file: bool = True,
) -> DemoConfig:
    """Load demo settings from the environment and optional dotenv file."""

    merged: dict[str, str] = {}
    file_path = env_file
    if file_path is None and read_default_env_file:
        candidate = DEMO_ROOT / ".env"
        file_path = candidate if candidate.is_file() else None
    if file_path is not None and file_path.is_file():
        merged.update(parse_env_file(file_path.read_text(encoding="utf-8")))

    source = os.environ if environ is None else environ
    for key in REQUIRED_ENV_NAMES:
        if key in source and source[key] != "":
            merged[key] = source[key]

    missing = [key for key in REQUIRED_ENV_NAMES if not merged.get(key, "").strip()]
    if missing:
        raise ConfigError("missing_env:" + ",".join(missing))

    api_host = _loopback_host(merged["OPENMED_DEMO_API_HOST"], "OPENMED_DEMO_API_HOST")
    web_host = _loopback_host(merged["OPENMED_DEMO_WEB_HOST"], "OPENMED_DEMO_WEB_HOST")
    api_port = _port(merged["OPENMED_DEMO_API_PORT"], "OPENMED_DEMO_API_PORT")
    web_port = _port(merged["OPENMED_DEMO_WEB_PORT"], "OPENMED_DEMO_WEB_PORT")
    cors_origin = _origin(
        merged["OPENMED_DEMO_CORS_ORIGIN"],
        "OPENMED_DEMO_CORS_ORIGIN",
        expected_port=web_port,
    )
    api_origin = _origin(
        merged["VITE_OPENMED_DEMO_API_ORIGIN"],
        "VITE_OPENMED_DEMO_API_ORIGIN",
        expected_port=api_port,
    )
    backend = merged["OPENMED_DEMO_BACKEND"].strip().lower()
    if backend not in _ALLOWED_BACKENDS:
        raise ConfigError("invalid_backend")

    return DemoConfig(
        api_host=api_host,
        api_port=api_port,
        web_host=web_host,
        web_port=web_port,
        cors_origin=cors_origin,
        api_origin=api_origin,
        ner_model=_model_name(
            merged["OPENMED_DEMO_NER_MODEL"], "OPENMED_DEMO_NER_MODEL"
        ),
        pii_model=_model_name(
            merged["OPENMED_DEMO_PII_MODEL"], "OPENMED_DEMO_PII_MODEL"
        ),
        cache_dir=_directory(merged["OPENMED_DEMO_CACHE_DIR"]),
        hf_home=_directory(merged["HF_HOME"]),
        hf_hub_cache=_directory(merged["HF_HUB_CACHE"]),
        hf_hub_offline=_flag(merged["HF_HUB_OFFLINE"], "HF_HUB_OFFLINE"),
        transformers_offline=_flag(
            merged["TRANSFORMERS_OFFLINE"],
            "TRANSFORMERS_OFFLINE",
        ),
        hf_datasets_offline=_flag(
            merged["HF_DATASETS_OFFLINE"],
            "HF_DATASETS_OFFLINE",
        ),
        openmed_offline=_flag(merged["OPENMED_OFFLINE"], "OPENMED_OFFLINE"),
        backend=backend,
    )


def apply_process_environment(config: DemoConfig) -> None:
    """Export cache and offline flags before the SDK constructs a loader.

    Hugging Face telemetry is disabled for this process. The demo does not
    send usage analytics.
    """

    os.environ["HF_HOME"] = config.hf_home
    os.environ["HF_HUB_CACHE"] = config.hf_hub_cache
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_OFFLINE"] = _flag_token(config.hf_hub_offline)
    os.environ["TRANSFORMERS_OFFLINE"] = _flag_token(config.transformers_offline)
    os.environ["HF_DATASETS_OFFLINE"] = _flag_token(config.hf_datasets_offline)
    os.environ["OPENMED_OFFLINE"] = _flag_token(config.openmed_offline)
    Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
    Path(config.hf_home).mkdir(parents=True, exist_ok=True)
    Path(config.hf_hub_cache).mkdir(parents=True, exist_ok=True)


def _flag_token(enabled: bool) -> str:
    return "1" if enabled else "0"


def _loopback_host(value: str, key: str) -> str:
    host = value.strip()
    if not is_loopback_host(host):
        raise ConfigError(f"not_loopback:{key}")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    return host


def _port(value: str, key: str) -> int:
    raw = value.strip()
    if not raw.isdigit():
        raise ConfigError(f"invalid_port:{key}")
    port = int(raw)
    if port < 1 or port > 65535:
        raise ConfigError(f"invalid_port:{key}")
    return port


def _flag(value: str, key: str) -> bool:
    token = value.strip().lower()
    if token in _TRUE_FLAGS:
        return True
    if token in _FALSE_FLAGS:
        return False
    raise ConfigError(f"invalid_flag:{key}")


def _model_name(value: str, key: str) -> str:
    from demo.api.catalog import validate_model_name

    try:
        return validate_model_name(value)
    except ValueError as exc:
        raise ConfigError(f"invalid_model:{key}") from exc


def _directory(value: str) -> str:
    raw = value.strip()
    if not raw:
        raise ConfigError("invalid_cache_dir")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = DEMO_ROOT / path
    return str(path)


def _origin(value: str, key: str, *, expected_port: int) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ConfigError(f"invalid_origin:{key}")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ConfigError(f"invalid_origin:{key}")
    if parsed.path not in {"", "/"}:
        raise ConfigError(f"invalid_origin:{key}")
    if not is_loopback_host(parsed.hostname):
        raise ConfigError(f"not_loopback:{key}")
    if parsed.port is None:
        raise ConfigError(f"origin_port_required:{key}")
    if parsed.port != expected_port:
        raise ConfigError(f"origin_port_mismatch:{key}")
    host = parsed.hostname
    netloc = f"[{host}]:{parsed.port}" if ":" in host else f"{host}:{parsed.port}"
    return f"{parsed.scheme}://{netloc}"
