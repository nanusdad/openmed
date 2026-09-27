"""One in-process inference session.

Models are loaded through a single SDK ``ModelLoader`` and reused. Result
caching stays off so PHI is not kept in the SDK LRU. The session does not
write request or response bodies.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Callable

from demo.api.catalog import (
    CatalogEntry,
    is_mlx_artifact,
    mlx_rank,
    model_role,
    validate_model_name,
)
from demo.api.config import DemoConfig
from demo.api.errors import DemoError
from demo.api.sdk import DemoSdk
from openmed.core.offline import (
    HF_OFFLINE_ENV_VARS,
    OFFLINE_ENV_VAR,
    enable_hf_offline_flags,
)

MAX_TEXT_CHARS = 20_000


def _plain(value: Any) -> Any:
    """Reduce an SDK payload to JSON values without stringifying objects."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _plain(item)
            for key, item in value.items()
            if key not in {"mapping", "audit_report"}
        }
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _plain(item())
        except Exception:
            return None
    return None


def json_ready(result: Any) -> dict[str, Any]:
    """Return a JSON object and drop reversible de-identification maps."""

    if isinstance(result, dict):
        payload = dict(result)
    else:
        to_dict = getattr(result, "to_dict", None)
        if not callable(to_dict):
            raise DemoError("inference_failed", 500)
        payload = to_dict()
    if not isinstance(payload, dict):
        raise DemoError("inference_failed", 500)
    return _plain(payload)


class InferenceSession:
    """Shared loader, active model ids, and local download status."""

    def __init__(self, config: DemoConfig, sdk: DemoSdk) -> None:
        self.config = config
        self.sdk = sdk
        self.mlx_available = bool(sdk.mlx_available())
        self._entries = list(sdk.list_models())
        self._by_key = {entry.key: entry for entry in self._entries}
        self._by_id: dict[str, CatalogEntry] = {}
        self._canonical_key: dict[str, str] = {}
        for entry in self._entries:
            self._by_id.setdefault(entry.model_id, entry)
            current = self._canonical_key.get(entry.model_id)
            if current is None or _shorter_key(entry.key, current):
                self._canonical_key[entry.model_id] = entry.key
        self.ner_model = self._canonical_name(config.ner_model)
        self.pii_model = self._canonical_name(config.pii_model)
        backend = None if config.backend == "auto" else config.backend
        self.om_config = sdk.open_config(
            cache_dir=config.cache_dir,
            backend=backend,
            local_only=config.requests_offline,
        )
        self._loader: Any = None
        self._state = threading.Lock()
        self._inference = threading.Lock()
        self._ready: set[str] = set()
        self._known_cached: set[str] = set()
        self._downloads: dict[str, dict[str, int]] = {}
        self._errors: dict[str, str] = {}
        self._threads: list[threading.Thread] = []
        self._started = False
        self._runtime_offline = config.requests_offline
        self._cache_snapshot: set[str] | None = None
        self._cache_snapshot_at = 0.0

    @property
    def loader(self) -> Any:
        """Return the shared SDK loader, creating it on first use."""

        if self._loader is None:
            try:
                self._loader = self.sdk.open_loader(self.om_config)
            except Exception as exc:
                if exc.__class__.__name__ in {
                    "ImportError",
                    "ModuleNotFoundError",
                    "MissingExtraError",
                }:
                    raise DemoError("hf_extra_missing", 503) from exc
                raise
        return self._loader

    def start(self) -> None:
        """Warm active models that are already in the local cache."""

        with self._state:
            if self._started:
                return
            self._started = True
        thread = threading.Thread(
            target=self._startup_warm,
            name="openmed-demo-warm",
            daemon=True,
        )
        self._threads.append(thread)
        thread.start()

    def close(self) -> None:
        """Wait briefly for warmup and download threads."""

        self.wait_for_background(timeout=0.2)

    def wait_for_background(self, timeout: float = 2.0) -> None:
        """Join background threads started by this session."""

        for thread in list(self._threads):
            thread.join(timeout)

    def catalog(self) -> dict[str, Any]:
        """Return active ids and NER/PII rows with cache status."""

        with self._state:
            cached = self._cached_repo_ids_unlocked()
            canonical_keys = set(self._canonical_key.values())
            models = [
                self._public_model(entry, role, cached)
                for entry in self._entries
                if entry.key in canonical_keys
                and (role := model_role(entry)) is not None
            ]
            return {
                "active": {
                    "ner_model": self.ner_model,
                    "pii_model": self.pii_model,
                    "resolved_ner_model": self._safe_resolve(self.ner_model),
                    "resolved_pii_model": self._safe_resolve(self.pii_model),
                    "backend": self.config.backend,
                    "preferred_backend": self._preferred_backend(),
                    "offline": self._runtime_offline or self.config.requests_offline,
                },
                "platform": {
                    "system": _platform_system(),
                    "machine": _platform_machine(),
                    "mlx_available": self.mlx_available,
                },
                "models": models,
            }

    def select(self, *, ner_model: str | None, pii_model: str | None) -> dict[str, Any]:
        """Switch the active models on the existing loader."""

        if ner_model is None and pii_model is None:
            raise DemoError("invalid_request", 422)
        with self._state:
            if ner_model is not None:
                entry = self._require_role(ner_model, "ner")
                self.ner_model = entry.key
                self._errors.pop(entry.key, None)
            if pii_model is not None:
                entry = self._require_role(pii_model, "pii")
                self.pii_model = entry.key
                self._errors.pop(entry.key, None)
            self._sync_offline_unlocked()
        self._warm_active_in_background()
        return self.catalog()

    def pull(self, model_name: str, *, background: bool = True) -> dict[str, Any]:
        """Download a catalog model into the local cache and warm it."""

        with self._state:
            entry = self._require_known(model_name)
            if entry.key in self._downloads:
                return {"model_name": entry.key, "status": "downloading"}
            self._downloads[entry.key] = {"files_done": 0, "files_total": 0}
            self._errors.pop(entry.key, None)
            key = entry.key
        if background:
            thread = threading.Thread(
                target=self._pull_one,
                args=(key,),
                name="openmed-demo-pull",
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()
            return {"model_name": key, "status": "downloading"}
        self._pull_one(key)
        with self._state:
            return {"model_name": key, "status": self._status_for(entry)}

    def analyze(
        self,
        text: str,
        *,
        model_name: str | None = None,
        confidence_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Run ``analyze_text`` on the shared loader."""

        cleaned = _require_text(text)
        with self._inference:
            with self._state:
                model = self._resolve_unlocked(model_name or self.ner_model)
                self._require_known(model)
            kwargs: dict[str, Any] = {
                "model_name": model,
                "loader": self.loader,
                "cache_results": False,
                "output_format": "dict",
            }
            if confidence_threshold is not None:
                kwargs["confidence_threshold"] = confidence_threshold
            result = self._invoke(self.sdk.analyze_text, cleaned, **kwargs)
            with self._state:
                self._ready.add(model)
                self._sync_offline_unlocked()
            return result

    def extract_pii(
        self,
        text: str,
        *,
        model_name: str | None = None,
        confidence_threshold: float | None = None,
        lang: str = "en",
    ) -> dict[str, Any]:
        """Run ``extract_pii`` on the shared loader."""

        cleaned = _require_text(text)
        language = _require_lang(lang)
        with self._inference:
            with self._state:
                model = self._resolve_unlocked(model_name or self.pii_model)
                self._require_known(model)
            kwargs: dict[str, Any] = {
                "model_name": model,
                "loader": self.loader,
                "cache_results": False,
                "lang": language,
            }
            if confidence_threshold is not None:
                kwargs["confidence_threshold"] = confidence_threshold
            result = self._invoke(self.sdk.extract_pii, cleaned, **kwargs)
            with self._state:
                self._ready.add(model)
                self._sync_offline_unlocked()
            return result

    def deidentify(
        self,
        text: str,
        *,
        method: str,
        model_name: str | None = None,
        confidence_threshold: float | None = None,
        lang: str = "en",
    ) -> dict[str, Any]:
        """Run ``deidentify`` on the shared loader."""

        cleaned = _require_text(text)
        if method not in {"mask", "replace"}:
            raise DemoError("invalid_request", 422)
        language = _require_lang(lang)
        with self._inference:
            with self._state:
                model = self._resolve_unlocked(model_name or self.pii_model)
                self._require_known(model)
            kwargs: dict[str, Any] = {
                "method": method,
                "model_name": model,
                "loader": self.loader,
                "cache_results": False,
                "keep_mapping": False,
                "audit": False,
                "lang": language,
            }
            if confidence_threshold is not None:
                kwargs["confidence_threshold"] = confidence_threshold
            result = self._invoke(self.sdk.deidentify, cleaned, **kwargs)
            with self._state:
                self._ready.add(model)
                self._sync_offline_unlocked()
            return result

    def _invoke(
        self,
        fn: Callable[..., Any],
        text: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            return json_ready(fn(text, **kwargs))
        except DemoError:
            raise
        except Exception as exc:
            code = "inference_failed"
            status = 500
            if exc.__class__.__name__ == "OfflineModeError":
                code = "offline_model_missing"
                status = 409
            elif exc.__class__.__name__ == "ModelLoadError":
                code = "model_load_failed"
                status = 409
            elif exc.__class__.__name__ in {
                "ImportError",
                "ModuleNotFoundError",
                "MissingExtraError",
            }:
                code = "hf_extra_missing"
                status = 503
            raise DemoError(code, status) from exc

    def _startup_warm(self) -> None:
        with self._state:
            self._sync_offline_unlocked()
            targets = [
                self._safe_resolve(self.ner_model),
                self._safe_resolve(self.pii_model),
            ]
            cached = self._cached_repo_ids_unlocked() | self._known_cached
        for model in targets:
            entry = self._entry_for(model)
            repo_id = entry.model_id if entry is not None else model
            if repo_id not in cached and model not in cached:
                continue
            with self._inference:
                with self._state:
                    try:
                        self._warm_unlocked(model)
                        self._sync_offline_unlocked()
                    except Exception as exc:
                        self._errors[model] = exc.__class__.__name__

    def _warm_active_in_background(self) -> None:
        thread = threading.Thread(
            target=self._startup_warm,
            name="openmed-demo-rewarm",
            daemon=True,
        )
        self._threads.append(thread)
        thread.start()

    def _pull_one(self, model_name: str) -> None:
        try:
            with self._state:
                entry = self._require_known(model_name)
                resolved = self._resolve_unlocked(entry.key)
            with self._inference:

                def progress(files_done: int, files_total: int) -> None:
                    with self._state:
                        if entry.key in self._downloads:
                            self._downloads[entry.key] = {
                                "files_done": files_done,
                                "files_total": files_total,
                            }

                with self._hub_window():
                    self.sdk.prefetch(
                        resolved,
                        cache_dir=self.config.cache_dir,
                        config=self.om_config,
                        progress=progress,
                    )
                with self._state:
                    self._known_cached.add(self._repo_id(resolved))
                    self._cache_snapshot = None
                    self._warm_unlocked(resolved)
                    self._sync_offline_unlocked()
        except DemoError as exc:
            with self._state:
                self._errors[model_name] = exc.code
        except Exception as exc:
            with self._state:
                self._errors[model_name] = exc.__class__.__name__
        finally:
            with self._state:
                self._downloads.pop(model_name, None)

    def _warm_unlocked(self, model_name: str) -> None:
        self.loader.create_pipeline(
            model_name,
            task="token-classification",
            aggregation_strategy="simple",
            use_fast_tokenizer=True,
        )
        self._ready.add(model_name)
        self._errors.pop(model_name, None)

    def _hub_window(self) -> _HubWindow:
        return _HubWindow(self)

    def _sync_offline_unlocked(self) -> None:
        if self.config.requests_offline:
            self._enable_runtime_offline()
            return
        cached = self._cached_repo_ids_unlocked() | self._known_cached
        ready_repos = {self._repo_id(name) for name in self._ready}
        needed = {
            self._repo_id(self._safe_resolve(self.ner_model)),
            self._repo_id(self._safe_resolve(self.pii_model)),
        }
        if needed <= (cached | ready_repos):
            self._enable_runtime_offline()
        elif self._runtime_offline:
            self._runtime_offline = False
            self.om_config.local_only = False

    def _enable_runtime_offline(self) -> None:
        self._runtime_offline = True
        self.om_config.local_only = True
        enable_hf_offline_flags()
        os.environ[OFFLINE_ENV_VAR] = "1"

    def _cached_repo_ids_unlocked(self) -> set[str]:
        import time

        now = time.monotonic()
        if self._cache_snapshot is not None and now - self._cache_snapshot_at < 1.0:
            return set(self._cache_snapshot)
        try:
            found = self.sdk.list_cached_repo_ids(cache_dir=self.config.cache_dir)
        except Exception:
            found = set()
        self._cache_snapshot = set(found)
        self._cache_snapshot_at = now
        return set(found)

    def _safe_resolve(self, name: str) -> str:
        try:
            return self._resolve_unlocked(name)
        except DemoError:
            return name

    def _resolve_unlocked(self, name: str) -> str:
        try:
            cleaned = validate_model_name(name)
        except ValueError as exc:
            raise DemoError("invalid_model", 422) from exc
        entry = self._entry_for(cleaned)
        if entry is None:
            return cleaned
        entry = self._canonical_entry(entry)
        if not self._use_mlx():
            return entry.key
        if is_mlx_artifact(entry):
            return entry.key
        sibling = self._mlx_sibling(entry)
        if sibling is not None:
            return sibling.key
        if self.config.backend == "mlx":
            raise DemoError("mlx_artifact_unavailable", 409)
        return entry.key

    def _use_mlx(self) -> bool:
        if self.config.backend == "hf":
            return False
        return self.mlx_available

    def _mlx_sibling(self, entry: CatalogEntry) -> CatalogEntry | None:
        canonical_keys = set(self._canonical_key.values())
        candidates = [
            other
            for other in self._entries
            if other.key in canonical_keys
            and other.base_model == entry.model_id
            and is_mlx_artifact(other)
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda item: (mlx_rank(item.formats), item.key))
        return candidates[0]

    def _entry_for(self, name: str) -> CatalogEntry | None:
        return self._by_key.get(name) or self._by_id.get(name)

    def _repo_id(self, name: str) -> str:
        entry = self._entry_for(name)
        if entry is None:
            return name
        return entry.model_id

    def _require_known(self, name: str) -> CatalogEntry:
        try:
            cleaned = validate_model_name(name)
        except ValueError as exc:
            raise DemoError("invalid_model", 422) from exc
        entry = self._entry_for(cleaned)
        if entry is None or model_role(entry) is None:
            raise DemoError("unknown_model", 422)
        return self._canonical_entry(entry)

    def _canonical_entry(self, entry: CatalogEntry) -> CatalogEntry:
        key = self._canonical_key.get(entry.model_id, entry.key)
        return self._by_key.get(key, entry)

    def _canonical_name(self, name: str) -> str:
        entry = self._entry_for(name)
        if entry is None:
            return name
        return self._canonical_entry(entry).key

    def _require_role(self, name: str, role: str) -> CatalogEntry:
        entry = self._require_known(name)
        if model_role(entry) != role:
            raise DemoError("unknown_model", 422)
        return entry

    def _preferred_backend(self) -> str:
        if self.config.backend == "hf":
            return "hf"
        if self.config.backend == "mlx":
            return "mlx" if self.mlx_available else "unavailable"
        if self.mlx_available:
            return "mlx"
        return "hf"

    def _public_model(
        self,
        entry: CatalogEntry,
        role: str,
        cached: set[str],
    ) -> dict[str, Any]:
        return {
            "key": entry.key,
            "model_id": entry.model_id,
            "display_name": entry.display_name,
            "role": role,
            "category": entry.category,
            "family": entry.family,
            "formats": list(entry.formats),
            "languages": list(entry.languages),
            "param_count": entry.param_count,
            "status": self._status_for(entry, cached),
            "error": self._errors.get(entry.key),
            "progress": self._downloads.get(entry.key),
        }

    def _status_for(
        self,
        entry: CatalogEntry,
        cached: set[str] | None = None,
    ) -> str:
        if entry.key in self._downloads:
            return "downloading"
        resolved = self._safe_resolve(entry.key)
        if resolved in self._ready or entry.key in self._ready:
            return "ready"
        present = self._known_cached if cached is None else cached | self._known_cached
        if entry.model_id in present or entry.key in present:
            return "cached"
        return "available"


class _HubWindow:
    """Temporarily allow a Hub download when the operator did not force offline."""

    def __init__(self, session: InferenceSession) -> None:
        self._session = session
        self._saved_env: dict[str, str | None] = {}
        self._saved_local = False

    def __enter__(self) -> _HubWindow:
        session = self._session
        if session.config.requests_offline:
            return self
        keys = (*HF_OFFLINE_ENV_VARS, OFFLINE_ENV_VAR)
        self._saved_env = {key: os.environ.get(key) for key in keys}
        self._saved_local = bool(getattr(session.om_config, "local_only", False))
        for key in HF_OFFLINE_ENV_VARS:
            os.environ[key] = "0"
        os.environ[OFFLINE_ENV_VAR] = "0"
        session.om_config.local_only = False
        return self

    def __exit__(self, *_exc: object) -> None:
        session = self._session
        if session.config.requests_offline:
            return
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        session.om_config.local_only = self._saved_local


def _require_text(text: str) -> str:
    if not isinstance(text, str) or text.strip() == "":
        raise DemoError("empty_text", 422)
    if len(text) > MAX_TEXT_CHARS:
        raise DemoError("text_too_long", 413)
    return text


def _require_lang(lang: str) -> str:
    if not isinstance(lang, str):
        raise DemoError("invalid_request", 422)
    cleaned = lang.strip().lower()
    if len(cleaned) < 2 or len(cleaned) > 16:
        raise DemoError("invalid_request", 422)
    for char in cleaned:
        if char not in "abcdefghijklmnopqrstuvwxyz-":
            raise DemoError("invalid_request", 422)
    return cleaned


def _shorter_key(candidate: str, current: str) -> bool:
    if len(candidate) != len(current):
        return len(candidate) < len(current)
    return candidate < current


def _platform_system() -> str:
    import platform

    return platform.system()


def _platform_machine() -> str:
    import platform

    return platform.machine()
