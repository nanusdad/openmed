"""Adapters over the public OpenMed SDK.

The demo calls ``analyze_text``, ``extract_pii``, ``deidentify``,
``prefetch_model``, and the registry/cache helpers. It does not import the
REST service stack or any remote inference adapter.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

from demo.api.catalog import CatalogEntry


class DemoSdk(Protocol):
    """Public SDK surface the demo session is allowed to call."""

    def open_config(
        self,
        *,
        cache_dir: str,
        backend: str | None,
        local_only: bool,
    ) -> Any:
        """Return an SDK config object bound to the local cache."""

    def open_loader(self, config: Any) -> Any:
        """Return a process-wide model loader."""

    def analyze_text(self, text: str, **kwargs: Any) -> Any:
        """Run clinical NER."""

    def extract_pii(self, text: str, **kwargs: Any) -> Any:
        """Extract PII spans."""

    def deidentify(self, text: str, **kwargs: Any) -> Any:
        """De-identify text."""

    def list_models(self) -> list[CatalogEntry]:
        """Return deduplicated registry rows."""

    def list_cached_repo_ids(self, *, cache_dir: str) -> set[str]:
        """Return Hub repo ids present in the local cache."""

    def prefetch(
        self,
        model_name: str,
        *,
        cache_dir: str,
        config: Any,
        progress: Callable[[int, int], None] | None,
    ) -> str:
        """Download one model into the local cache."""

    def mlx_available(self) -> bool:
        """Return whether Apple MLX can run on this machine."""


class RealSdk:
    """Lazy calls into the installed OpenMed package."""

    def open_config(
        self,
        *,
        cache_dir: str,
        backend: str | None,
        local_only: bool,
    ) -> Any:
        from openmed.core.config import OpenMedConfig

        return OpenMedConfig(
            cache_dir=cache_dir,
            backend=backend,
            local_only=local_only,
            remote_inference_endpoint=None,
        )

    def open_loader(self, config: Any) -> Any:
        from openmed.core.models import ModelLoader

        return ModelLoader(config)

    def analyze_text(self, text: str, **kwargs: Any) -> Any:
        import openmed

        return openmed.analyze_text(text, **kwargs)

    def extract_pii(self, text: str, **kwargs: Any) -> Any:
        import openmed

        return openmed.extract_pii(text, **kwargs)

    def deidentify(self, text: str, **kwargs: Any) -> Any:
        import openmed

        return openmed.deidentify(text, **kwargs)

    def list_models(self) -> list[CatalogEntry]:
        from openmed.core.model_registry import get_all_models

        entries: list[CatalogEntry] = []
        for key, info in get_all_models().items():
            entries.append(
                CatalogEntry(
                    key=key,
                    model_id=info.model_id,
                    display_name=info.display_name,
                    category=info.category,
                    family=info.family,
                    formats=tuple(info.formats or ()),
                    languages=tuple(info.languages or ()),
                    param_count=info.param_count,
                    base_model=info.base_model,
                )
            )
        entries.sort(key=lambda item: item.key)
        return entries

    def list_cached_repo_ids(self, *, cache_dir: str) -> set[str]:
        from openmed.core.hf_hub import list_cached_models

        try:
            cached = list_cached_models(cache_dir=cache_dir)
        except ImportError:
            return set()
        return {item.repo_id for item in cached}

    def prefetch(
        self,
        model_name: str,
        *,
        cache_dir: str,
        config: Any,
        progress: Callable[[int, int], None] | None,
    ) -> str:
        from openmed.core.hf_hub import prefetch_model

        def callback(update: Any) -> None:
            if progress is None:
                return
            progress(int(update.files_done), int(update.files_total))

        return prefetch_model(
            model_name,
            cache_dir=cache_dir,
            config=config,
            progress_callback=callback,
        )

    def mlx_available(self) -> bool:
        from openmed.core.backends import MLXBackend

        return bool(MLXBackend().is_available())
