"""Registry rows used by the local model catalog.

The demo lists OpenMed registry entries. It does not invent model ids.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CatalogEntry:
    """One registry artifact the demo can select or download."""

    key: str
    model_id: str
    display_name: str
    category: str
    family: str
    formats: tuple[str, ...]
    languages: tuple[str, ...]
    param_count: int | None
    base_model: str | None


def model_role(entry: CatalogEntry) -> str | None:
    """Return ``pii`` or ``ner`` when the artifact belongs on the demo picker."""

    if (
        entry.category == "Privacy"
        or entry.family.upper() == "PII"
        or entry.key.startswith("pii_")
    ):
        return "pii"
    if entry.family.upper() == "NER":
        return "ner"
    return None


def is_mlx_artifact(entry: CatalogEntry) -> bool:
    """Return whether the registry row ships MLX weights."""

    return any(fmt.startswith("mlx") for fmt in entry.formats)


def mlx_rank(formats: tuple[str, ...]) -> int:
    """Prefer full-precision MLX weights over quantized MLX siblings."""

    if "mlx-fp" in formats:
        return 0
    if any(
        fmt.startswith("mlx") and "8bit" not in fmt and "q8" not in fmt
        for fmt in formats
    ):
        return 1
    if any(fmt.startswith("mlx") for fmt in formats):
        return 2
    return 9


def validate_model_name(name: str) -> str:
    """Accept a registry key or Hub repo id and reject paths and URLs."""

    cleaned = name.strip()
    if not cleaned or len(cleaned) > 200:
        raise ValueError("invalid_model")
    if any(char in cleaned for char in " \t\r\n\\"):
        raise ValueError("invalid_model")
    if "://" in cleaned or cleaned.startswith("/"):
        raise ValueError("invalid_model")
    if ".." in cleaned.split("/"):
        raise ValueError("invalid_model")
    return cleaned
