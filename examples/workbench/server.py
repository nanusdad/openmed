"""Local web bench for trying OpenMed models on clinical notes.

Run from the repo root:

    pip install "openmed[hf,service]"
    python examples/workbench/server.py
"""

from __future__ import annotations

import os
import re
import sys
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import openmed
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from openmed.core.model_registry import CATEGORIES, OPENMED_MODELS, ModelInfo
from openmed.core.models import ModelLoader

STATIC_DIR = Path(__file__).resolve().parent / "static"

SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "conditions",
        "name": "Conditions",
        "task": "analyze",
        "summary": "Diseases and diagnoses in a note.",
        "sample": (
            "Patient started on imatinib for chronic myeloid leukemia. "
            "History of type 2 diabetes mellitus and essential hypertension. "
            "No evidence of pneumonia on this admission."
        ),
        "models": ["disease_detection_tiny", "disease_detection_superclinical"],
    },
    {
        "id": "medications",
        "name": "Medications",
        "task": "analyze",
        "summary": "Drugs and doses.",
        "sample": (
            "Discharge meds: metformin 500 mg twice daily, atorvastatin 20 mg "
            "at night, and a five-day course of amoxicillin."
        ),
        "models": ["pharma_detection_supermedical", "pharma_detection_superclinical"],
    },
    {
        "id": "cancer",
        "name": "Cancer",
        "task": "analyze",
        "summary": "Tumors, cells, and genes in oncology notes.",
        "sample": (
            "Metastatic breast cancer treated with paclitaxel and trastuzumab. "
            "Biopsy shows invasive ductal carcinoma. HER2 amplified."
        ),
        "models": ["oncology_detection_tiny", "oncology_detection_superclinical"],
    },
    {
        "id": "anatomy",
        "name": "Anatomy",
        "task": "analyze",
        "summary": "Organs and body structures.",
        "sample": (
            "CT shows a lesion in the right hepatic lobe. "
            "The left kidney and spleen are unremarkable. "
            "No mediastinal lymphadenopathy."
        ),
        "models": ["anatomy_detection_electramed"],
    },
    {
        "id": "pathology",
        "name": "Pathology",
        "task": "analyze",
        "summary": "Pathologic findings.",
        "sample": (
            "Skin biopsy consistent with melanoma in situ. "
            "Margins are clear. Background solar elastosis."
        ),
        "models": ["pathology_detection_modern"],
    },
    {
        "id": "genes",
        "name": "Genes",
        "task": "analyze",
        "summary": "Genes, proteins, and DNA mentions.",
        "sample": (
            "BRCA1 variant of uncertain significance. "
            "TP53 expression is elevated. EGFR mutation testing is pending."
        ),
        "models": ["genome_detection_bioclinical", "dna_detection_supermedical"],
    },
    {
        "id": "identifiers",
        "name": "Identifiers",
        "task": "pii",
        "summary": "Names, dates, and other identifiers. Uses the same cleanup as the Swift demo.",
        "sample": (
            "Patient: John Doe, DOB: 01/15/1970, SSN: 000-00-0000, "
            "MRN: MRN-TEST-88421, Address: 123 Example Street, Springfield, CA 90000, "
            "Phone: (555) 010-2244, Email: john.doe@example.test."
        ),
        "models": [
            "pii_clinical_e5_small",
            "pii_lite_clinical",
            "pii_fast_clinical",
            "pii_detection",
        ],
    },
    {
        "id": "redaction",
        "name": "Redaction",
        "task": "deidentify",
        "summary": "Replace identifiers in the note and show the redacted text.",
        "sample": (
            "Patient: John Doe, DOB: 01/15/1970, SSN: 000-00-0000, "
            "Phone: (555) 010-2244, Email: john.doe@example.test. "
            "Follow up with Jane Doe at Example Manufacturing LLC."
        ),
        "models": ["pii_lite_clinical", "pii_clinical_e5_small", "pii_detection"],
    },
]

_loader: ModelLoader | None = None
_loader_lock = threading.Lock()
_infer_lock = threading.Lock()


class AnalyzeBody(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    model_name: str = Field(min_length=1, max_length=200)
    task: str = Field(pattern="^(analyze|pii|deidentify)$")
    confidence_threshold: float = Field(ge=0.0, le=1.0, default=0.5)


_PARAM_RE = re.compile(r"(\d+)M")


def _size_mb(info: ModelInfo) -> int | None:
    """Prefer the parameter count in the model id over the registry estimate."""
    match = _PARAM_RE.search(info.model_id)
    if match:
        return int(match.group(1))
    return info.size_mb


def _model_payload(key: str, info: ModelInfo) -> dict[str, Any]:
    return {
        "key": key,
        "display_name": info.display_name,
        "category": info.category,
        "description": info.description,
        "entity_types": list(info.entity_types),
        "size_category": info.size_category,
        "size_mb": _size_mb(info),
        "recommended_confidence": info.recommended_confidence,
        "model_id": info.model_id,
    }


def _catalog() -> dict[str, Any]:
    models = {
        key: _model_payload(key, info) for key, info in OPENMED_MODELS.items()
    }
    grouped: dict[str, list[str]] = {}
    for category, keys in CATEGORIES.items():
        present = [key for key in keys if key in models]
        present.sort(
            key=lambda key: (
                models[key]["size_mb"] is None,
                models[key]["size_mb"] or 0,
                models[key]["display_name"],
            )
        )
        if present:
            grouped[category] = present

    scenarios = []
    for scenario in SCENARIOS:
        scenarios.append(
            {
                **scenario,
                "models": [key for key in scenario["models"] if key in models],
            }
        )
    return {"scenarios": scenarios, "models": models, "groups": grouped}


def _get_loader() -> ModelLoader:
    global _loader
    with _loader_lock:
        if _loader is None:
            _loader = ModelLoader()
        return _loader


def _entities_from(result: Any) -> list[dict[str, Any]]:
    raw = getattr(result, "entities", None)
    if raw is None:
        raw = getattr(result, "pii_entities", None) or []
    entities = []
    for entity in raw:
        if hasattr(entity, "to_dict"):
            item = entity.to_dict()
        else:
            item = dict(entity)
        entities.append(
            {
                "text": item.get("text") or "",
                "label": item.get("label") or item.get("entity_type") or "ENTITY",
                "confidence": float(item.get("confidence") or 0.0),
                "start": item.get("start"),
                "end": item.get("end"),
            }
        )
    return entities


def _run(body: AnalyzeBody) -> dict[str, Any]:
    loader = _get_loader()
    text = body.text.strip()
    if body.task == "pii":
        result = openmed.extract_pii(
            text,
            model_name=body.model_name,
            confidence_threshold=body.confidence_threshold,
            loader=loader,
        )
        payload = {
            "text": getattr(result, "text", text),
            "entities": _entities_from(result),
            "model_name": getattr(result, "model_name", body.model_name),
            "processing_time": getattr(result, "processing_time", None),
            "deidentified_text": None,
        }
        return payload

    if body.task == "deidentify":
        result = openmed.deidentify(
            text,
            method="mask",
            model_name=body.model_name,
            confidence_threshold=body.confidence_threshold,
            loader=loader,
        )
        return {
            "text": result.original_text,
            "entities": _entities_from(result),
            "model_name": body.model_name,
            "processing_time": None,
            "deidentified_text": result.deidentified_text,
        }

    result = openmed.analyze_text(
        text,
        model_name=body.model_name,
        loader=loader,
        output_format="dict",
        confidence_threshold=body.confidence_threshold,
        group_entities=True,
        aggregation_strategy="simple",
    )
    data = result.to_dict() if hasattr(result, "to_dict") else {}
    return {
        "text": data.get("text", text),
        "entities": _entities_from(result),
        "model_name": data.get("model_name", body.model_name),
        "processing_time": data.get("processing_time"),
        "deidentified_text": None,
    }


app = FastAPI(title="OpenMed chart bench", version=openmed.__version__)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/catalog")
def catalog() -> dict[str, Any]:
    return _catalog()


@app.post("/api/analyze")
def analyze(body: AnalyzeBody) -> dict[str, Any]:
    if body.model_name not in OPENMED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{body.model_name}'. Pick a model from the list.",
        )
    try:
        with _infer_lock:
            return _run(body)
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "The model runtime is not installed. "
                'Run: pip install "openmed[hf]"'
            ),
        ) from exc
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        raise HTTPException(status_code=502, detail=message[:500]) from exc


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", "8765"))
    host = os.environ.get("HOST", "127.0.0.1")
    if os.environ.get("PORT") and "HOST" not in os.environ:
        host = "0.0.0.0"
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
