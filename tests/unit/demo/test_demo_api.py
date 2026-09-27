"""Loopback demo API tests.

These tests use a fake SDK so they never download weights or persist text.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from demo.api.app import CATALOG_ROUTES, INFERENCE_ROUTES, create_app, route_paths
from demo.api.catalog import CatalogEntry
from demo.api.config import (
    REQUIRED_ENV_NAMES,
    ConfigError,
    apply_process_environment,
    assert_loopback_bind,
    load_demo_config,
    parse_env_file,
)
from demo.api.sdk import RealSdk
from demo.api.session import InferenceSession

ROOT = Path(__file__).resolve().parents[3]
TOKEN = "SYNTHETIC-PHI-TOKEN-9f3a"
HUB_KEYS = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "OPENMED_OFFLINE",
    "HF_HUB_DISABLE_TELEMETRY",
    "HF_HOME",
    "HF_HUB_CACHE",
)


def _env(**overrides: str) -> dict[str, str]:
    values = {
        "OPENMED_DEMO_API_HOST": "127.0.0.1",
        "OPENMED_DEMO_API_PORT": "8765",
        "OPENMED_DEMO_WEB_HOST": "127.0.0.1",
        "OPENMED_DEMO_WEB_PORT": "5173",
        "OPENMED_DEMO_CORS_ORIGIN": "http://127.0.0.1:5173",
        "VITE_OPENMED_DEMO_API_ORIGIN": "http://127.0.0.1:8765",
        "OPENMED_DEMO_NER_MODEL": "disease_detection_electramed_33m",
        "OPENMED_DEMO_PII_MODEL": "pii_superclinical_small",
        "OPENMED_DEMO_CACHE_DIR": ".cache/openmed-demo/hub",
        "HF_HOME": ".cache/openmed-demo",
        "HF_HUB_CACHE": ".cache/openmed-demo/hub",
        "HF_HUB_OFFLINE": "0",
        "TRANSFORMERS_OFFLINE": "0",
        "HF_DATASETS_OFFLINE": "0",
        "OPENMED_OFFLINE": "0",
        "OPENMED_DEMO_BACKEND": "auto",
    }
    values.update(overrides)
    return values


def _entries() -> list[CatalogEntry]:
    ner = CatalogEntry(
        key="disease_detection_electramed_33m",
        model_id="OpenMed/OpenMed-NER-DiseaseDetect-ElectraMed-33M",
        display_name="NER DiseaseDetect ElectraMed 33M",
        category="Disease",
        family="NER",
        formats=("pytorch",),
        languages=("en",),
        param_count=33_000_000,
        base_model=None,
    )
    ner_mlx = CatalogEntry(
        key="disease_detection_electramed_33m_mlx",
        model_id="OpenMed/OpenMed-NER-DiseaseDetect-ElectraMed-33M-mlx",
        display_name="NER DiseaseDetect ElectraMed 33M MLX",
        category="Disease",
        family="NER",
        formats=("mlx-fp", "pytorch"),
        languages=("en",),
        param_count=33_000_000,
        base_model=ner.model_id,
    )
    ner_q8 = CatalogEntry(
        key="disease_detection_electramed_33m_mlx_q8",
        model_id="OpenMed/OpenMed-NER-DiseaseDetect-ElectraMed-33M-mlx-q8",
        display_name="NER DiseaseDetect ElectraMed 33M MLX Q8",
        category="Disease",
        family="NER",
        formats=("mlx-8bit",),
        languages=("en",),
        param_count=33_000_000,
        base_model=ner.model_id,
    )
    pii = CatalogEntry(
        key="pii_superclinical_small",
        model_id="OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
        display_name="PII SuperClinical Small",
        category="Privacy",
        family="PII",
        formats=("pytorch",),
        languages=("en",),
        param_count=44_000_000,
        base_model=None,
    )
    pii_alias = CatalogEntry(
        key="pii_superclinical_small_alias",
        model_id=pii.model_id,
        display_name=pii.display_name,
        category=pii.category,
        family=pii.family,
        formats=pii.formats,
        languages=pii.languages,
        param_count=pii.param_count,
        base_model=None,
    )
    other = CatalogEntry(
        key="pharma_detection_electramed_33m",
        model_id="OpenMed/OpenMed-NER-PharmaDetect-ElectraMed-33M",
        display_name="Pharma",
        category="Pharmaceutical",
        family="NER",
        formats=("pytorch",),
        languages=("en",),
        param_count=33_000_000,
        base_model=None,
    )
    return [ner, ner_mlx, ner_q8, pii, pii_alias, other]


class _Loader:
    def __init__(self) -> None:
        self.pipelines: list[str] = []
        self.config = None

    def create_pipeline(self, model_name: str, **kwargs: object) -> object:
        self.pipelines.append(model_name)
        return {"model": model_name, "kwargs": kwargs}


class _FakeSdk:
    def __init__(self, *, mlx: bool = False) -> None:
        self.loader = _Loader()
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.cached: set[str] = set()
        self.prefetched: list[str] = []
        self.config = None
        self._mlx = mlx

    def open_config(self, *, cache_dir: str, backend: str | None, local_only: bool):
        self.config = type(
            "Cfg",
            (),
            {
                "cache_dir": cache_dir,
                "backend": backend,
                "local_only": local_only,
                "remote_inference_endpoint": None,
            },
        )()
        return self.config

    def open_loader(self, config: object) -> _Loader:
        self.loader.config = config
        return self.loader

    def list_models(self) -> list[CatalogEntry]:
        return _entries()

    def list_cached_repo_ids(self, *, cache_dir: str) -> set[str]:
        assert cache_dir
        return set(self.cached)

    def prefetch(self, model_name: str, *, cache_dir: str, config: object, progress):
        assert cache_dir
        assert config is not None
        self.prefetched.append(model_name)
        if progress is not None:
            progress(1, 2)
        entry = next(item for item in _entries() if item.key == model_name)
        self.cached.add(entry.model_id)
        return cache_dir

    def mlx_available(self) -> bool:
        return self._mlx

    def analyze_text(self, text: str, **kwargs: object) -> dict[str, object]:
        self.calls.append(("analyze_text", dict(kwargs)))
        return {
            "text": text,
            "entities": [
                {
                    "text": "asthma",
                    "label": "DISEASE",
                    "confidence": 0.91,
                    "start": text.find("asthma"),
                    "end": text.find("asthma") + 6,
                }
            ],
            "model_name": kwargs["model_name"],
            "mapping": {"drop": "me"},
            "timestamp": "t",
            "processing_time": 0.01,
            "metadata": {},
        }

    def extract_pii(self, text: str, **kwargs: object) -> dict[str, object]:
        self.calls.append(("extract_pii", dict(kwargs)))
        start = text.find("Casey")
        return {
            "text": text,
            "entities": [
                {
                    "text": "Casey Example",
                    "label": "NAME",
                    "confidence": 0.88,
                    "start": start,
                    "end": start + len("Casey Example"),
                }
            ],
            "model_name": kwargs["model_name"],
            "mapping": {"drop": "me"},
        }

    def deidentify(self, text: str, **kwargs: object) -> dict[str, object]:
        self.calls.append(("deidentify", dict(kwargs)))
        return {
            "original_text": text,
            "deidentified_text": text.replace("Casey Example", "[NAME]"),
            "pii_entities": [
                {
                    "text": "Casey Example",
                    "label": "NAME",
                    "confidence": 0.88,
                    "start": text.find("Casey Example"),
                    "end": text.find("Casey Example") + len("Casey Example"),
                }
            ],
            "method": kwargs["method"],
            "model_name": kwargs["model_name"],
            "mapping": {"[NAME]": "Casey Example"},
            "audit_report": {"original_text": text},
        }


@pytest.fixture
def restore_hub_env():
    saved = {key: os.environ.get(key) for key in HUB_KEYS}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _config(tmp_path: Path, **overrides: str):
    cache = tmp_path / "hub"
    home = tmp_path / "hf"
    env = _env(
        OPENMED_DEMO_CACHE_DIR=str(cache),
        HF_HOME=str(home),
        HF_HUB_CACHE=str(cache),
        **overrides,
    )
    return load_demo_config(environ=env, read_default_env_file=False)


def _client(tmp_path: Path, *, mlx: bool = False, **overrides: str):
    sdk = _FakeSdk(mlx=mlx)
    config = _config(tmp_path, **overrides)
    session = InferenceSession(config, sdk)
    app = create_app(config, session)
    return TestClient(app), sdk, session


def test_example_env_names_match_required_keys() -> None:
    example = (ROOT / "demo" / ".env.example").read_text(encoding="utf-8")
    parsed = parse_env_file(example)
    assert set(REQUIRED_ENV_NAMES) <= set(parsed)
    for name in REQUIRED_ENV_NAMES:
        assert parsed[name].strip()


def test_missing_env_and_non_loopback_fail_closed(tmp_path: Path) -> None:
    env = _env()
    env.pop("OPENMED_DEMO_API_PORT")
    with pytest.raises(ConfigError) as missing:
        load_demo_config(environ=env, read_default_env_file=False)
    assert "OPENMED_DEMO_API_PORT" in missing.value.code

    with pytest.raises(ConfigError) as bind:
        load_demo_config(
            environ=_env(OPENMED_DEMO_API_HOST="0.0.0.0"),
            read_default_env_file=False,
        )
    assert bind.value.code == "not_loopback:OPENMED_DEMO_API_HOST"
    with pytest.raises(ConfigError):
        assert_loopback_bind("0.0.0.0")

    with pytest.raises(ConfigError) as remote:
        load_demo_config(
            environ=_env(OPENMED_DEMO_BACKEND="remote"),
            read_default_env_file=False,
        )
    assert remote.value.code == "invalid_backend"

    with pytest.raises(ConfigError):
        load_demo_config(
            environ=_env(OPENMED_DEMO_CORS_ORIGIN="https://example.com"),
            read_default_env_file=False,
        )


def test_relative_cache_dir_resolves_under_demo() -> None:
    config = load_demo_config(environ=_env(), read_default_env_file=False)
    assert config.cache_dir.endswith(".cache/openmed-demo/hub")
    assert Path(config.cache_dir).is_absolute()
    assert config.api_host == "127.0.0.1"
    assert config.cors_origin == "http://127.0.0.1:5173"


def test_env_file_is_read_and_process_env_wins(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(f"{key}={value}" for key, value in _env().items())
        + "\nOPENMED_DEMO_API_PORT=1111\n",
        encoding="utf-8",
    )
    config = load_demo_config(
        environ={
            "OPENMED_DEMO_API_PORT": "2222",
            "VITE_OPENMED_DEMO_API_ORIGIN": "http://127.0.0.1:2222",
        },
        env_file=env_file,
        read_default_env_file=False,
    )
    assert config.api_port == 2222
    assert config.api_origin == "http://127.0.0.1:2222"


def test_routes_are_the_paste_and_catalog_surface(tmp_path: Path) -> None:
    client, _sdk, session = _client(tmp_path)
    with client:
        session.wait_for_background()
        assert route_paths(client.app) == sorted([*INFERENCE_ROUTES, *CATALOG_ROUTES])


def test_analyze_extract_and_deidentify_use_one_loader(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client, sdk, session = _client(tmp_path)
    note = f"Casey Example has asthma. {TOKEN}"
    with client:
        session.wait_for_background()
        with caplog.at_level(logging.DEBUG):
            analyze = client.post("/analyze", json={"text": note})
            extract = client.post("/pii/extract", json={"text": note})
            deidentify = client.post(
                "/pii/deidentify",
                json={"text": note, "method": "replace"},
            )
    assert analyze.status_code == 200
    assert extract.status_code == 200
    assert deidentify.status_code == 200
    assert analyze.json()["entities"][0]["label"] == "DISEASE"
    assert "mapping" not in analyze.json()
    assert "mapping" not in deidentify.json()
    assert "audit_report" not in deidentify.json()
    assert deidentify.json()["deidentified_text"].count("[NAME]") == 1
    assert TOKEN not in caplog.text
    loaders = {call[1]["loader"] for call in sdk.calls}
    assert loaders == {session.loader}
    for name, kwargs in sdk.calls:
        assert kwargs["cache_results"] is False
        assert name in {"analyze_text", "extract_pii", "deidentify"}
    assert sdk.calls[2][1]["method"] == "replace"
    assert sdk.calls[2][1]["keep_mapping"] is False
    assert sdk.calls[2][1]["audit"] is False


def test_validation_errors_do_not_echo_text(tmp_path: Path) -> None:
    client, sdk, session = _client(tmp_path)
    with client:
        session.wait_for_background()
        response = client.post(
            "/analyze",
            json={"text": TOKEN, "webhook": "http://127.0.0.1/hook"},
        )
        catalog = client.post("/models/pull", json={"model_name": "x", "text": TOKEN})
    assert response.status_code == 422
    assert response.json() == {"error": "invalid_request"}
    assert TOKEN not in response.text
    assert catalog.status_code == 422
    assert TOKEN not in catalog.text
    assert sdk.calls == []


def test_cors_allows_only_the_configured_origin(tmp_path: Path) -> None:
    client, _sdk, session = _client(tmp_path)
    headers = {
        "Origin": "http://127.0.0.1:5173",
        "Access-Control-Request-Method": "POST",
    }
    with client:
        session.wait_for_background()
        allowed = client.options("/analyze", headers=headers)
        blocked = client.options(
            "/analyze",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
    assert allowed.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
    assert "access-control-allow-origin" not in blocked.headers


def test_model_switch_reuses_loader_and_mlx_sibling(
    tmp_path: Path,
    restore_hub_env: None,
) -> None:
    client, sdk, session = _client(tmp_path, mlx=True)
    with client:
        session.wait_for_background()
        switched = client.post(
            "/models/active",
            json={
                "ner_model": "pharma_detection_electramed_33m",
                "pii_model": "pii_superclinical_small",
            },
        )
        session.wait_for_background()
        analyzed = client.post("/analyze", json={"text": "asthma follow up"})
    assert switched.status_code == 200
    assert switched.json()["active"]["ner_model"] == "pharma_detection_electramed_33m"
    assert analyzed.json()["model_name"] == "pharma_detection_electramed_33m"
    assert sdk.calls[-1][1]["loader"] is session.loader

    mlx_client, mlx_sdk, mlx_session = _client(tmp_path, mlx=True)
    with mlx_client:
        mlx_session.wait_for_background()
        mlx_result = mlx_client.post("/analyze", json={"text": "asthma today"})
    assert mlx_result.status_code == 200
    assert mlx_result.json()["model_name"] == "disease_detection_electramed_33m_mlx"
    assert mlx_sdk.loader is mlx_session.loader

    alias_client, _alias_sdk, alias_session = _client(
        tmp_path,
        OPENMED_DEMO_PII_MODEL="pii_superclinical_small_alias",
    )
    with alias_client:
        alias_session.wait_for_background()
        catalog = alias_client.get("/models").json()
    assert alias_session.pii_model == "pii_superclinical_small"
    assert catalog["active"]["pii_model"] == "pii_superclinical_small"
    assert all(
        item["key"] != "pii_superclinical_small_alias" for item in catalog["models"]
    )


def test_pull_downloads_once_and_marks_ready(
    tmp_path: Path,
    restore_hub_env: None,
) -> None:
    client, sdk, session = _client(tmp_path)
    with client:
        session.wait_for_background()
        first = client.post(
            "/models/pull",
            json={"model_name": "disease_detection_electramed_33m"},
        )
        session.wait_for_background()
        catalog = client.get("/models").json()
    assert first.json()["status"] == "downloading"
    row = next(
        item
        for item in catalog["models"]
        if item["key"] == "disease_detection_electramed_33m"
    )
    assert row["status"] == "ready"
    assert sdk.prefetched == ["disease_detection_electramed_33m"]
    assert sdk.loader.pipelines.count("disease_detection_electramed_33m") >= 1


def test_requests_do_not_write_note_files(tmp_path: Path) -> None:
    client, _sdk, session = _client(tmp_path)
    cache = Path(session.config.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    before = {path.relative_to(cache) for path in cache.rglob("*")}
    with client:
        session.wait_for_background()
        response = client.post(
            "/pii/deidentify", json={"text": f"Casey Example {TOKEN}", "method": "mask"}
        )
    after = {path.relative_to(cache) for path in cache.rglob("*")}
    assert response.status_code == 200
    assert after == before
    assert TOKEN not in response.json().get("mapping", {})


def test_apply_environment_disables_hf_telemetry(
    tmp_path: Path,
    restore_hub_env: None,
) -> None:
    config = _config(tmp_path, HF_HUB_OFFLINE="1", OPENMED_OFFLINE="1")
    apply_process_environment(config)
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["OPENMED_OFFLINE"] == "1"
    assert Path(config.cache_dir).is_dir()


def test_real_registry_includes_example_models() -> None:
    entries = RealSdk().list_models()
    by_key = {entry.key: entry for entry in entries}
    assert by_key["disease_detection_electramed_33m"].family == "NER"
    assert by_key["pii_superclinical_small"].category == "Privacy"
    assert by_key["pii_superclinical_small"].model_id.startswith("OpenMed/")


def test_demo_code_does_not_wrap_service_or_bind_all_interfaces() -> None:
    for path in (ROOT / "demo").rglob("*"):
        if "node_modules" in path.parts or "dist" in path.parts:
            continue
        if path.suffix not in {".py", ".ts", ".tsx"} or path.name.endswith(".test.ts"):
            continue
        text = path.read_text(encoding="utf-8")
        assert "openmed.service" not in text
        assert "0.0.0.0" not in text


def test_ui_copy_is_local_first_and_not_a_compliance_claim() -> None:
    readme = (ROOT / "demo" / "README.md").read_text(encoding="utf-8")
    app = (ROOT / "demo" / "web" / "src" / "App.tsx").read_text(encoding="utf-8")
    combined = readme + app
    assert "not a HIPAA compliance claim" in combined
    assert "HIPAA compliant" not in combined
    assert "Local ·" in (ROOT / "demo" / "web" / "src" / "statusLine.ts").read_text(
        encoding="utf-8"
    )
