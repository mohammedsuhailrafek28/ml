"""In-memory report isolation, concurrency, and privacy tests."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import app
from src.api.settings import reload_settings
from src.utils.config import DISEASES

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((ROOT / "tests/fixtures/model_release_golden.json").read_text())


def _report(vector):
    disease = vector["disease_identifier"]
    response = TestClient(app).post(
        f"/api/v1/reports/{disease}",
        json={"measurements": vector["measurements"]},
    )
    return disease, vector, response


def test_parallel_reports_are_isolated_and_never_written(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("API_KEY", raising=False)
    reload_settings()
    monkeypatch.chdir(tmp_path)

    vectors = GOLDEN["vectors"]
    with ThreadPoolExecutor(max_workers=len(vectors)) as executor:
        responses = list(executor.map(_report, vectors))

    all_markers = {
        "liver": "Age: 45",
        "diabetes": "glucose: 148",
        "heart": "trestbps: 145",
        "kidney": "hemo: 15.4",
        "parkinsons": "PPE: 0.284654",
    }
    for disease, vector, response in responses:
        assert response.status_code == 200
        assert response.content.startswith(b"%PDF")
        assert response.headers["content-type"].startswith("application/pdf")
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["content-disposition"] == (
            f'attachment; filename="{disease}_report.pdf"'
        )
        text = response.content.decode("latin-1")
        assert f"Module: {DISEASES[disease].title}" in text
        assert f"Uncalibrated model score: {vector['expected_model_score']:.4f}" in text
        assert f"Decision threshold: {vector['threshold']:.4f}" in text
        assert "Release status:" in text
        assert "Known limitations:" in text
        assert "RESEARCH-USE DISCLAIMER:" in text
        assert all_markers[disease] in text
        for other, marker in all_markers.items():
            if other != disease:
                assert marker not in text

    assert list(tmp_path.rglob("*.pdf")) == []


def test_same_disease_parallel_inputs_do_not_cross_contaminate(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("API_KEY", raising=False)
    reload_settings()
    monkeypatch.chdir(tmp_path)
    base = next(v for v in GOLDEN["vectors"] if v["disease_identifier"] == "liver")
    vectors = [
        {**base, "measurements": {**base["measurements"], "Age": age}}
        for age in (41, 77)
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(_report, vectors))
    texts = [response.content.decode("latin-1") for _, _, response in responses]
    assert "Age: 41" in texts[0] and "Age: 77" not in texts[0]
    assert "Age: 77" in texts[1] and "Age: 41" not in texts[1]
    assert list(tmp_path.rglob("*.pdf")) == []


def test_failed_report_generation_leaves_no_partial_file(monkeypatch, tmp_path):
    import src.api.main as main_mod

    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("API_KEY", raising=False)
    reload_settings()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main_mod, "create_report", lambda *_args: (_ for _ in ()).throw(RuntimeError("boom")))
    vector = GOLDEN["vectors"][0]
    response = TestClient(app).post(
        "/api/v1/reports/liver",
        json={"measurements": vector["measurements"]},
    )
    assert response.status_code == 500
    assert list(tmp_path.rglob("*.pdf")) == []
