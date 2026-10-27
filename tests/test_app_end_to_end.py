"""The app against the real models.

Slow and worth it: everything else in the app's tests uses a fake pipeline, so
without these the app could quietly become a second implementation of the
measurement that disagrees with the one that was validated.

Run with: PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_app_end_to_end.py -q
Skipped automatically when the models or the tray are not on disk.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.app.server import create_app

TRAY = Path("data/interim/validation/lot_10.png")
MODELS = (Path("models/localiser-real.pt"), Path("models/classifier.session.pt"))

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not TRAY.exists() or not all(m.exists() for m in MODELS),
        reason="needs the trained models and a synthetic tray on disk",
    ),
]


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def test_the_app_reports_what_the_pipeline_reports(client):
    from src.pipeline import Pipeline

    direct = Pipeline().read(
        TRAY, lot_id="lot_10", temperature_c=25.0, moisture_pct_wb=14.0
    )

    response = client.post(
        "/api/lots",
        files={"photo": ("lot_10.png", TRAY.read_bytes(), "image/png")},
        data={"lot_id": "lot_10", "temperature_c": "25", "moisture_pct_wb": "14"},
    )
    assert response.status_code == 201
    body = response.json()

    assert body["kernels_counted"] == direct.kernels_counted
    assert body["damage_mass_pct"] == pytest.approx(
        direct.damage_mass_pct, abs=1e-6
    )


def test_the_measurement_lands_near_the_tray_s_known_damage(client):
    # lot_10 is 4.82% damaged by weight by construction.
    body = client.get("/api/lots").json()["lots"][0]
    assert body["damage_mass_pct"] == pytest.approx(4.82, abs=2.5)


def test_the_overlay_matches_the_photograph_s_dimensions(client):
    import io

    from PIL import Image

    response = client.get("/api/lots/lot_10/overlay.png")
    assert response.status_code == 200
    assert Image.open(io.BytesIO(response.content)).size == Image.open(TRAY).size
