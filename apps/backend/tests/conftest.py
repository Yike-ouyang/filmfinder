from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ZIP = (
    REPO_ROOT
    / "packages"
    / "recommendation-engine"
    / "tests"
    / "fixtures"
    / "letterboxd-eslpej-2026-11-20-10-33-utc.zip"
)
ARTIFACTS = (
    REPO_ROOT / "packages" / "recommendation-engine" / "artifacts"
)


@pytest.fixture(scope="module")
def client():
    if not ARTIFACTS.exists():
        pytest.skip("Recommendation artifacts are unavailable.")
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def letterboxd_zip():
    if not FIXTURE_ZIP.exists():
        pytest.skip("Letterboxd fixture is unavailable.")
    return FIXTURE_ZIP

