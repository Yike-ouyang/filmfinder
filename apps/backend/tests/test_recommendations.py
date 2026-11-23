from __future__ import annotations

import io
import json
import zipfile

import pytest

from filmfinder.letterboxd_import import load_letterboxd_profile

from app.services.recommendation_service import RecommendationService


def _upload(client, path):
    with open(path, "rb") as handle:
        return client.post(
            "/api/v1/recommendations/from-letterboxd?limit=100",
            files={"file": ("profile.zip", handle, "application/zip")},
        )


class RecordingAnalytics:
    def __init__(self):
        self.calls = []

    def record_letterboxd_import(self, db, user_id, now=None):
        del db
        self.calls.append((user_id, now))


@pytest.fixture(autouse=True)
def _replace_analytics(client):
    previous = client.app.state.analytics_service
    fake = RecordingAnalytics()
    client.app.state.analytics_service = fake
    yield fake
    client.app.state.analytics_service = previous


def _zip_bytes(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def test_valid_letterboxd_upload(client, letterboxd_zip):
    response = _upload(client, letterboxd_zip)
    assert response.status_code == 200
    payload = response.json()
    assert payload["profile"]["ratings_matched"] >= 65
    assert payload["recommendation_info"]["requested_count"] == 100
    assert payload["recommendation_info"]["returned_count"] == 100
    assert len(payload["recommendations"]) == 100
    profile = load_letterboxd_profile(
        letterboxd_zip.open("rb"),
        movie_metadata=client.app.state.engine.store.movies,
    )
    returned = {
        row["movie_id"] for row in payload["recommendations"]
    }
    assert returned.isdisjoint(profile.ratings)
    assert returned.isdisjoint(profile.watched)
    scores = [row["score"] for row in payload["recommendations"]]
    assert scores == sorted(scores, reverse=True)
    assert all(
        row["score_type"] == "m5_predicted_rating"
        for row in payload["recommendations"]
    )
    assert all("internal_movie_id" in row for row in payload["recommendations"])


def test_invalid_zip(client):
    response = client.post(
        "/api/v1/recommendations/from-letterboxd",
        files={"file": ("bad.zip", b"not a zip", "application/zip")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_zip"


def test_missing_file(client):
    response = client.post(
        "/api/v1/recommendations/from-letterboxd"
    )
    assert response.status_code == 422


def test_limit_above_pool(client, letterboxd_zip):
    with open(letterboxd_zip, "rb") as handle:
        response = client.post(
            "/api/v1/recommendations/from-letterboxd?limit=101",
            files={"file": ("profile.zip", handle, "application/zip")},
        )
    assert response.status_code == 422


def test_empty_ratings(client):
    data = _zip_bytes({
        "ratings.csv": "Date,Name,Year,Letterboxd URI,Rating\n",
        "watched.csv": "Date,Name,Year,Letterboxd URI\n",
    })
    response = client.post(
        "/api/v1/recommendations/from-letterboxd",
        files={"file": ("empty.zip", data, "application/zip")},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "no_matched_ratings"


def test_malformed_ratings(client):
    data = _zip_bytes({
        "ratings.csv": "bad,header\nnot,a,letterboxd,file\n",
    })
    response = client.post(
        "/api/v1/recommendations/from-letterboxd",
        files={"file": ("bad-csv.zip", data, "application/zip")},
    )
    assert response.status_code == 422


def test_import_records_analytics_event(
    client, letterboxd_zip, _replace_analytics
):
    response = _upload(client, letterboxd_zip)
    assert response.status_code == 200
    assert _replace_analytics.calls == [(None, None)]


def test_deterministic_response(client, letterboxd_zip):
    first = _upload(client, letterboxd_zip).json()
    second = _upload(client, letterboxd_zip).json()
    assert first == second


def test_upload_size_limit(client, monkeypatch, letterboxd_zip):
    service = client.app.state.recommendation_service
    monkeypatch.setattr(service.settings, "max_upload_size_mb", 0)
    response = _upload(client, letterboxd_zip)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "upload_too_large"
