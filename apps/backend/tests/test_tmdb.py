from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.metadata_service import TMDBMetadataService
from scripts.enrich_tmdb_metadata import enrich

ARTIFACTS = REPO_ROOT / "packages" / "recommendation-engine" / "artifacts"


class _FakeResponse:
    def __init__(self, status_code, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self):
        self.calls = []

    def get(self, url, headers=None, params=None):
        self.calls.append((url, headers, params))
        return _FakeResponse(
            200,
            {
                "poster_path": "/poster.jpg",
                "backdrop_path": "/backdrop.jpg",
            },
        )


def test_metadata_service_missing_file(tmp_path):
    service = TMDBMetadataService.load(tmp_path / "missing.json")
    assert service.size == 0
    assert service.enrich({"internal_movie_id": 1}).get("poster_path") is None


def test_metadata_service_enrich(tmp_path):
    path = tmp_path / "tmdb_metadata.json"
    path.write_text(
        json.dumps({
            "movies": {
                "12": {
                    "tmdb_id": 34,
                    "poster_path": "/a.jpg",
                    "backdrop_path": "/b.jpg",
                }
            }
        })
    )
    service = TMDBMetadataService.load(path)
    row = service.enrich({"internal_movie_id": 12})
    assert row["tmdb_id"] == 34
    assert row["poster_path"] == "/a.jpg"


def test_offline_enrichment_with_mock_client(tmp_path):
    if not ARTIFACTS.exists():
        return
    metadata = json.loads(
        (ARTIFACTS / "movie_metadata.json").read_text()
    )
    movie = metadata["movies"][0]
    source_id = movie["source_ids"]["_id"]
    csv_path = tmp_path / "catalogue.csv"
    csv_path.write_text(
        "_id,movie_id,tmdb_id\n"
        f"{source_id},{movie['movie_id']},12345\n"
    )
    output = tmp_path / "tmdb_metadata.json"
    client = _FakeClient()
    report = enrich(
        catalogue_csv=csv_path,
        artifacts_dir=ARTIFACTS,
        output=output,
        api_key="fake",
        client=client,
        request_delay=0,
    )
    assert report["matched"] == 1
    assert report["enriched"] == 1
    payload = json.loads(output.read_text())
    entry = payload["movies"][str(movie["internal_id"])]
    assert entry["tmdb_id"] == 12345
    assert entry["poster_path"] == "/poster.jpg"
    assert client.calls


def test_dry_run_does_not_call_api(tmp_path):
    if not ARTIFACTS.exists():
        return
    metadata = json.loads(
        (ARTIFACTS / "movie_metadata.json").read_text()
    )
    movie = metadata["movies"][0]
    csv_path = tmp_path / "catalogue.csv"
    csv_path.write_text(
        "_id,movie_id,tmdb_id\n"
        f"{movie['source_ids']['_id']},{movie['movie_id']},999\n"
    )
    client = _FakeClient()
    report = enrich(
        catalogue_csv=csv_path,
        artifacts_dir=ARTIFACTS,
        output=tmp_path / "out.json",
        api_key="fake",
        client=client,
        dry_run=True,
    )
    assert report["dry_run"] is True
    assert not client.calls
