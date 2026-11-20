from __future__ import annotations


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready(client):
    response = client.get("/api/v1/ready")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["model"] == "M5"
    assert payload["recommendation_algorithm"] == "knn_m5"
