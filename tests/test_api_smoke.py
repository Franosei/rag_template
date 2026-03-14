"""Smoke tests for the FastAPI application."""

from fastapi.testclient import TestClient

from src.api.server import app


def test_health_endpoint() -> None:
    """The health endpoint should respond successfully."""

    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


def test_query_endpoint_returns_answer() -> None:
    """The query endpoint should return an answer and citations."""

    with TestClient(app) as client:
        response = client.post(
            "/api/query",
            json={"question": "What is an estimand?"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert isinstance(payload["citations"], list)
    assert "retrieval_diagnostics" in payload
    assert payload["retrieval_diagnostics"]["strategy"] in {"hybrid_plus_graph", "semantic_hybrid_graph"}
