"""API integration tests for GitContext FastAPI backend."""
from fastapi.testclient import TestClient
from rag_project.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_root_serves_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "GitContext" in response.text


def test_ingest_and_query_flow():
    # Ingest current directory
    ingest_resp = client.post("/ingest", json={"repo_path": "."})
    assert ingest_resp.status_code == 200
    ingest_data = ingest_resp.json()
    assert ingest_data["status"] == "success"
    assert ingest_data["chunks_indexed"] > 0

    # Query for GitMiner
    query_resp = client.post("/query", json={"question": "Where is GitMiner defined?", "top_k": 3})
    assert query_resp.status_code == 200
    query_data = query_resp.json()
    assert len(query_data["results"]) > 0
    assert any("GitMiner" in r["text"] for r in query_data["results"])

    # Ask endpoint (with citations)
    ask_resp = client.post("/ask", json={"question": "Where is GitMiner defined?", "top_k": 3})
    assert ask_resp.status_code == 200
    ask_data = ask_resp.json()
    assert "answer" in ask_data
    assert len(ask_data["citations"]) > 0


def test_sync_endpoint():
    response = client.post("/sync")
    assert response.status_code == 200


def test_evaluate_endpoint():
    response = client.get("/evaluate?top_k=3")
    assert response.status_code == 200
    data = response.json()
    assert "recall_at_3_percent" in data
    assert "avg_latency_ms" in data
