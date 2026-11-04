from fastapi.testclient import TestClient

from tests.conftest import client


def test_health(client: TestClient):
    """Test the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_ask_return_grounded_answer(client):
    r = client.post("/ask", json={"question": "How much does the pro plan cost?"})
    assert r.status_code == 200
    body = r.json()
    assert "$12" in body["answer"]
    assert body["sources"], "answer must cite their sources"
    assert body["flagged"] is False

def test_unanswerable_question_is_refused(client):
    r = client.post("/ask", json={"question": "Is there a student discount for Pro?"})
    assert r.status_code == 200
    assert "don't know" in r.json()["answer"].lower()

def test_response_has_latency_and_cost(client):
    body = client.post("/ask", json={"question": "How long is the password reset link valid?"}).json()
    assert body["latency_ms"] >= 0 and body["cost_usd"] >= 0

def test_validation_rejects_bad_input(client):
    # missing field
    assert client.post("/ask", json={}).status_code == 422
    # too short
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    # too long
    assert client.post("/ask", json={"question": "x" * 501}).status_code == 422
    # blank
    assert client.post("/ask", json={"question": ""}).status_code == 422
    # wrong type
    assert client.post("/ask", json={"question": 12345}).status_code == 422
    # extra field
    assert client.post("/ask", json={"question": "valid question", "admin":
    True}).status_code == 422
    