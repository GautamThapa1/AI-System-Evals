import pytest
from fastapi.testclient import TestClient
from app.main import app, limiter

@pytest.fixture(autouse=True)
def _offline_and_clean(monkeypatch):
    """Every test should run offline and with a fresh rate limiter state."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.delenv("MOCK_BEHAVIOR", raising=False)
    monkeypatch.delenv("API_TOKEN", raising=False)
    limiter.reset()
    yield
    limiter.reset()

@pytest.fixture
def client() -> TestClient:
    """A test client for the FastAPI app."""
    return TestClient(app)