import hmac
import os

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from app.config import MAX_QUESTION_CHARS, RATE_LIMIT_PER_MIN, RATE_LIMIT_WINDOW
from app.rag import answer_question
from app.security import RateLimiter

app = FastAPI(title="DocQA", version="1.0")
limiter = RateLimiter(RATE_LIMIT_PER_MIN, RATE_LIMIT_WINDOW)  # Initialize the rate limiter with the configured limit and window

class AskRequest(BaseModel):
    model_config = {"extra": "forbid"} # built-in validator to forbid extra fields in the request body
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS) # ... means required field

    @field_validator("question")
    def validate_question(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question cannot be empty or whitespace")
        return v.strip()

class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    flagged: bool
    latency_ms: float
    cost_usd: float


def check_api_key(x_api_key: str | None = Header(default=None),) -> None:
    """Require a matching API key only if API_TOKEN is configured."""
    expected = os.getenv("API_TOKEN")

    if expected and not hmac.compare_digest(
        x_api_key or "", expected
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
        )

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse, dependencies=[Depends(check_api_key)])
def ask(body: AskRequest, request: Request):
    client = request.client.host if request.client else "unknown"
    if not limiter.allow(client):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {RATE_LIMIT_PER_MIN} requests per minute",
        )
    result = answer_question(body.question)
    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        flagged=result["flagged"],
        latency_ms=result["latency_ms"],
        cost_usd=result["cost_usd"],
    )
