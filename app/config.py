from pathlib import Path

KB_PATH = Path(__file__).parent.parent / "data" / "knowledge_base.md"
MAX_QUESTION_CHARS = 500
RATE_LIMIT_PER_MIN = 30
RATE_LIMIT_WINDOW = 60.0
LLM_TIMEOUT_S = 20
REFUSAL = "I don't know based on the provided documents."
BLOCKED = "I can't help with that request. Please ask a question about Acme Cloud."
