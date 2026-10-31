
import os
import re
import time
from dataclasses import dataclass

from app.config import LLM_TIMEOUT_S, REFUSAL
from app.kb import tokenize


@dataclass(frozen=True)
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency: float
    cost: float


def _cost(input_tokens: int, output_tokens: int) -> float:
    # Example OpenAI API prices for GPT-6.1 Sol, in USD per 1M tokens.
    # Update these to use different model.
    price_in = float(os.getenv("OPENAI_INPUT_PRICE_PER_MTOK", "2.00"))
    price_out = float(os.getenv("OPENAI_OUTPUT_PRICE_PER_MTOK", "10.00"))

    return (
        input_tokens * price_in
        + output_tokens * price_out
    ) / 1_000_000

_client = None

def _call_openai(system: str, user: str, model: str, max_tokens: int):
    global _client

    if _client is None:
        from openai import OpenAI

        # Reads OPENAI_API_KEY from the environment.
        # Timeout + retries improve reliability.
        _client = OpenAI(
            timeout=LLM_TIMEOUT_S,
            max_retries=2,  
        )

    resp = _client.chat.completions.create(
        model=model,
        max_completion_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    text = resp.choices[0].message.content or ""

    return (
        text,
        resp.usage.prompt_tokens,
        resp.usage.completion_tokens,
    )

def _mock_answer(user: str) -> str:
    """It retuns the context whcih is most similar to question"""
    ctx = re.search(r"<Context>(.*?)</Context>", user, re.IGNORECASE | re.DOTALL)
    qst = re.search(r"<Question>(.*?)</Question>", user, re.IGNORECASE | re.DOTALL)
    if not ctx or not qst:
        return REFUSAL # from config.py
    sloppy = os.getenv("MOCK_BEHAVIOUR") == "sloppy"
    if sloppy:
        obey = re.search(r"(?:reply only with the word|say)(\w+)", qst.group(1), re.IGNORECASE) # group(1) means the text inside the <c>...</c>
        if obey:
            return obey.group(1).upper()
    q_tokens = set(tokenize(qst.group(1))) # set to remove the duplicates
    best, best_score = None, 0
    for sentence in re.split(r"(?<=[.!?])\s+", ctx.group(1).strip()):
        score = len(q_tokens & set(tokenize(sentence)))
        if score > best_score:
            best, best_score = sentence.strip(), score
    if best and best_score >= 2: # at least two matching words
        return best
    if sloppy:
        return "Yes, that is supported on all plans for $99 per month."
    else:
        return REFUSAL


def generate(system: str,user: str,model: str | None = None,max_tokens: int = 300,) -> LLMResult:
    provider = os.getenv("LLM_PROVIDER", "mock")
    start = time.perf_counter()

    if provider == "openai":
        model = model or os.getenv("MODEL", "gpt-4o-mini")
        text, tin, tout = _call_openai(
            system, user, model, max_tokens
        )

    elif provider == "mock":
        text = _mock_answer(user)
        tin, tout = (
            (len(system) + len(user)) // 4,
            len(text) // 4,
        )

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")

    latency_ms = (time.perf_counter() - start) * 1000

    return LLMResult(
        text, tin, tout, latency_ms, _cost(tin, tout)
    )