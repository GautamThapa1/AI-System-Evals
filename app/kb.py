import re
from functools import lru_cache

from dataclasses import dataclass

from app.config import KB_PATH

STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "do", "does", "did", "i", "you", "my", "your", "me", "we",
    "our", "can", "could", "how", "what", "which", "who",
    "when", "where", "why", "of", "in", "on", "at", "to",
    "for", "with", "and", "or", "not", "it", "its", "this",
    "that", "by", "as", "if", "there", "any", "have", "has",
    "from", "about", "acme", "cloud"
}

# stemming method is used here: refuned -> refun, customer -> custo, etc.
def tokenize(text: str) -> list[str]:
    """Trim the word into 5 character"""
    abc = re.findall(r"[a-z0-9]+", text.lower())
    return [word[:5] for word in abc if word not in STOP and len(word) > 1]

@dataclass(frozen=True)
class Chunk:
    id: str
    text: str

@lru_cache(maxsize=1)
def load_chunks() -> tuple[Chunk, ...]:
    content = "\n".join(
        line
        for line in KB_PATH.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )
    paragraphs = [p.strip() for p in content.split("\n\n")]
    paragraphs = [p for p in paragraphs if p]
    return tuple(Chunk(f"kb-{i}",p) for i, p in enumerate(paragraphs, start=1))

def retrieve(question: str, k:int=2) -> list[Chunk]:
    """Return the top k chunks by number of shared tokens"""
    q = set(tokenize(question))
    scored = []
    for chunk in load_chunks():
        c = set(tokenize(chunk.text))
        score = len(q.intersection(c))
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:k]]