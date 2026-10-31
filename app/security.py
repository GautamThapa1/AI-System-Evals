from collections import defaultdict, deque
import re
import threading
import time

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

def sanitize_question(text: str) -> str:
    """Remove control characters and collapse whitespace."""
    text = _CONTROL_CHARS.sub("", text) # replace control characters(x008, x0b etc) with empty string
    return re.sub(r"\s+", " ", text).strip() # replace multiple whitespace with single space

INJECTION_PATTERNS = [
    r"ignore (all |any |the |your )?(previous |prior |above |earlier )?(instructions|rules|prompts?|context)",
    r"disregard .{0,30}(instructions|rules|context|prompt)",
    r"(reveal|show|print|repeat|leak).{0,30}(system prompt|your instructions|your rules|your prompt)",
    r"\byou are now\b",
    r"\bdan\b",
    r"developer mode",
    r"pretend (that )?you",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

def detect_injection(text: str) -> bool:
    """Detect prompt injection attempts in the text. First line. Never to rely on this alone"""
    return any(p.search(text) for p in _COMPILED)

class RateLimiter:
    """Sliding-window limiter: at most `limit` requests per `window` seconds
per key."""

    def __init__(self, limit: int, window: float):
        self.limit, self.window = limit, window
        self._hits: dict[str, deque] = defaultdict(deque) # request history per key(deuble-ended queue)
        self._lock = threading.Lock()

    def allow(self, key:str) -> bool:
        now = time.monotonic() # current ticking in seconds 
        with self._lock:
            hits = self._hits[key]
            # gets the oldest as it is deque
            while hits and now - hits[0] > self.window: # remove old requests outside the window
                hits.popleft()
            if len(hits) < self.limit: # if current requests are less than limit, allow
                hits.append(now)
                return True
            return False
        
    def reset(self) -> None:
        """Reset the request history"""
        with self._lock:
            self._hits.clear()