import tiktoken


class TokenBudgetManager:
    """Tracks token usage per session and rejects oversized/over-budget requests.
    Currently in-memory — we'll move this to Redis in Phase 3 so it survives restarts
    and works across multiple server workers."""

    def __init__(self, max_tokens_per_request: int, max_tokens_per_session: int):
        self.max_tokens_per_request = max_tokens_per_request
        self.max_tokens_per_session = max_tokens_per_session
        self._session_usage: dict[str, int] = {}
        self._encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def check_and_consume(self, session_id: str, text: str) -> tuple[bool, str]:
        tokens = self.count_tokens(text)

        if tokens > self.max_tokens_per_request:
            return False, f"Request too long ({tokens} tokens, limit is {self.max_tokens_per_request})."

        used = self._session_usage.get(session_id, 0)
        if used + tokens > self.max_tokens_per_session:
            return False, "Session token budget exceeded. Please start a new session."

        self._session_usage[session_id] = used + tokens
        return True, ""