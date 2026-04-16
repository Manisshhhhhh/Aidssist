from __future__ import annotations


class MemoryEngine:
    def __init__(self, max_history: int = 20):
        self.history: list[dict[str, str]] = []
        self.max_history = max(1, int(max_history))

    def add_entry(self, query, code, result):
        self.history.append(
            {
                "query": str(query),
                "code": str(code),
                "result": str(result)[:500],
            }
        )
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

    def get_context(self):
        return self.history[-5:]

    def clear(self) -> None:
        self.history.clear()


__all__ = ["MemoryEngine"]
