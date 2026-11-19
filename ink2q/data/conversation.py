from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Conversation:
    id: str
    source: str
    host: str
    messages: list[dict] = field(default_factory=list)
    group: str = ""

    def __post_init__(self):
        if not self.group:
            self.group = self.id

    def has_assistant(self) -> bool:
        return any(m["role"] == "assistant" for m in self.messages)
