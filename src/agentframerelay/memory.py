from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Memory:
    """Simple in-memory conversation history shared by the core Agent API."""

    messages: list[dict[str, Any]] = field(default_factory=list)

    def add_user_message(self, content: Any) -> dict[str, Any]:
        message = {"role": "user", "content": str(content)}
        self.messages.append(message)
        return message

    def add_assistant_message(self, content: Any) -> dict[str, Any]:
        message = {"role": "assistant", "content": str(content)}
        self.messages.append(message)
        return message

    def add_message(self, role: str, content: Any) -> dict[str, Any]:
        message = {"role": role, "content": str(content)}
        self.messages.append(message)
        return message

    def clear(self) -> None:
        self.messages.clear()

    def __bool__(self) -> bool:
        return True

    def __len__(self) -> int:
        return len(self.messages)
