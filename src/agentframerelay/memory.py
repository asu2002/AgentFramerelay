from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

_MEMORY_BACKENDS: dict[str, type["Memory"]] = {}


def register_memory_backend(name: str, backend: type["Memory"]) -> None:
    """Register a memory backend factory by name."""
    _MEMORY_BACKENDS[name.lower()] = backend


def get_memory_backend(name: str | None = None, **kwargs) -> "Memory":
    """Instantiate a supported memory backend by name."""
    backend_name = (name or "memory").lower()
    if backend_name in {"memory", "inmemory", "in_memory"}:
        return Memory(**kwargs)
    if backend_name in {"sqlite", "sqlite_memory", "sql"}:
        return SQLiteMemory(**kwargs)
    if backend_name in _MEMORY_BACKENDS:
        return _MEMORY_BACKENDS[backend_name](**kwargs)
    raise ValueError(f"Unknown memory backend: {name}")


@dataclass
class Memory:
    """Simple in-memory conversation history shared by the core Agent API."""

    messages: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_config(cls, backend: str | dict | None = None, **kwargs) -> "Memory":
        if backend is None:
            return cls(**kwargs)
        if isinstance(backend, cls):
            return backend
        if isinstance(backend, dict):
            backend_name = backend.get("type") or backend.get("backend") or "memory"
            config = {k: v for k, v in backend.items() if k not in {"type", "backend"}}
            config.update(kwargs)
            return get_memory_backend(backend_name, **config)
        return get_memory_backend(backend, **kwargs)

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


class SQLiteMemory(Memory):
    """Session-aware memory persisted to SQLite so history survives process restarts."""

    def __init__(self, *, path: str = ":memory:", session_id: str = "default"):
        self.path = path
        self.session_id = session_id
        self.messages: list[dict[str, Any]] = []
        self._connect()
        self._ensure_schema()
        self._load()

    def _connect(self) -> sqlite3.Connection:
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_memory (
                session_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                PRIMARY KEY (session_id, seq)
            )
            """
        )
        self._conn.commit()

    def _load(self) -> None:
        rows = self._conn.execute(
            "SELECT role, content FROM agent_memory WHERE session_id = ? ORDER BY seq ASC",
            (self.session_id,),
        ).fetchall()
        self.messages = [{"role": row["role"], "content": json.loads(row["content"])} for row in rows]

    def add_user_message(self, content: Any) -> dict[str, Any]:
        message = {"role": "user", "content": str(content)}
        self._append_message(message)
        return message

    def add_assistant_message(self, content: Any) -> dict[str, Any]:
        message = {"role": "assistant", "content": str(content)}
        self._append_message(message)
        return message

    def add_message(self, role: str, content: Any) -> dict[str, Any]:
        message = {"role": role, "content": str(content)}
        self._append_message(message)
        return message

    def _append_message(self, message: dict[str, Any]) -> None:
        seq = self._conn.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 FROM agent_memory WHERE session_id = ?",
            (self.session_id,),
        ).fetchone()[0]
        self._conn.execute(
            "INSERT INTO agent_memory (session_id, seq, role, content) VALUES (?, ?, ?, ?)",
            (self.session_id, seq, message["role"], json.dumps(message["content"])),
        )
        self._conn.commit()
        self.messages.append({"role": message["role"], "content": message["content"]})

    def clear(self) -> None:
        self._conn.execute("DELETE FROM agent_memory WHERE session_id = ?", (self.session_id,))
        self._conn.commit()
        self.messages.clear()

    def __bool__(self) -> bool:
        return True

    def __len__(self) -> int:
        return len(self.messages)


register_memory_backend("memory", Memory)
register_memory_backend("inmemory", Memory)
register_memory_backend("in_memory", Memory)
register_memory_backend("sqlite", SQLiteMemory)
register_memory_backend("sqlite_memory", SQLiteMemory)
register_memory_backend("sql", SQLiteMemory)
