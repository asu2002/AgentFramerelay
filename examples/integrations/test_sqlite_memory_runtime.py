"""SQLite-backed persistent session memory example.

Run with an actual runtime such as LiteLLM or OpenAI by setting the env vars below.
This example persists the session across process restarts by using SQLiteMemory.

Example:
    $env:AGENT_RUNTIME='litellm'
    $env:AGENT_MODEL_PROVIDER='openai'
    $env:AGENT_MODEL='gpt-4.1-mini'
    $env:OPENAI_API_KEY='your_key_here'
    python .\examples\integrations\test_sqlite_memory_runtime.py
"""

import os

from _bootstrap import ensure_local_package

ensure_local_package()

from agentframerelay import Agent
from agentframerelay.memory import SQLiteMemory

RUNTIME = os.getenv("AGENT_RUNTIME", "litellm")
MODEL_PROVIDER = os.getenv("AGENT_MODEL_PROVIDER", "openai")
MODEL_NAME = os.getenv("AGENT_MODEL", "gpt-4.1-mini")
MODEL_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("LITELLM_API_KEY") or os.getenv("GOOGLE_ADK_API_KEY")
SESSION_ID = os.getenv("AGENT_SESSION_ID", "demo-session")
DB_PATH = os.getenv("MEMORY_DB_PATH", os.path.join(os.path.dirname(__file__), "memory.sqlite3"))

memory = SQLiteMemory(path=DB_PATH, session_id=SESSION_ID)

agent = Agent(
    name="sqlite-memory-agent",
    runtime=RUNTIME,
    memory=memory,
    model={
        "provider": MODEL_PROVIDER,
        "model": MODEL_NAME,
        "api_key": MODEL_API_KEY,
    },
    instructions=(
        "You are a persistent memory assistant. "
        "Use the previous conversation from this session to answer follow-up questions. "
        "Keep answers brief and grounded in the stored session history."
    ),
)

print("=" * 70)
print("SQLITE MEMORY SESSION")
print("=" * 70)
print(f"Session ID: {SESSION_ID}")
print(f"Database: {DB_PATH}\n")

print("Turn 1: My name is Asutosh and I am learning Python.")
first = agent.run("My name is Asutosh and I am learning Python.")
print(first)

print("\nTurn 2: What am I learning and what is my name?")
second = agent.run("What am I learning and what is my name?")
print(second)

print("\nTurn 3: What did I tell you earlier about myself?")
third = agent.run("What did I tell you earlier about myself?")
print(third)

print("\n" + "=" * 70)
print("MEMORY LOG")
print("=" * 70)
for i, message in enumerate(agent.memory.messages, 1):
    print(f"{i}. {message['role']}: {message['content']}")

print("\nThis history is persisted to SQLite and can be reloaded in a later process with the same session_id.")
