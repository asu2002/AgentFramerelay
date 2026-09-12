"""Restart demo for SQLite session memory.

This script shows the expected behavior:
1. create a session memory and ask a question
2. simulate a restart by creating a new Agent with the same DB and same session_id
3. ask a follow-up question and confirm the earlier memory is reloaded

Usage:
    $env:AGENT_RUNTIME='litellm'
    $env:AGENT_MODEL_PROVIDER='openai'
    $env:AGENT_MODEL='gpt-4.1-mini'
    $env:OPENAI_API_KEY='your_key_here'
    python .\examples\integrations\test_sqlite_memory_restart_demo.py
"""

import os

from _bootstrap import ensure_local_package

ensure_local_package()

from agentframerelay import Agent
from agentframerelay.memory import SQLiteMemory

RUNTIME = os.getenv("AGENT_RUNTIME", "langchain")
MODEL_PROVIDER = os.getenv("AGENT_MODEL_PROVIDER", "groq")
MODEL_NAME = os.getenv("AGENT_MODEL", "openai/gpt-oss-20b")
MODEL_API_KEY = os.getenv("groq_api_key") or os.getenv("GOOGLE_ADK_API_KEY")
SESSION_ID = os.getenv("AGENT_SESSION_ID", "restart-demo-session")
DB_PATH = os.getenv("MEMORY_DB_PATH", os.path.join(os.path.dirname(__file__), "restart_demo.sqlite3"))

print("=" * 70)
print("SQLite session memory restart demo")
print("=" * 70)
print(f"DB path: {DB_PATH}")
print(f"Session ID: {SESSION_ID}\n")

# First run: create the memory and ask a fact that should be remembered.
memory_one = SQLiteMemory(path=DB_PATH, session_id=SESSION_ID)
agent_one = Agent(
    name="restart-demo-agent",
    runtime=RUNTIME,
    memory=memory_one,
    model={
        "provider": MODEL_PROVIDER,
        "model": MODEL_NAME,
        "api_key": MODEL_API_KEY,
    },
    instructions=(
        "You are a memory assistant. Use the session's previous messages to answer follow-up questions."
    ),
)

print("Step 1: first run")
print("User: My name is Asutosh and I am learning Python.")
print(agent_one.run("My name is Asutosh and I am learning Python."))

print("\nSimulating restart: new memory instance loads the saved session...")

# Simulate restart by creating a new Agent with the same DB + same session_id.
memory_two = SQLiteMemory(path=DB_PATH, session_id=SESSION_ID)
agent_two = Agent(
    name="restart-demo-agent",
    runtime=RUNTIME,
    memory=memory_two,
    model={
        "provider": MODEL_PROVIDER,
        "model": MODEL_NAME,
        "api_key": MODEL_API_KEY,
    },
    instructions=(
        "You are a memory assistant. Use the session's previous messages to answer follow-up questions."
    ),
)

print("Step 2: reloaded session from SQLite")
print("Existing messages loaded from DB:")
for idx, message in enumerate(agent_two.memory.messages, 1):
    print(f"  {idx}. {message['role']}: {message['content']}")

print("\nUser: What am I learning and what is my name?")
print(agent_two.run("What am I learning and what is my name?"))

print("\nUser: What did I tell you earlier about myself?")
print(agent_two.run("What did I tell you earlier about myself?"))

print("\n" + "=" * 70)
print("Final stored memory:")
for idx, message in enumerate(agent_two.memory.messages, 1):
    print(f"{idx}. {message['role']}: {message['content']}")
