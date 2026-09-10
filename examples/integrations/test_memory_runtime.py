"""Live memory sanity check for a selected runtime.

This example creates one Agent, passes a shared Memory instance, and then asks
follow-up questions to verify that session memory is active.

Usage:
    set AGENT_RUNTIME=crewai
    set AGENT_MODEL_PROVIDER=google
    set AGENT_MODEL=gemini-3.7-flash
    set AGENT_API_KEY=your_key_here
    python examples/integrations/test_memory_runtime.py
"""

import os

from _bootstrap import ensure_local_package

ensure_local_package()

from agentframerelay import Agent, Memory


RUNTIME = os.getenv("AGENT_RUNTIME", "litellm")
MODEL_PROVIDER = os.getenv("AGENT_MODEL_PROVIDER", "groq")
MODEL_NAME = os.getenv("AGENT_MODEL", "openai/gpt-oss-20b")
MODEL_API_KEY = os.getenv("groq_api_key") or os.getenv("GOOGLE_ADK_API_KEY")


agent = Agent(
    name="memory-check-agent",
    runtime=RUNTIME,
    memory=Memory(),
    model={
        "provider": MODEL_PROVIDER,
        "model": MODEL_NAME,
        "api_key": MODEL_API_KEY,
    },
    instructions=(
        "You are a short-term memory assistant. "
        "Use the previous conversation to answer follow-up questions. "
        "Be concise but remember facts from earlier in the same session."
    ),
)


print("=" * 70)
print("SESSION MEMORY CHECK")
print("=" * 70)
print("\nTurn 1: tell me your name and that you are learning Python.")
first = agent.run("My name is Asutosh and I am learning Python.")
print(first)

print("\nTurn 2: ask a memory-dependent question")
second = agent.run("What am I learning and what is my name?")
print(second)

print("\nTurn 3: ask another follow-up")
third = agent.run("What did I tell you earlier about myself?")
print(third)

print("\n" + "=" * 70)
print("MEMORY LOG")
print("=" * 70)
for i, message in enumerate(agent.memory.messages, 1):
    print(f"{i}. {message['role']}: {message['content']}")
