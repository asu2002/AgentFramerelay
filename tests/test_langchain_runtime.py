import sys
from types import SimpleNamespace

import pytest

from agentframerelay.adapters.langgraph import _resolve_model
from agentframerelay.specs import ModelSpec


@pytest.mark.parametrize("provider", ["google", "google_ai", "gemini"])
def test_langchain_resolves_google_models_for_litellm(monkeypatch, provider):
    created = {}

    class ChatLiteLLM:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "langchain_litellm",
        SimpleNamespace(ChatLiteLLM=ChatLiteLLM),
    )

    _resolve_model(
        ModelSpec(
            provider=provider,
            model="gemini-3.6-flash",
            api_key="google-key",
            parameters={"temperature": 0},
        )
    )

    assert created == {
        "model": "gemini/gemini-3.6-flash",
        "api_key": "google-key",
        "temperature": 0,
    }