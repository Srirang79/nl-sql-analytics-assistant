"""Thin, provider-agnostic LLM clients. The engine only needs ``complete(system, messages) -> str``."""
from __future__ import annotations

import os
from typing import Protocol

from . import config


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, messages: list[dict], max_tokens: int = 1200) -> str: ...


class AnthropicClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        import anthropic  # lazy import: the app works in demo mode without the SDK key

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("No Anthropic API key. Set ANTHROPIC_API_KEY or paste it in the sidebar.")
        self._client = anthropic.Anthropic(api_key=key)
        self.model = model or os.environ.get("LLM_MODEL") or config.DEFAULT_MODELS["anthropic"]
        self.name = f"anthropic:{self.model}"

    def complete(self, system: str, messages: list[dict], max_tokens: int = 1200) -> str:
        resp = self._client.messages.create(
            model=self.model, system=system, messages=messages, max_tokens=max_tokens, temperature=0
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()


class OpenAIClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        import openai

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("No OpenAI API key. Set OPENAI_API_KEY or paste it in the sidebar.")
        self._client = openai.OpenAI(api_key=key)
        self.model = model or os.environ.get("LLM_MODEL") or config.DEFAULT_MODELS["openai"]
        self.name = f"openai:{self.model}"

    def complete(self, system: str, messages: list[dict], max_tokens: int = 1200) -> str:
        resp = self._client.chat.completions.create(
            model=self.model, temperature=0, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, *messages],
        )
        return (resp.choices[0].message.content or "").strip()


class DemoClient:
    """No-LLM demo mode: answers only the built-in example questions using their pre-written reference SQL.

    Lets anyone try the interface (validation, execution, charts) without an API key. It does NOT
    understand free-form questions, and the UI labels it clearly.
    """

    name = "demo (pre-written SQL, no LLM)"

    def __init__(self, examples: dict[str, str]):
        self.examples = {self._norm(q): sql for q, sql in examples.items()}

    @staticmethod
    def _norm(q: str) -> str:
        return " ".join(q.lower().split()).rstrip("?.! ")

    def complete(self, system: str, messages: list[dict], max_tokens: int = 1200) -> str:
        if "explain query results" in system.lower():  # explanation request: answer with a neutral templated summary
            return "Demo mode: results are shown in the table above. Connect an LLM in the sidebar for a written explanation."
        # the question is the latest user message that starts with "Question:" (repair messages come after it)
        question = next((m["content"] for m in reversed(messages) if m["role"] == "user" and m["content"].startswith("Question:")), "")
        q = self._norm(question.split("\n", 1)[0].replace("Question:", ""))
        sql = self.examples.get(q)
        if sql is None:
            return "UNANSWERABLE: Demo mode only supports the example questions. Add an API key in the sidebar to ask anything."
        return f"```sql\n{sql}\n```"


def make_client(provider: str, api_key: str | None = None, model: str | None = None) -> LLMClient:
    provider = provider.lower()
    if provider == "anthropic":
        return AnthropicClient(api_key, model)
    if provider == "openai":
        return OpenAIClient(api_key, model)
    raise ValueError(f"Unknown provider '{provider}'. Use 'anthropic' or 'openai'.")
