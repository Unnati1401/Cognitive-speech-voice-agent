"""Tiny provider-agnostic LLM client for the report writer.

Reads the API key from the environment (OPENAI_API_KEY / ANTHROPIC_API_KEY).
Heavy SDKs are imported lazily so the rest of the package never needs them.
"""
from __future__ import annotations


def complete(
    system: str,
    user: str,
    provider: str = "openai",
    model: str = "gpt-4o",
    temperature: float = 0.2,
    max_tokens: int = 1200,
) -> str:
    """Return the model's text completion for a system+user prompt."""
    if provider == "openai":
        from openai import OpenAI

        client = OpenAI()
        resp = client.chat.completions.create(
            model=model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""

    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=model, max_tokens=max_tokens, temperature=temperature,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    raise ValueError(f"Unknown LLM provider: {provider!r} (use 'openai' or 'anthropic').")
