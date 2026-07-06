import json
import os
from typing import AsyncIterator

import httpx

SYSTEM_PROMPT = (
    "You are a helpful blog writing assistant. Continue the following blog "
    "post naturally, matching the author's tone and style. Return only the "
    "continuation text, no preamble."
)


def _user_message(body: str) -> str:
    return body or "(The post is empty — start it.)"


async def stream_ollama(body: str) -> AsyncIterator[bytes]:
    url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    payload = {
        "model": model,
        "stream": True,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_message(body)},
        ],
    }
    timeout = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            async with client.stream("POST", f"{url}/api/chat", json=payload) as response:
                if response.status_code != 200:
                    detail = await response.aread()
                    yield f"[Ollama error {response.status_code}: {detail.decode(errors='replace')}]".encode()
                    return
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    content = (chunk.get("message") or {}).get("content", "")
                    if content:
                        yield content.encode("utf-8")
                    if chunk.get("done"):
                        break
        except httpx.RequestError as exc:
            yield f"[Failed to reach Ollama at {url}: {exc}]".encode()


async def stream_anthropic(body: str) -> AsyncIterator[bytes]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        yield b"[ANTHROPIC_API_KEY is not set]"
        return

    from anthropic import AsyncAnthropic

    model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", "1024"))

    client = AsyncAnthropic(api_key=api_key)
    try:
        async with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _user_message(body)}],
        ) as stream:
            async for text in stream.text_stream:
                if text:
                    yield text.encode("utf-8")
    except Exception as exc:  # noqa: BLE001 — surface any provider error to the client
        yield f"[Anthropic error: {exc}]".encode()


def stream(body: str) -> AsyncIterator[bytes]:
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider == "anthropic":
        return stream_anthropic(body)
    return stream_ollama(body)
