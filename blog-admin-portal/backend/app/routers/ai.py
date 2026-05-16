import json
import os
from typing import AsyncIterator

import httpx
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..schemas import SuggestRequest

router = APIRouter(prefix="/ai", tags=["ai"])

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

SYSTEM_PROMPT = (
    "You are a helpful blog writing assistant. Continue the following blog "
    "post naturally, matching the author's tone and style. Return only the "
    "continuation text, no preamble."
)


async def _stream_ollama(body: str) -> AsyncIterator[bytes]:
    payload = {
        "model": OLLAMA_MODEL,
        "stream": True,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": body or "(The post is empty — start it.)"},
        ],
    }
    timeout = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            async with client.stream(
                "POST", f"{OLLAMA_URL}/api/chat", json=payload
            ) as response:
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
                    message = chunk.get("message") or {}
                    content = message.get("content", "")
                    if content:
                        yield content.encode("utf-8")
                    if chunk.get("done"):
                        break
        except httpx.RequestError as exc:
            yield f"[Failed to reach Ollama at {OLLAMA_URL}: {exc}]".encode()


@router.post("/suggest")
async def suggest(payload: SuggestRequest):
    return StreamingResponse(
        _stream_ollama(payload.body),
        media_type="text/plain; charset=utf-8",
    )
