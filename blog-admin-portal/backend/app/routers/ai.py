from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from .. import llm
from ..schemas import SuggestRequest

router = APIRouter(prefix="/ai", tags=["ai"])


@router.post("/suggest")
async def suggest(payload: SuggestRequest):
    return StreamingResponse(
        llm.stream(payload.body),
        media_type="text/plain; charset=utf-8",
    )
