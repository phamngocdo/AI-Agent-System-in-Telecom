import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llm_serve import TelcoLLM
from models.user import User
from services.auth_service import get_current_user
from services.session_service import get_session
from services.chat_service import process_chat_message, process_chat_message_stream

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    stream: bool = False
    file_ids: Optional[list[str]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    think: Optional[bool] = None
    memory: Optional[str] = None

class ChatResponse(BaseModel):
    response: str

def build_user_personalization_context(current_user: User, request_memory: Optional[str]) -> Optional[str]:
    lines = []
    full_name = (current_user.full_name or "").strip()
    saved_context = (current_user.personal_context or "").strip()
    request_context = (request_memory or "").strip()

    if full_name:
        lines.append(f"Display name: {full_name}")
    if saved_context:
        lines.append(f"Saved user preferences: {saved_context}")
    if request_context and request_context != saved_context:
        lines.append(f"Current request preferences: {request_context}")

    return "\n".join(lines) if lines else None

async def format_sse_stream(chunks):
    try:
        async for chunk in chunks:
            if chunk:
                payload = json.dumps({"content": chunk}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as exc:
        payload = json.dumps({"error": str(exc)}, ensure_ascii=False)
        yield f"event: error\ndata: {payload}\n\n"

def get_telco_llm(request: Request) -> TelcoLLM:
    telco_llm = getattr(request.app.state, "telco_llm", None)
    if telco_llm is None:
        raise HTTPException(status_code=503, detail="TelcoLLM is not initialized")
    return telco_llm

@router.post("/{session_id}", response_model=ChatResponse)
async def chat_with_ai(
    session_id: str,
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    telco_llm: TelcoLLM = Depends(get_telco_llm),
):
    # Verify ownership
    session = await get_session(session_id, current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    user_context = build_user_personalization_context(current_user, request.memory)

    if request.stream:
        chunks = process_chat_message_stream(
            telco_llm, session_id, request.message,
            file_ids=request.file_ids,
            temperature=request.temperature, top_p=request.top_p, top_k=request.top_k,
            think=request.think, user_context=user_context
        )
        return StreamingResponse(
            format_sse_stream(chunks),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        response_text = await process_chat_message(
            telco_llm, session_id, request.message,
            file_ids=request.file_ids,
            temperature=request.temperature, top_p=request.top_p, top_k=request.top_k,
            think=request.think, user_context=user_context
        )
        return ChatResponse(response=response_text)
