import json
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from models.user import User
from services.auth_service import get_current_user
from services.session_service import get_session
from services.chat_service import ChatService

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


def parse_form_file_ids(raw_file_ids: Optional[str]) -> Optional[list[str]]:
    if raw_file_ids is None:
        return None

    try:
        parsed = json.loads(raw_file_ids)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="file_ids phải là JSON array.") from exc

    if not isinstance(parsed, list):
        raise HTTPException(status_code=400, detail="file_ids phải là JSON array.")

    return [str(file_id).strip() for file_id in parsed if str(file_id).strip()]


async def format_sse_stream(chunks):
    try:
        async for chunk in chunks:
            if not chunk:
                continue

            if isinstance(chunk, dict):
                event = chunk.get("event", "message")
                content = chunk.get("content", "")
                payload = json.dumps({"content": content}, ensure_ascii=False)
                if event == "message":
                    yield f"data: {payload}\n\n"
                else:
                    yield f"event: {event}\ndata: {payload}\n\n"
                continue

            payload = json.dumps({"content": chunk}, ensure_ascii=False)
            yield f"data: {payload}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as exc:
        payload = json.dumps({"error": str(exc)}, ensure_ascii=False)
        yield f"event: error\ndata: {payload}\n\n"


def get_chat_service(request: Request) -> ChatService:
    chat_service = getattr(request.app.state, "chat_service", None)
    if chat_service is None:
        raise HTTPException(status_code=503, detail="Chat service is not initialized")
    return chat_service


@router.post("/{session_id}", response_model=ChatResponse)
async def chat_with_ai(
    session_id: str,
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
):
    session = await get_session(session_id, current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    user_context = build_user_personalization_context(current_user, request.memory)

    if request.stream:
        chunks = chat_service.process_message_stream(
            session_id,
            request.message,
            user_id=current_user.id,
            file_ids=request.file_ids,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            think=request.think,
            user_context=user_context,
        )
        return StreamingResponse(
            format_sse_stream(chunks),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    response_text = await chat_service.process_message(
        session_id,
        request.message,
        user_id=current_user.id,
        file_ids=request.file_ids,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        think=request.think,
        user_context=user_context,
    )
    return ChatResponse(response=response_text)


@router.post("/{session_id}/files")
async def chat_with_ai_and_files(
    session_id: str,
    message: str = Form(...),
    stream: bool = Form(True),
    temperature: Optional[float] = Form(None),
    top_p: Optional[float] = Form(None),
    top_k: Optional[int] = Form(None),
    think: Optional[bool] = Form(None),
    memory: Optional[str] = Form(None),
    file_ids: Optional[str] = Form(None),
    files: list[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
):
    session = await get_session(session_id, current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not files:
        raise HTTPException(status_code=400, detail="Cần upload ít nhất một file PDF hoặc Markdown.")

    user_context = build_user_personalization_context(current_user, memory)
    selected_file_ids = parse_form_file_ids(file_ids)

    if stream:
        chunks = chat_service.process_message_stream(
            session_id,
            message,
            user_id=current_user.id,
            file_ids=selected_file_ids,
            uploaded_files=files,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            think=think,
            user_context=user_context,
        )
        return StreamingResponse(
            format_sse_stream(chunks),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    response_text = await chat_service.process_message(
        session_id,
        message,
        user_id=current_user.id,
        file_ids=selected_file_ids,
        uploaded_files=files,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        think=think,
        user_context=user_context,
    )
    return ChatResponse(response=response_text)
