from typing import List
from fastapi import APIRouter, Depends, HTTPException, Response

from models.user import User
from models.session import ChatFile, ChatSession, ChatSessionCreate, ChatSessionUpdate, ChatMessage
from services.auth_service import get_current_user
from services.session_service import (
    get_user_sessions,
    create_session,
    get_session,
    update_session,
    delete_session,
    get_session_files,
    get_session_messages,
)

router = APIRouter()

@router.get("/", response_model=List[ChatSession])
async def read_user_sessions(current_user: User = Depends(get_current_user)):
    sessions = await get_user_sessions(current_user.id)
    return sessions

@router.post("/", response_model=ChatSession)
async def create_new_session(session: ChatSessionCreate, current_user: User = Depends(get_current_user)):
    return await create_session(current_user.id, session)

@router.get("/{session_id}", response_model=ChatSession)
async def read_session(session_id: str, current_user: User = Depends(get_current_user)):
    session = await get_session(session_id, current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@router.put("/{session_id}", response_model=ChatSession)
async def update_existing_session(
    session_id: str,
    session_update: ChatSessionUpdate,
    current_user: User = Depends(get_current_user),
):
    session = await update_session(session_id, current_user.id, session_update)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@router.delete("/{session_id}", status_code=204)
async def delete_existing_session(session_id: str, current_user: User = Depends(get_current_user)):
    deleted = await delete_session(session_id, current_user.id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return Response(status_code=204)

@router.get("/{session_id}/messages", response_model=List[ChatMessage])
async def read_session_messages(session_id: str, current_user: User = Depends(get_current_user)):
    session = await get_session(session_id, current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = await get_session_messages(session_id)
    return messages

@router.get("/{session_id}/files", response_model=List[ChatFile])
async def read_session_files(session_id: str, current_user: User = Depends(get_current_user)):
    session = await get_session(session_id, current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    files = await get_session_files(session_id, current_user.id)
    return files
