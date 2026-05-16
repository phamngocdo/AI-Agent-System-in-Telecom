import logging
from typing import List, Optional
from datetime import datetime
from random import randint

from bson import ObjectId
from database import get_database
from llm_serve.file_rag.qdrant_store import QdrantChunkStore
from models.session import ChatSessionCreate, ChatMessageCreate, ChatSessionUpdate

logger = logging.getLogger("telcollm.sessions")

async def _get_unique_session_title(
    db,
    user_id: str,
    title: str,
    exclude_session_id: Optional[str] = None,
) -> str:
    base_title = title.strip() if title and title.strip() else "Hội thoại"
    candidate = base_title

    query = {"user_id": user_id, "title": candidate}
    if exclude_session_id:
        query["_id"] = {"$ne": ObjectId(exclude_session_id)}

    for _ in range(100):
        query["title"] = candidate
        existing = await db["sessions"].find_one(query)
        if existing is None:
            return candidate
        candidate = f"{base_title} {randint(1000, 9999)}"

    return f"{base_title} {datetime.utcnow().strftime('%H%M%S%f')[-8:]}"

async def create_session(user_id: str, session: ChatSessionCreate) -> dict:
    db = get_database()
    now = datetime.utcnow()
    session_dict = session.model_dump()
    session_dict["title"] = await _get_unique_session_title(db, user_id, session_dict.get("title", ""))
    session_dict.update({
        "user_id": user_id,
        "file_ids": [],
        "created_at": now,
        "updated_at": now
    })
    result = await db["sessions"].insert_one(session_dict)
    session_dict["_id"] = str(result.inserted_id)
    return session_dict

async def get_user_sessions(user_id: str) -> List[dict]:
    db = get_database()
    cursor = db["sessions"].find({"user_id": user_id}).sort("updated_at", -1)
    sessions = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        doc["file_ids"] = doc.get("file_ids") or []
        sessions.append(doc)
    return sessions

async def get_session(session_id: str, user_id: str) -> Optional[dict]:
    db = get_database()
    try:
        session = await db["sessions"].find_one({"_id": ObjectId(session_id), "user_id": user_id})
        if session:
            session["_id"] = str(session["_id"])
            session["file_ids"] = session.get("file_ids") or []
        return session
    except Exception:
        return None

async def update_session(
    session_id: str,
    user_id: str,
    session_update: ChatSessionUpdate,
) -> Optional[dict]:
    db = get_database()
    try:
        object_id = ObjectId(session_id)
        existing = await db["sessions"].find_one({"_id": object_id, "user_id": user_id})
        if existing is None:
            return None

        now = datetime.utcnow()
        title = await _get_unique_session_title(
            db,
            user_id,
            session_update.title,
            exclude_session_id=session_id,
        )
        await db["sessions"].update_one(
            {"_id": object_id, "user_id": user_id},
            {"$set": {"title": title, "updated_at": now}},
        )
        return await get_session(session_id, user_id)
    except Exception:
        return None

async def delete_session(session_id: str, user_id: str) -> bool:
    db = get_database()
    try:
        object_id = ObjectId(session_id)
        existing = await db["sessions"].find_one({"_id": object_id, "user_id": user_id}, {"_id": 1})
        if existing is None:
            return False

        await _delete_session_file_chunks(session_id=session_id, user_id=user_id)
        await db["files"].delete_many({"session_id": session_id, "user_id": user_id})
        await db["messages"].delete_many({"session_id": session_id})

        result = await db["sessions"].delete_one({"_id": object_id, "user_id": user_id})
        if result.deleted_count == 0:
            return False

        return True
    except Exception:
        logger.exception("Failed to delete session | session_id=%s user_id=%s", session_id, user_id)
        return False

async def _delete_session_file_chunks(*, session_id: str, user_id: str) -> None:
    try:
        await QdrantChunkStore().delete_session_chunks(session_id=session_id, user_id=user_id)
    except Exception:
        logger.exception(
            "Failed to delete Qdrant file chunks | session_id=%s user_id=%s",
            session_id,
            user_id,
        )

async def add_message_to_session(session_id: str, message: ChatMessageCreate) -> dict:
    db = get_database()
    now = datetime.utcnow()
    message_dict = message.model_dump()
    message_dict.update({
        "session_id": session_id,
        "created_at": now
    })
    
    result = await db["messages"].insert_one(message_dict)
    message_dict["_id"] = str(result.inserted_id)
    
    await db["sessions"].update_one(
        {"_id": ObjectId(session_id)},
        {"$set": {"updated_at": now}}
    )
    
    return message_dict

async def add_file_ids_to_session(session_id: str, user_id: str, file_ids: List[str]) -> None:
    if not file_ids:
        return

    db = get_database()
    now = datetime.utcnow()
    await db["sessions"].update_one(
        {"_id": ObjectId(session_id), "user_id": user_id},
        {
            "$addToSet": {"file_ids": {"$each": file_ids}},
            "$set": {"updated_at": now},
        },
    )

async def get_session_file_ids(session_id: str, user_id: str) -> List[str]:
    db = get_database()
    try:
        session = await db["sessions"].find_one(
            {"_id": ObjectId(session_id), "user_id": user_id},
            {"file_ids": 1},
        )
    except Exception:
        return []

    if not session:
        return []
    return session.get("file_ids") or []

async def get_session_files(session_id: str, user_id: str) -> List[dict]:
    db = get_database()
    cursor = db["files"].find(
        {"session_id": session_id, "user_id": user_id},
        {
            "_id": 0,
            "file_id": 1,
            "filename": 1,
            "file_type": 1,
            "status": 1,
            "chunk_count": 1,
            "created_at": 1,
            "updated_at": 1,
        },
    ).sort("filename", 1)

    files = []
    async for doc in cursor:
        doc["file_type"] = doc.get("file_type") or "pdf"
        doc["status"] = doc.get("status") or "unknown"
        doc["chunk_count"] = doc.get("chunk_count") or 0
        files.append(doc)
    return files

async def get_session_messages(session_id: str) -> List[dict]:
    db = get_database()
    cursor = db["messages"].find({"session_id": session_id}).sort("created_at", 1)
    messages = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        messages.append(doc)
    return messages
